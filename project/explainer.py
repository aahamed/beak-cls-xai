import torch
import project.utils as utils
import project.config as config
import saliency.core as saliency
import captum.attr as cattr
from captum._utils.models.linear_model import SkLearnLinearRegression, \
        SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from project.wrappers.scikit_image import SegmentationAlgorithm
from project.beak_classifier import BeakClassifier 

def get_conv_layer_gradcam( model ):
    return model.backbone.model[-2][-1]

def get_explainer( xai_method, model ):
    if xai_method == 'gradcam':
        conv_layer = get_conv_layer_gradcam( model )
        return GradCamExplainer( model, conv_layer )
    elif xai_method == 'integrated-gradients':
        return IGExplainer( model )
    elif xai_method == 'saliency':
        return SaliencyExplainer( model )
    elif xai_method == 'lime':
        exp_eucl_distance = get_exp_kernel_similarity_function(
            'euclidean', kernel_width=1000)
        return LimeExplainer( model,
                interpretable_model=SkLearnLinearRegression(),
                similarity_func=exp_eucl_distance )
    else:
        raise Exception(f'Unsupported xai-method: {xai_method}')

class Explainer( object ):
    
    def __init__( self, model ):
        '''
        Explainer base class
        '''
        self.model = model
        # method needs to be set in child class
        self.method = None

    def explain(self, img, target, **kwargs):
        '''
        Takes an input image and target class for that image and returns
        an explanation. The explanation is same resolution as img but with
        only 1 channel.
        '''
        target = torch.tensor([[target]])
        heatmap = self.method.attribute( img, target=target, **kwargs )
        heatmap = heatmap.squeeze().mean( dim=0 )
        return heatmap


class GradCamExplainer( Explainer ):

    def __init__( self, model, conv_layer ):
        '''
        GradCam based explainer
        model: model to explain
        conv_layer: Conv layer that gradcam uses to generate explanations
        '''
        super( GradCamExplainer, self ).__init__( model )
        self.method = cattr.LayerGradCam( model, conv_layer ) 

    def explain( self, img, target ):
        H, W = img.shape[2], img.shape[3]
        heatmap = self.method.attribute( img, target )
        # upsample
        heatmap = cattr.LayerAttribution.interpolate(heatmap, (H, W))
        heatmap = heatmap.squeeze()
        return heatmap

class IGExplainer( Explainer ):

    def __init__( self, model ):
        super( IGExplainer, self ).__init__( model )
        self.method = cattr.IntegratedGradients( model )

    def explain_dep( self, img, target ):
        target = torch.tensor([[target]])
        heatmap = self.method.attribute( img, target=target )
        heatmap = heatmap.squeeze().mean( dim=0 )
        return heatmap

class SaliencyExplainer( Explainer ):

    def __init__( self, model ):
        super( SaliencyExplainer, self ).__init__( model )
        self.method = cattr.Saliency( model )

    def explain_dep( self, img, target ):
        target = torch.tensor([[target]])
        heatmap = self.method.attribute( img, target=target, abs=False )
        heatmap = heatmap.squeeze().mean( dim=0 )
        return heatmap

    def explain( self, img, target ):
        kwargs = {'abs':False}
        heatmap = super().explain(img, target, **kwargs)
        return heatmap

class LimeExplainer( Explainer ):
    
    def __init__( self, model, interpretable_model, similarity_func ):
        super( LimeExplainer, self ).__init__( model )
        self.method = cattr.Lime( model,
                interpretable_model=interpretable_model,
                similarity_func=similarity_func )
        random_seed=1
        self.segmentation_fn = SegmentationAlgorithm(
                                    'quickshift',
                                    kernel_size=4,
                                    max_dist=200, ratio=0.2,
                                    random_seed=random_seed )

    def explain( self, img, target ):
        target = torch.tensor([[target]])
        # get original img before normalization
        orig_img = img[0].detach().cpu().numpy().transpose([1,2,0])
        orig_img = utils.deprocess_image(orig_img)
        # segment image
        segments = torch.tensor( 
                self.segmentation_fn(orig_img) ).to( config.DEVICE )
        # run lime
        heatmap = self.method.attribute( img, target=target,
                n_samples=1000, perturbations_per_eval=16,
                feature_mask=segments.unsqueeze(0),
                show_progress=True )
        heatmap = heatmap.squeeze().mean( dim=0 )
        return heatmap


class XraiExplainer( Explainer ):

    def __init__( self, model ):
        super( XraiExplainer, self ).__init__( model )
        xrai_object = saliency.XRAI()
        self.method = xrai_object.GetMask
        self.sigmoid = torch.nn.Sigmoid()

    def call_model_function( img, call_model_args=None,
            expected_keys=None ):
        target = call_model_args[ 'target' ]
        output = self.sigmoid( self.model( img ) )
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            raise Exception( 'Not supported' )
        else:
            raise Exception( 'Not Supported' )

    def explain( self, img, target ):
        pass

class Args:
    """
    Dummy class for testing
    """

    def __init__( self, backbone ):
        self.backbone = backbone
        self.tune_mode = 'feature-extract'
        self.learning_rate = 1e-3
        self.num_classes = 3

def get_model():
    args = Args( backbone='resnet18' )
    model = BeakClassifier(args.backbone, args.learning_rate,
            args.num_classes, args.tune_mode)
    return model
    

def test_explainer( xai_method ):
    print(f'Testing xai method: {xai_method}')
    # import pdb; pdb.set_trace()
    model = get_model()
    m_explainer = get_explainer( xai_method, model )

    # input
    batch_size, H, W = 1, 512, 512
    img = torch.randn( (batch_size, 3, H, W),
            device=config.DEVICE, requires_grad=True )
    target = 1
    heatmap = m_explainer.explain( img, target )
    assert heatmap.shape == ( H, W )
    print('Test passed!')

def test_gradcam():
    xai_method = 'gradcam'
    test_explainer( xai_method )

def test_ig():
    xai_method = 'integrated-gradients'
    test_explainer( xai_method )

def test_saliency():
    xai_method = 'saliency'
    test_explainer( xai_method )

def test_lime():
    xai_method = 'lime'
    test_explainer( xai_method ) 

def test():
    test_gradcam()
    test_ig()
    test_saliency()
    test_lime()

if __name__ == '__main__':
    test()
