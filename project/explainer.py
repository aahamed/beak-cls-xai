import torch
import numpy as np
import project.utils as utils
import project.config as config
import saliency.core as saliency
import captum.attr as cattr
from captum._utils.models.linear_model import SkLearnLinearRegression, \
        SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from project.wrappers.scikit_image import SegmentationAlgorithm
from project.beak_classifier import BeakClassifier
from torchvision import transforms 

def get_conv_layer_gradcam( model ):
    return model.backbone.model[-2][-1]
    # import pdb; pdb.set_trace()
    # return model.backbone.model[-3][0].conv2

def get_explainer( xai_method, model ):
    if xai_method == 'gradcam':
        conv_layer = get_conv_layer_gradcam( model )
        return GradCamExplainer( model, conv_layer )
    elif xai_method == 'guided-gradcam':
        conv_layer = get_conv_layer_gradcam( model )
        return GuidedGradCamExplainer( model, conv_layer )
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
    elif xai_method == 'xrai':
        conv_layer = get_conv_layer_gradcam( model )
        return XraiExplainer( model, conv_layer )
    elif xai_method == 'input-x-gradients':
        return InputXGradExplainer( model )
    elif xai_method == 'gradient-shap':
        return GShapExplainer( model )
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
        # import pdb; pdb.set_trace()
        # target = torch.tensor([[target]])
        heatmap = self.method.attribute( img, target=target, **kwargs )
        heatmap = heatmap.mean( dim=1 )
        heatmap = heatmap.squeeze()
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

class GuidedGradCamExplainer( Explainer ):

    def __init__( self, model, conv_layer ):
        super().__init__( model )
        self.method = cattr.GuidedGradCam( model, conv_layer )
    
# Integrated Gradients
class IGExplainer( Explainer ):

    def __init__( self, model ):
        super( IGExplainer, self ).__init__( model )
        self.method = cattr.IntegratedGradients( model )

class SaliencyExplainer( Explainer ):

    def __init__( self, model ):
        super( SaliencyExplainer, self ).__init__( model )
        self.method = cattr.Saliency( model )

    def explain( self, img, target ):
        kwargs = {'abs':False}
        heatmap = super().explain(img, target, **kwargs)
        return heatmap

class InputXGradExplainer( Explainer ):

    def __init__( self, model ):
        super().__init__( model )
        self.method = cattr.InputXGradient(model)

class GShapExplainer( Explainer ):

    def __init__( self, model ):
        super().__init__( model )
        self.method = cattr.GradientShap(model)

    def explain( self, img, target ):
        rand_img_dist = torch.cat([img * 0, img * 1])
        heatmap = self.method.attribute( img, target=target,
            n_samples=50, stdevs=1e-4, baselines=rand_img_dist )
        heatmap = heatmap.mean( dim=1 )
        heatmap = heatmap.squeeze()
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


transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class XraiExplainer( Explainer ):

    def __init__( self, model, conv_layer ):
        super( XraiExplainer, self ).__init__( model )
        xrai_object = saliency.XRAI()
        self.xrai_params = saliency.XRAIParameters()
        self.xrai_params.algorithm = 'fast'
        self.method = xrai_object.GetMask
        self.softmax = torch.nn.Softmax(dim=1)
        self.conv_layer = conv_layer
        self.conv_layer_outputs = {}
        self.register_hooks()

    def conv_layer_forward(self, m, i, o):
        # move the RGB dimension to the last dimension
        self.conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = \
                o.permute([0,3,2,1]).detach().cpu().numpy()

    def conv_layer_backward(self, m, i, o):
        # move the RGB dimension to the last dimension
        self.conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = \
            o[0].permute([0, 3, 2, 1]).detach().cpu().numpy()

    def register_hooks(self):
        self.conv_layer.register_forward_hook(self.conv_layer_forward)
        self.conv_layer.register_backward_hook(self.conv_layer_backward)

    def preprocess_images( self, images ):
        images = np.array(images)
        images = images/255
        images = np.transpose(images, (0,3,1,2))
        images = torch.tensor(images, dtype=torch.float32)
        images = torch.stack( [ transformer( image ) for image in images ] )
        return images.requires_grad_(True)

    def call_model_function( self, images, call_model_args=None,
            expected_keys=None ):
        # import pdb; pdb.set_trace()
        images = self.preprocess_images(images).to( config.DEVICE )
        target = call_model_args[ 'target' ]
        output = self.softmax( self.model( images ) )
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            outputs = output[:,target]
            grads = torch.autograd.grad(outputs,
                    images, grad_outputs=torch.ones_like(outputs))
            grads = grads[0].permute([0,3,2,1])
            gradients = grads.detach().cpu().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            raise Exception( 'Not Supported' )

    def explain( self, imgs, targets ):
        # import pdb; pdb.set_trace()     
        assert len(imgs) == 1
        # recover original image
        orig_img = imgs[0].detach().cpu().numpy().transpose([1,2,0])
        orig_img = utils.deprocess_image( orig_img )
        # get pred
        pred = self.model( imgs ).argmax(dim=1)[0].item()
        # setup call_model_function
        call_model_args = {'target': pred}
        # Compute XRAI attributions with default parameters
        img = orig_img.astype(np.float32)
        heatmap = self.method(
                img, self.call_model_function, call_model_args,
                extra_parameters=self.xrai_params,
                batch_size=20)
        heatmap = torch.from_numpy(heatmap)
        return heatmap

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
