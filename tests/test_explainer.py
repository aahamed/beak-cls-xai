import torch
import project.config as config
from project.explainer import get_explainer
from project.beak_classifier import BeakClassifier

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
    model.to( config.DEVICE )
    model.eval()
    return model    

def explainer_test( xai_method ):
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
    explainer_test( xai_method )

def test_guided_gradcam():
    xai_method = 'guided-gradcam'
    explainer_test( xai_method )

def test_ig():
    xai_method = 'integrated-gradients'
    explainer_test( xai_method )

def test_saliency():
    xai_method = 'saliency'
    explainer_test( xai_method )

def test_lime():
    xai_method = 'lime'
    explainer_test( xai_method ) 

#def test_xrai():
#    xai_method = 'xrai'
#    explainer_test( xai_method ) 

def test_gshap():
    xai_method = 'gradient-shap'
    explainer_test( xai_method ) 

def test_ix_grad():
    xai_method = 'input-x-gradients'
    explainer_test( xai_method ) 
