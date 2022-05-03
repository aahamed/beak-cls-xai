import torch
import torch.nn as nn
from torchvision import models

class Backbone( nn.Module ):

    def __init__( self, mode='fine-tune' ):
        super( Backbone, self ).__init__()
        assert mode in [ 'fine-tune', 'feature-extract' ]
        self.mode = mode

    def forward( self, x ):
        raise NotImplementedError()
    

class Resnet18( Backbone ):

    def __init__( self, mode='fine-tune' ):
        super( Resnet18, self ).__init__( mode )
        resnet = models.resnet18( pretrained=True )
        self.out_ch = resnet.fc.in_features
        # remove fc layer from resnet
        resnet = list(resnet.children())[:-1]
        self.model = nn.Sequential( *resnet )
        self.fine_tune( self.mode == 'fine-tune' )

    def forward( self, x ):
        out = self.model( x )
        return out.view( out.shape[0], out.shape[1] )
    
    def fine_tune( self, tune=True ):
        '''
        Enable fine-tuning weights if tune=True
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        
        # enable weight updates for layers 2 to 4
        for child in list( self.model.children() )[ -4: ]:
            for param in child.parameters():
                param.requires_grad = tune

class Vgg19( Backbone ):
    
    def __init__( self, mode='fine-tune' ):
        super( Vgg19, self ).__init__( mode )
        vgg = models.vgg19( pretrained=True )
        self.out_ch = vgg.classifier[-1].in_features
        # remove last fc layer
        vgg.classifier = nn.Sequential(
            *list(vgg.classifier.children())[:-1] )
        self.model = vgg
        self.fine_tune( self.mode == 'fine-tune' )
    
    def forward( self, x ):
        out = self.model( x )
        return out.squeeze()

    def fine_tune( self, tune=True ):
        # import pdb; pdb.set_trace()
        for param in self.model.parameters():
            param.requires_grad = False

        # enable weight updates
        for child in self.model.classifier.children():
            for param in child.parameters():
                param.requires_grad = tune

        for i, child in enumerate( self.model.features.children() ):
            if i < 36: continue
            for param in child.parameters():
                param.requires_grad = tune

class InceptionV3( Backbone ):

    def __init__( self, mode='fine-tune' ):
        super( InceptionV3, self ).__init__()
        inv3 = models.inception_v3( pretrained=True )
        self.out_ch = inv3.fc.in_features
        self.aux_out_ch = inv3.AuxLogits.fc.in_features
        inv3.AuxLogits.fc = nn.Identity()
        inv3.fc = nn.Identity()
        self.model = inv3
        self.fine_tune( self.mode == 'fine-tune' )

    def forward( self, x ):
        out = self.model( x )[ 0 ]
        return out

    def fine_tune( self, tune=True ):
        for param in self.model.parameters():
            param.requires_grad = False

        # enable weight updates
        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = tune

def get_backbone( backbone_str, tune_mode="feature-extract" ):
    assert backbone_str in [ 'resnet18', 'vgg19', 'inception_v3' ]
    backbone = None
    if backbone_str == 'resnet18':
        backbone = Resnet18( tune_mode )
    elif backbone_str == 'vgg19':
        backbone = Vgg19( tune_mode )
    elif backbone_str == 'inception_v3':
        backbone = InceptionV3( tune_mode )
    else:
        raise Exception( f'unrecognized backbone: {backbone_str}' )
    return backbone
