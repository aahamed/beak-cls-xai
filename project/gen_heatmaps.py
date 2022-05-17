import logging
import h5py
import os
import numpy as np
import sys
import project.config as config
import project.utils as utils
from argparse import ArgumentParser
from project.beak_classifier import BeakClassifier 
from project.dataset import BeakDataModule
from project.explainer import get_explainer
from torchmetrics import Accuracy

def setup_logger( out_dir, xai_method ):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    log_path = os.path.join(out_dir, f'{xai_method}.log')
    if os.path.exists( log_path ): os.remove( log_path )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    log( f'Results will be stored in {out_dir}' )

def log( log_str ):
    logging.info( log_str )

class GenHeatmaps( object ):
    
    def __init__( self, model_path, xai_method,
            data_dir, batch_size, num_workers,
            out_dir ):
        self.model = BeakClassifier.load_from_checkpoint(
            model_path,
            config.DEVICE ).to( config.DEVICE )
        self.model.eval()
        self.xai_method = xai_method
        self.dataModule = BeakDataModule( data_dir,
                batch_size, num_workers )
        self.dataModule.setup()
        self.explainer = get_explainer( self.xai_method, self.model )
        self.out_dir = out_dir
        self.h5_file = self.create_h5_file()

    def create_h5_file( self ):
        save_path = os.path.join( self.out_dir, f'{self.xai_method}.h5' )
        N = len( self.dataModule.beak_test )
        f = h5py.File( save_path, 'w' )
        f.create_dataset( 'heatmaps', (N, 224, 224), dtype=np.float32 )
        f.create_dataset( 'coms', (N, 2), dtype=np.float32 )
        return f
    
    def run( self ):
        log( f'Generating heatmaps using {self.xai_method}' )
        # import pdb; pdb.set_trace()
        all_heatmaps = self.h5_file['heatmaps']
        accuracy = Accuracy()
        # get test dataloader
        testloader = self.dataModule.test_dataloader()
        N = len( testloader )
        start, end = 0, 0
        for batch_id, (imgs, labels) in enumerate( testloader ):
            batch_size = len(imgs)
            imgs, labels = imgs.to( config.DEVICE ), labels.to( config.DEVICE )
            # get model predictions
            preds = self.model.forward( imgs ).argmax(dim=1)
            # calc acc
            batch_acc = accuracy(preds.detach().cpu(), labels.cpu())
            # generate heatmaps
            heatmaps = self.explainer.explain( imgs, preds )

            # import pdb; pdb.set_trace()
            if True not in (heatmaps>0):
                # import pdb; pdb.set_trace()
                a = 1
            # store heatmaps in h5 file
            end = start + batch_size
            all_heatmaps[start:end] = heatmaps.detach().cpu().numpy()
            start = end
            log(f'batch[{batch_id+1}/{N}] acc: {batch_acc:.2f}')
        log(f'total acc: {accuracy.compute():.2f}')

        # calculate com data in separate loop
        # in case batch_size > 1
        all_coms = self.h5_file['coms']
        H, W = all_heatmaps[0].shape
        for i, heatmap in enumerate( all_heatmaps ):
            com = utils.get_com( heatmap )
            all_coms[i] = com

        self.h5_file.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', required=True, type=str,
            help='path to model')
    parser.add_argument('--xai-method', type=str, default='gradcam',
            choices=['gradcam', 'guided-gradcam', 'integrated-gradients',
                'saliency', 'lime', 'xrai'],
            help='XAI method')
    parser.add_argument('--out-dir', type=str,
            help='Directory to store outputs')
    parser.add_argument('--data-dir', default='../../data/224Beaks',
            help='Directory containing image data')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.dirname( args.model_path )
    setup_logger( args.out_dir, args.xai_method )
    log( f'Args: {args}' )
    GenHeatmaps( args.model_path, args.xai_method, args.data_dir,
            args.batch_size, args.num_workers, args.out_dir ).run()
    log('Done!')


if __name__ == '__main__':
    main()
