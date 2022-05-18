import h5py
import os
import numpy as np
from argparse import ArgumentParser
from project.utils import setup_logger, log
from project.dataset import BeakDataAnnModule
from scipy.ndimage.measurements import center_of_mass

class EvaluateHeatmaps( object ):

    def __init__( self, heatmap_paths, avg_heatmap_path,
            out_dir, xai_method, data_dir ):
        self.heatmap_paths = heatmap_paths
        self.avg_heatmap_path = avg_heatmap_path
        self.out_dir = out_dir
        self.xai_method = xai_method
        batch_size, num_workers = 1, 2
        self.dataModule = BeakDataAnnModule(
            data_dir, batch_size, num_workers
        )
        self.dataModule.setup()
        self.h5_files = self.load_h5_files()

    def load_h5_files( self ):
        h5_files = {}
        for i, h_path in enumerate(self.heatmap_paths):
            key = f'model{i}'
            f = h5py.File(h_path, 'r')
            h5_files[key] = f
        h5_files['avg_model'] = h5py.File(self.avg_heatmap_path, 'r')
        return h5_files

    def close_h5_files( self ):
        for f in self.h5_files.values():
            f.close()

    def run( self ):
        log('Evaluating Heatmaps')
        testloader = self.dataModule.test_dataloader()
        ens_distances = []
        avg_distances = []
        N = len( testloader )
        for i, ( _, _, gt_com, valid ) in enumerate( testloader ):
            log( f'processing heatmap [{i+1}/{N}]' )
            if not valid:
                log('Skipping since gt com not valid')
                continue
            gt_com = gt_com.numpy()
            for key, f in self.h5_files.items():
                if key == 'avg_model': continue
                pred_com = f['coms'][i]
                dist = np.linalg.norm( gt_com - pred_com )
                ens_distances.append( dist )
            pred_com = self.h5_files['avg_model']['coms'][i]
            dist = np.linalg.norm( gt_com - pred_com )
            avg_distances.append( dist )
        assert len( ens_distances ) >= len( avg_distances )
        ens_mean = np.mean( ens_distances )
        avg_mean = np.mean( avg_distances )
        log( f'ens_mean: {ens_mean:.3f} avg_mean: {avg_mean:.3f}' )
        self.close_h5_files()

def main():
    parser = ArgumentParser()
    parser.add_argument('--heatmap-path', nargs='+',
            required=True, help='paths to individual heatmaps')
    parser.add_argument('--avg-heatmap-path', type=str, required=True,
            help='path to average heatmap')
    parser.add_argument('--out-dir', type=str, required=True,
            help='Directory to store outputs')
    parser.add_argument('--data-dir', default='../../data/224Beaks',
            help='Directory containing image data')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # extract xai_method from heatmap path
    # ex. lightning_logs/version0/checkpoints/gradcam.h5
    xai_method = os.path.splitext(
            args.heatmap_path[0].split('/')[-1])[0]
    setup_logger(args.out_dir, f'eval-{xai_method}')
    log(f'Args: {args}')
    EvaluateHeatmaps( args.heatmap_path,
            args.avg_heatmap_path,
            args.out_dir, xai_method,
            args.data_dir ).run()
    log('Done!')

if __name__ == '__main__':
    main()
