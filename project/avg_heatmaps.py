import h5py
import os
import numpy as np
import project.utils as utils
from project.utils import setup_logger, log
from argparse import ArgumentParser

class AverageHeatmaps( object ):

    def __init__( self, heatmap_paths, out_dir, xai_method ):
        self.heatmap_paths = heatmap_paths
        self.out_dir = out_dir
        self.xai_method = xai_method
        self.h5_files = self.load_h5_files()

    def load_h5_files( self ):
        h5_files = {}
        for i, h_path in enumerate(self.heatmap_paths):
            key = f'model{i}'
            f = h5py.File(h_path, 'r')
            h5_files[key] = f

        # create h5 file for averaged heatmap
        save_path = os.path.join(self.out_dir, f'avg_{self.xai_method}.h5')
        shape = h5_files['model0']['heatmaps'].shape
        f = h5py.File( save_path, 'w' )
        f.create_dataset('heatmaps', shape, dtype=np.float32)
        N = shape[0]
        f.create_dataset('coms', (N, 2), dtype=np.float32)
        h5_files['avg_model'] = f

        return h5_files

    def close_h5_files( self ):
        for f in self.h5_files.values():
            f.close()
    
    def process_heatmap( self, heatmap ):
        # zero out negative values
        heatmap = ( heatmap > 0 ) * heatmap
        return heatmap

    def run( self ):
        log('Averaging Heatmaps')
        # import pdb; pdb.set_trace()
        N = self.h5_files['model0']['heatmaps'].shape[0]
        avg_heatmaps = self.h5_files['avg_model']['heatmaps']
        for i in range(N):
            log(f'processing heatmap [{i+1}/{N}]')
            ind_heatmaps = []
            for key, f in self.h5_files.items():
                if key == 'avg_model': continue
                ind_heatmaps.append( f['heatmaps'][i] )
            avg_heatmap = np.stack( ind_heatmaps ).mean( axis=0 )
            avg_heatmaps[i] = avg_heatmap
        
        # calculate com data
        all_coms = self.h5_files['avg_model']['coms']
        H, W = avg_heatmaps[0].shape
        for i, heatmap in enumerate( avg_heatmaps ):
            com = utils.get_com( heatmap )
            all_coms[i] = com
        self.close_h5_files()

def main():
    parser = ArgumentParser()
    parser.add_argument('--heatmap-path', nargs='+',
            required=True, help='paths to heatmaps')
    parser.add_argument('--out-dir', type=str, required=True,
            help='Directory to store outputs')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # extract xai_method from heatmap path
    # ex. lightning_logs/version0/checkpoints/gradcam.h5
    xai_method = os.path.splitext(
            args.heatmap_path[0].split('/')[-1])[0]
    setup_logger(args.out_dir, f'avg-{xai_method}')
    log(f'Args: {args}')
    AverageHeatmaps( args.heatmap_path, args.out_dir, xai_method ).run()
    log('Done!')

if __name__ == '__main__':
    main()
