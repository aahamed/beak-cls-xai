import logging
import numpy as np
import os
import sys
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import gaussian_filter

def setup_logger( out_dir, title ):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    log_path = os.path.join(out_dir, f'{title}.log')
    if os.path.exists( log_path ): os.remove( log_path )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    log( f'Results will be stored in {out_dir}' )

def log( log_str ):
    logging.info( log_str )

def deprocess_image(img):
    """
    see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65
    """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)
    
def process_heatmap( heatmap ):
    # zero out negative values
    # heatmap = ( heatmap > 0 ) * heatmap
    heatmap = abs( heatmap )
    # smooth heatmap
    heatmap = gaussian_filter(heatmap, sigma=2)
    return heatmap

def get_com( heatmap ):
    H, W = heatmap.shape
    heatmap = process_heatmap( heatmap )
    com = center_of_mass( heatmap )
    # switch to (x,y) format
    com = [com[1]/W, com[0]/H]
    try:
        assert com[0] >= 0 and com[1] >= 0
        assert com[0] <= 1 and com[1] <= 1
    except:
        # import pdb; pdb.set_trace()
        a = 1
        com = [0, 0]
    return com
