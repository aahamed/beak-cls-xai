import logging
import numpy as np
import os
import sys

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
