import csv
import json
import os
import pickle
from argparse import ArgumentParser

class Annotation( object ):
    
    def __init__( self, img_file, keypoint=None, color=None ):
        self.img_file = img_file
        self.keypoint = keypoint
        self.color = color
        self.inconclusive = True if self.keypoint is None else False
        
def read_ann( ann_file ):
    print(f'Parsing keypoint annotations in {ann_file}')
    img_to_ann = {}
    with open(ann_file, 'r') as csvfile:
        
        datareader = csv.reader(csvfile, delimiter=',')
        for i, line in enumerate(datareader):
            if i == 0: continue
            # img_file = '/'.join( line[0].split('/')[-2:] )
            img_file = '/'.join( line[0].split('/')[-1:] )
            try:
                status = json.loads(line[3])
                kp = ( status[0]['x'], status[0]['y'] )
                color = status[0]['label']
            except:
                kp, color = None, None
            img_to_ann[img_file] = Annotation(img_file, kp, color)
    return img_to_ann

def main(): 
    parser = ArgumentParser()
    parser.add_argument('--ann-path', required=True, type=str,
            help='path to keypoint annotations')
    parser.add_argument('--out-dir', type=str,
            help='Directory to store outputs')
    args = parser.parse_args()
    print(f'Args: {args}')
    img_to_ann = read_ann(args.ann_path)
    args.out_dir = '.' if args.out_dir is None else args.out_dir
    save_path = os.path.join(args.out_dir, 'keypoints.pkl')
    pickle.dump(img_to_ann, open(save_path, 'wb'))
    print('Done!')


if __name__ == '__main__':
    main()
