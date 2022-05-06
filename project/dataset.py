import numpy as np
import os
import pickle
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pytorch_lightning import LightningDataModule

CLASS_TO_IDX = {
    "Small" : 0,
    "Medium" : 1,
    "Large" : 2,
}

IDX_TO_CLASS = {
    0 : "Small",
    1 : "Medium",
    2 : "Large",
}

class Annotation( object ):
    
    def __init__( self, img_file, keypoint=None, color=None ):
        self.img_file = img_file
        self.keypoint = keypoint
        self.color = color
        self.inconclusive = True if self.keypoint is None else False

class BeakData( Dataset ):

    def __init__( self, data_dir, transforms=None ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.img_paths, self.labels = self.setup()
        self.num_classes = 3

    def setup( self ):
        img_paths, labels = [], []
        for cls, idx in CLASS_TO_IDX.items():
            cls_path = os.path.join(self.data_dir, cls)
            cls_images = os.listdir( cls_path )
            for img in cls_images:
                if img.endswith(".jpg"):
                    img_paths.append( os.path.join(cls_path, img) )
                    labels.append( idx )
        return img_paths, labels

    def __getitem__( self, index ):
        label = self.labels[index]
        img = Image.open(self.img_paths[index])
        img = self.transforms(img) if self.transforms else img
        return img, label

    def __len__( self ):
        return len( self.img_paths )

class BeakDataAnn( Dataset ):

    def __init__( self, data_dir, transforms=None ):
        self.data_dir = data_dir
        self.transforms = transforms
        self.img_paths, self.labels, self.anns = self.setup()
        self.num_classes = 3
    
    def setup( self ):
        root_dir = os.path.dirname( self.data_dir )
        ann_path = os.path.join(root_dir, 'keypoints.pkl')
        img_to_ann = pickle.load( open( ann_path, 'rb' ) )
        img_paths, labels, anns = [], [], []
        for cls, idx in CLASS_TO_IDX.items():
            cls_path = os.path.join(self.data_dir, cls)
            cls_images = os.listdir( cls_path )
            for img in cls_images:
                if img.endswith(".jpg"):
                    ann = img_to_ann[img]
                    if ann.inconclusive: continue
                    img_paths.append( os.path.join(cls_path, img) )
                    labels.append( idx )
                    anns.append( ann )
        return img_paths, labels, anns

    def __getitem__( self, index ):
        label, img_path = self.labels[index], self.img_paths[index]
        img = Image.open(img_path)
        img = self.transforms(img) if self.transforms else img
        ann = self.anns[index]
        kp = np.array( list(ann.keypoint) )
        return img, label, kp 
    
    def __len__( self ):
        return len( self.img_paths )


class BeakDataModule( LightningDataModule ):

    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        img_size = ( 224, 224 )
        self.tf = transforms.Compose([
            transforms.Resize( img_size ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, "Train")
            beak_dataset = BeakData( train_dir, self.tf )
            self.num_classes = beak_dataset.num_classes
            val_len = int( 0.1 * len( beak_dataset ) )
            train_len = len( beak_dataset ) - val_len
            self.beak_train, self.beak_val = random_split( 
                    beak_dataset, [train_len, val_len],
                    generator=torch.Generator().manual_seed(42) )
        if stage == "test" or stage is None:
            test_dir = os.path.join(self.data_dir, "Test")
            self.beak_test = BeakData( test_dir, self.tf )

    def train_dataloader(self):
        return DataLoader(self.beak_train, batch_size=self.batch_size,
                num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.beak_val, batch_size=self.batch_size,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.beak_test, batch_size=self.batch_size,
                num_workers=self.num_workers)
    
def collate_wrapper( batch ):
    import pdb; pdb.set_trace()
    imgs, labels, anns = list(zip(*batch))
    imgs = torch.stack( imgs )
    labels = torch.tensor( list(labels), dtype=int )
    anns = list( anns )
    return imgs, labels, anns

class BeakDataAnnModule( BeakDataModule ):

    def __init__( self, data_dir, batch_size, num_workers):
        super().__init__( data_dir, batch_size, num_workers )


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, "Train")
            beak_dataset = BeakDataAnn( train_dir, self.tf )
            self.num_classes = beak_dataset.num_classes
            val_len = int( 0.1 * len( beak_dataset ) )
            train_len = len( beak_dataset ) - val_len
            self.beak_train, self.beak_val = random_split( 
                    beak_dataset, [train_len, val_len],
                    generator=torch.Generator().manual_seed(42) )
        if stage == "test" or stage is None:
            test_dir = os.path.join(self.data_dir, "Test")
            self.beak_test = BeakDataAnn( test_dir, self.tf )
    
    def dep_train_dataloader(self):
        return DataLoader(self.beak_train, batch_size=self.batch_size,
                num_workers=self.num_workers, shuffle=True,
                collate_fn=collate_wrapper)

    def dep_val_dataloader(self):
        return DataLoader(self.beak_val, batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_wrapper)

    def dep_test_dataloader(self):
        return DataLoader(self.beak_test, batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_wrapper)

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    load_path = '../../data/224Beaks/keypoints.pkl'
    img_to_ann = pickle.load( open(load_path, 'rb') )
