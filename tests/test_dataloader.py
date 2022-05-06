import torchvision.transforms as transforms
from project.dataset import BeakDataModule, BeakDataAnnModule, Annotation
from torch.utils.data import DataLoader

def test_datamodule():
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 0
    img_size = 224
    beakDataModule = BeakDataModule( data_dir, batch_size, num_workers )
    beakDataModule.setup()
    trainloader = beakDataModule.train_dataloader()
    valloader = beakDataModule.val_dataloader()
    testloader = beakDataModule.test_dataloader()
    loaders = [ trainloader, valloader, testloader ]
    loader_iters = [ iter( loader ) for loader in loaders ]
    for i in range(3):
        for loader_iter in loader_iters:
            img, label = next( loader_iter )
            assert img.shape == ( batch_size, 3, img_size, img_size )
            assert label.shape == ( batch_size, )
    print( 'Test Passed!' )

def test_ann_datamodule():
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 0
    img_size = 224
    beakDataModule = BeakDataAnnModule( data_dir, batch_size, num_workers )
    beakDataModule.setup()
    trainloader = beakDataModule.train_dataloader()
    valloader = beakDataModule.val_dataloader()
    testloader = beakDataModule.test_dataloader()
    loaders = [ trainloader, valloader, testloader ]
    loader_iters = [ iter( loader ) for loader in loaders ]
    for i in range(3):
        for loader_iter in loader_iters:
            img, label, kp = next( loader_iter )
            assert img.shape == ( batch_size, 3, img_size, img_size )
            assert label.shape == ( batch_size, )
            assert kp.shape == ( batch_size, 2 )
    print( 'Test Passed!' )
