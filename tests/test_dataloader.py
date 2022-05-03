import torchvision.transforms as transforms
from project.dataset import BeakData, BeakDataModule
from torch.utils.data import DataLoader

def a_test_dataloader():
    data_dir = "../../data/224Beaks/Train"
    batch_size, num_workers = 8, 0
    img_size = 224
    tf = transforms.ToTensor() 
    dataset = BeakData( data_dir )
    dataloader = DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers)
    dataset.transforms = tf
    dataloader_iter = iter( dataloader )
    for i in range(10):
        img, label = next( dataloader_iter )
        assert img.shape == ( batch_size, 3, img_size, img_size )
        assert label.shape == ( batch_size, )
    print( 'Test Passed!' )

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
