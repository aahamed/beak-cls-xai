from project.dataset import BeakDataModule, \
        BeakDataAnnModule, Annotation, \
        BeakDataKfoldModule, BeakDataBagModule
from collections import defaultdict

def test_datamodule():
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 0
    img_size = 224
    beakDataModule = BeakDataModule( data_dir, batch_size, num_workers )
    beakDataModule.setup()
    assert len( beakDataModule.beak_train ) == 5003
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

def test_bag_datamodule():
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 0
    img_size = 224
    beakDataModule = BeakDataBagModule( data_dir, batch_size, num_workers )
    beakDataModule.setup()
    assert len( beakDataModule.beak_train ) == 5003
    img_paths = beakDataModule.beak_train.dataset.img_paths
    freq = defaultdict(int)
    for path in img_paths:
        freq[path] += 1
    assert max( freq.values() ) > 1
    trainloader = beakDataModule.train_dataloader()
    valloader = beakDataModule.val_dataloader()
    loaders = [ trainloader, valloader ]
    loader_iters = [ iter( loader ) for loader in loaders ]
    for i in range(3):
        for loader_iter in loader_iters:
            img, label = next( loader_iter )
            assert img.shape == ( batch_size, 3, img_size, img_size )
            assert label.shape == ( batch_size, )
    print( 'Test Passed!' )

def test_kfold_datamodule():
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 0
    model_id, ensemble_size = 3, 10
    img_size = 224
    # datamodule for dataset with k-1 folds
    beakDataModule = BeakDataKfoldModule( 
            data_dir, batch_size, num_workers,
            model_id, ensemble_size )
    beakDataModule.setup()
    # verify dataset lengths
    len( beakDataModule.beak_train ) == 4502
    trainloader = beakDataModule.train_dataloader()
    valloader = beakDataModule.val_dataloader()
    loaders = [ trainloader, valloader ]
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
