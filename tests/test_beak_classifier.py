from pytorch_lightning import Trainer, seed_everything
from project.beak_classifier import BeakClassifier
from project.dataset import BeakDataModule


def test_beak_classifier( backbone ):
    seed_everything(1234)
    
    # datamodule
    data_dir = "../../data/224Beaks"
    batch_size, num_workers = 8, 4
    beakDataModule = BeakDataModule( data_dir, batch_size, num_workers )

    # model
    lr = 1e-4
    beakDataModule.setup(stage="fit")
    model = BeakClassifier(backbone, lr, beakDataModule.num_classes)
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2,
            accelerator="gpu", devices=1)
    trainer.fit(model, datamodule=beakDataModule)

    beakDataModule.setup(stage="test")
    results = trainer.test(datamodule=beakDataModule)
    assert results[0]['test_acc'] > 0.5

def test_resnet50_classifier():
    test_beak_classifier( backbone='resnet50' )

def test_resnet18_classifier():
    test_beak_classifier( backbone='resnet18' )
