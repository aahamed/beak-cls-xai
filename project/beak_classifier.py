import logging
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from project.backbones import get_backbone
from project.dataset import BeakDataModule, \
        BeakDataKfoldModule
from torchvision import models
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers

class BeakClassifier( pl.LightningModule ):
    
    def __init__(self, backbone, learning_rate=1e-3,
            num_classes=3, tune_mode="fine-tune",
            weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = get_backbone( backbone, tune_mode )
        _fc_layers = [
            nn.Linear(self.backbone.out_ch, 256), nn.ReLU(),
            nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        ]
        self.fc = nn.Sequential(*_fc_layers)
        # self.fc = nn.Linear( self.backbone.out_ch, num_classes )

    def forward( self, x ):
        x = self.backbone( x )
        out = self.fc( x )
        return out

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        pred = self.forward(imgs)
        loss = F.cross_entropy(pred, labels)
        acc = accuracy(pred, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True,
                prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        pred = self.forward(imgs)
        loss = F.cross_entropy(pred, labels)
        acc = accuracy(pred, labels)
        self.log('val_loss', loss, on_epoch=True,
                prog_bar=True)
        self.log('val_acc', acc, on_epoch=True,
                prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        pred = self.forward(imgs)
        loss = F.cross_entropy(pred, labels)
        acc = accuracy(pred, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True )
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(trainable_parameters,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--backbone', type=str, default='resnet18',
            choices=['resnet18', 'resnet50', 'vgg19', 'inception-v3'])
        parser.add_argument('--tune_mode', default='fine-tune',
                choices=['feature-extract', 'fine-tune'])
        return parser

def cli_main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_dir', default='../../data/224Beaks')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--ens_size', default=10, type=int)
    parser.add_argument('--data_method', default='full',
            choices=['full', 'kfold', 'bagging'])
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BeakClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    #pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # import pdb; pdb.set_trace()
    if args.data_method == 'full':
        beakDataModule = BeakDataModule( args.data_dir, args.batch_size,
            args.num_workers )
    elif args.data_method == 'kfold':
        beakDataModule = BeakDataKfoldModule( args.data_dir,
                args.batch_size, args.num_workers,
                args.seed%args.ens_size, args.ens_size )
    else:
        raise Exception(f'Unrecognized data method: {args.data_method}')

    # ------------
    # model
    # ------------
    beakDataModule.setup(stage="fit")
    model = BeakClassifier(args.backbone, args.learning_rate,
            beakDataModule.num_classes, args.tune_mode,
            args.weight_decay)

    # ------------
    # training
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename='best_model',
            save_last=True)

    trainer = pl.Trainer.from_argparse_args(args,
            callbacks=[checkpoint_callback])
    
    # logging
    logger = logging.getLogger("pytorch_lightning")
    log_path = os.path.join(trainer.logger.log_dir, "log.txt")
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    logger.addHandler(logging.FileHandler(log_path))

    # resume_from_checkpoint gets added by 
    # pl.Trainer.add_argparse_args
    ckpt_path = args.resume_from_checkpoint
    trainer.fit(model, datamodule=beakDataModule, ckpt_path=ckpt_path )

    # ------------
    # testing
    # ------------
    beakDataModule.setup(stage="test")
    result = trainer.test(datamodule=beakDataModule,
            ckpt_path='best')[0]
    logger.info( f"test_loss: {result['test_loss']:.2f}" )
    logger.info( f"test_acc: {result['test_acc']:.2f}" )


if __name__ == '__main__':
    cli_main()
