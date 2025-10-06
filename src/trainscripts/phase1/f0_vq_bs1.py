import pytorch_lightning as pl
import warnings

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ...utils import build_env, AttrDict, get_dataset_filelist

from ...models.f0_quantizer import F0Quantizer
from ...datasets.f0_bs1_dataset import F0BS1Dataset

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='f0_vq')
    parser.add_argument('--config', default='')
    parser.add_argument('--early_stopping_patience', default=99, type=int)
    parser.add_argument('--summary_interval', default=100, type=int) 
    parser.add_argument('--validation_interval', default=1000, type=int)
    a = parser.parse_args()
    return a

def get_h(a):
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

def get_dataset_loaders(h):
    # training_filelist, validation_filelist = get_dataset_filelist(h)
    trainset = F0BS1Dataset(h.f0_cache_h5_train, h.f0_stats, "train")
    validset = F0BS1Dataset(h.f0_cache_h5_valid, h.f0_stats, "valid")
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)
    return train_loader, valid_loader

def get_models(h):
    return F0Quantizer(AttrDict(h.f0_quantizer))

def get_optims(h, models):
    generator = models

    last_epoch = -1

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    
    return [optim_g], [sched_g]

class F0VQ(pl.LightningModule):
    def __init__(self, models, optims, scheds, h):
        super(F0VQ, self).__init__()
        self.generator = models
        self.optim_g = optims[0]
        self.sched_g = scheds[0]
        self.h = h
        self.lam = h.get('lambda_commit', None)

    def forward(self, batch):
        return self.generator(f0=batch)[0]

    def training_step(self, batch):
        y_g_hat, commit_and_codebook_loss, metrics = self.generator(f0=batch)
        hubert_commit_and_codebook_loss = commit_and_codebook_loss[0]

        loss_recon = F.mse_loss(y_g_hat, batch)
        loss_commit = hubert_commit_and_codebook_loss
        total = loss_recon + self.lam * loss_commit
        
        self.log("training/loss_recon", loss_recon)
        self.log("training/loss_commit", loss_commit)
        self.log("training/loss_total", total, prog_bar=True)

        return total
    
    def validation_step(self, batch):
        y_g_hat, commit_and_codebook_loss, metrics = self.generator(f0=batch)
        hubert_commit_and_codebook_loss = commit_and_codebook_loss[0]

        loss_recon = F.mse_loss(y_g_hat, batch)
        loss_commit = hubert_commit_and_codebook_loss
        total = loss_recon + self.lam * loss_commit

        self.log("validation/loss_recon", loss_recon, prog_bar=True)
        self.log("validation/loss_commit", loss_commit)
        self.log("validation/loss_total", total)

        return total
    
    def configure_optimizers(self):
        return [self.optim_g], [self.sched_g]
    
def setup_training():
    a = get_args()
    h = get_h(a)

    build_env(a.config, 'config.json', a.checkpoint_path)

    pl.seed_everything(h.seed)

    models = get_models(h)
    optims, scheds = get_optims(h, models)

    device_count = torch.cuda.device_count()
    assert h.batch_size % device_count == 0
    h.batch_size = h.batch_size // device_count

    train_loader, valid_loader = get_dataset_loaders(h)

    module = F0VQ(models, optims, scheds, h)

    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(a.checkpoint_path, "logs")
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=a.checkpoint_path,
        filename="{step}",
        save_top_k=1,
        monitor="validation/loss_recon"
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="validation/loss_recon",
        patience=a.early_stopping_patience, # under default, we run at least 100k
    )

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],

        log_every_n_steps=a.summary_interval,

        val_check_interval=a.validation_interval * h.accumulate_grad_batches, 
        check_val_every_n_epoch=None,

        max_epochs=-1,

        gradient_clip_val=1,
        gradient_clip_algorithm="norm",

        accumulate_grad_batches=h.accumulate_grad_batches

        #limit_val_batches=40 # two to three minutes-ish
    )

    trainer.fit(module, train_loader, valid_loader)

if __name__=="__main__":
    setup_training()