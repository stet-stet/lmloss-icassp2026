
import argparse
import json
import re
import os
from numpy import argmax
import lightning.pytorch as pl
import torch
import itertools

from ..utils import build_env, AttrDict, get_dataset_filelist

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='hubert_vq')
    parser.add_argument('--config', default='')
    parser.add_argument('--early_stopping_patience', default=99, type=int) # 100k steps
    parser.add_argument('--summary_interval', default=10, type=int) 
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--checkpoint_interval', default=25000, type=int)
    # parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    return a

def get_h(a):
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

def init():
    a = get_args()
    h = get_h(a)

    build_env(a.config, 'config.json', a.checkpoint_path)

    return a, h

def find_most_recent_checkpoint(a, h):
    folder = a.checkpoint_path
    stepnumbercrit = re.compile(r'step=(\d+)')

    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.ckpt')]
    steps = [stepnumbercrit.findall(f) for f in files]
    steps = [int(e[0]) if len(e) > 0 else -1 for e in steps]
    if len(steps) == 0 or max(steps) == -1:
        return None
    
    return os.path.join(folder, files[argmax(steps)])

def get_right_batch_size(batch_size):
    device_count = torch.cuda.device_count()
    assert batch_size % device_count == 0
    return batch_size // device_count

def get_default_callbacks(a, h):
    # if h.validation_loss=='mel':
    #     monitor = "validation/gen_mel"
    # elif h.validation_loss == 'all':
    #     monitor = "validation/all_loss"
    monitor = "validation/gen_all"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=a.checkpoint_path,
        filename="{step}div2_isbest",
        save_top_k=1,
        monitor=monitor
    )
    checkpoint_callback_interval = pl.callbacks.ModelCheckpoint(
        dirpath=a.checkpoint_path,
        filename="{step}div2",
        save_top_k=-1,
        every_n_train_steps=a.checkpoint_interval * 2 # step counting is rather wacky. huh
    ) # and then 
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=monitor,
        patience=a.early_stopping_patience, # under default, we run at least 100k
    )
    return [
        checkpoint_callback,
        checkpoint_callback_interval,
        early_stopping
    ]

def get_default_logger(a, h):
    return pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(a.checkpoint_path, "logs")
    )

def get_optims(h, models):
    generator, mpd, msd = models

    last_epoch = -1

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    return [optim_g, optim_d], [sched_g, sched_d]



def setup_training(
        Module,
        get_models,
        get_dataset_loaders
    ):
    a, h = init()

    pl.seed_everything(h.seed)

    models = get_models(h)
    optims, scheds = get_optims(h, models)

    h.batch_size = get_right_batch_size(h.batch_size)
    train_loader, valid_loader = get_dataset_loaders(h)

    module = Module(models, optims, scheds, h)
    ckpt_path = find_most_recent_checkpoint(a, h)

    logger = get_default_logger(a, h)
    callbacks = get_default_callbacks(a, h)

    val_check_interval = a.validation_interval if "accumulate_grad_batches" not in h else a.validation_interval * h.accumulate_grad_batches

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        callbacks=callbacks,

        log_every_n_steps=a.summary_interval,

        val_check_interval=val_check_interval, 
        check_val_every_n_epoch=None,

        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_epochs=-1,
        #limit_val_batches=40 # two to three minutes-ish
    )

    trainer.fit(module, train_loader, valid_loader, ckpt_path=ckpt_path)