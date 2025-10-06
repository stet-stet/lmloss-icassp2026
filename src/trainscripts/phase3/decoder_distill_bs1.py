import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ljspeech_decoder_all_distill import LJSpeechDecoder
from ...datasets.cached_everything_bs1_dataset import CachedEverythingBS1Dataset
from ...loss.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from ...loss.ganstuff import generator_loss, discriminator_loss, feature_loss
from ...loss.mel import mel_spectrogram

from ..env import get_optims, get_right_batch_size, get_default_logger, find_most_recent_checkpoint, init

def get_dataset_loaders(h):
    # training_filelist, validation_filelist = get_dataset_filelist(h)
    trainset = CachedEverythingBS1Dataset(
        f0_h5_fn=h.f0_train_h5,
        f0_stats_fn=h.f0_stats,
        hubert_h5_fn=h.hubert_train_h5,
        mode='train'
    )
    validset = CachedEverythingBS1Dataset(
        f0_h5_fn=h.f0_val_h5,
        f0_stats_fn=h.f0_stats,
        hubert_h5_fn=h.hubert_val_h5,
        mode='valid'
    )
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)
    return train_loader, valid_loader

class Helper(torch.nn.Module):
    def __init__(self, model):
        super(Helper, self).__init__()
        self.generator = model

    def forward(self, x): 
        pass

    def load_into(self, fn):
        self.load_state_dict(torch.load(fn)["state_dict"])

def get_models(h):
    model = LJSpeechDecoder(h)
    
    f0_load_helper = Helper(model.f0_quantizer)
    f0_load_helper.load_into(h.f0_quantizer_path)

    hubert_load_helper = Helper(model.hubert_quantizer)
    hubert_load_helper.load_into(h.code_quantizer_path)

    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    ckpt_path = find_most_recent_checkpoint(a, h)
    if ckpt_path is not None:
        print(ckpt_path)
        state_dict = torch.load(ckpt_path)["state_dict"]
        
        mpd_state_dict = {k.replace("mpd.",''):v for k,v in state_dict.items() if k.startswith("mpd.")}
        msd_state_dict = {k.replace("msd.",''):v for k,v in state_dict.items() if k.startswith("msd.")}
        decoder_state_dict = {k.replace("generator.",''):v for k,v in state_dict.items() if k.startswith("generator.")}

        mpd.load_state_dict(mpd_state_dict)
        msd.load_state_dict(msd_state_dict)
        model.load_state_dict(decoder_state_dict)
    else:
        raise ValueError("You must provide a place to start from")

    return model, mpd, msd


class LJDecoderDistill(pl.LightningModule):
    def __init__(self, models, optims, scheds, h):
        super(LJDecoderDistill, self).__init__()
        self.generator, self.mpd, self.msd = models
        # self.generator, self.mpd, self.msd = torch.compile(self.generator), torch.compile(self.mpd), torch.compile(self.msd)
        self.optim_g, self.optim_d = optims
        self.sched_g, self.sched_d = scheds
        self.h = h
        self.automatic_optimization = False

    def configure_optimizers(self):
        return [self.optim_g, self.optim_d], [self.sched_g, self.sched_d]

    def forward(self, batch):
        the_dict, _ = batch
        return self.generator(**the_dict)
    
    def print_mpd_grads(self, string):
        for name, param in self.mpd.named_parameters():
            if "conv_post" in name and "bias" in name:
                print(string, name, param.grad)

    # modified from https://lightning.ai/docs/pytorch/stable/advanced/speed.html#model-toggling
    def training_step(self, batch, batch_idx):
        h = self.h
        optim_g, optim_d = self.optimizers()

        the_dict, y = batch
        y_g_hat, for_distill_branch, hubert_commit_losses, hubert_code_metrics = self.generator(code=the_dict['code'], f0=the_dict['f0'])

        y = y.unsqueeze(1)

        self.toggle_optimizer(optim_d)
        optim_d.zero_grad()
        y_det = y_g_hat.detach()

        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_det)
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_det)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f

        self.manual_backward(loss_disc_all)
        self.clip_gradients(optim_d, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optim_d.step()
        self.untoggle_optimizer(optim_d)

        self.toggle_optimizer(optim_g)
        optim_g.zero_grad()

        hubert_commit_and_codebook_loss = hubert_commit_losses[0] * h.commit_lambda

        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)
        y_mel = mel_spectrogram(y.squeeze(1), 
                n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate=h.sampling_rate,
                hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax_for_loss)
        
        the_code = the_dict['code']
        distill_loss = F.mse_loss(the_code, for_distill_branch) * h.hubert_distill_lambda

        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
        disc_lambda = h.get('disc_lambda', 1.0)
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_all = (loss_gen_s + loss_gen_f) * disc_lambda + distill_loss + loss_fm_s + loss_fm_f + loss_mel + hubert_commit_and_codebook_loss
        
        self.manual_backward(loss_gen_all)
        self.clip_gradients(optim_g, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optim_g.step()
        self.untoggle_optimizer(optim_g)

        # logging
        self.log("training/gen_loss_total", loss_gen_all.item())
        self.log("training/gen_mel", loss_mel.item(), prog_bar=True)
        self.log("training/gen_gan", loss_gen_all.item() - loss_mel.item())
        self.log("training/gen_commit_codebook", hubert_commit_and_codebook_loss.item())
        self.log("training/codebook_usage", hubert_code_metrics[0]['usage'].item())
        self.log("training/codebook_entropy", hubert_code_metrics[0]['entropy'].item())

        self.log("training/disc_gan", float(loss_disc_all))

    
    def validation_step(self, batch, batch_idx):
        h = self.h
        the_dict, y = batch
        y_g_hat, for_distill_branch, hubert_commit_losses, hubert_code_metrics = self.generator(**the_dict)
        hubert_commit_and_codebook_loss = hubert_commit_losses[0] * h.commit_lambda
        the_code = the_dict['code']
        distill_loss = F.mse_loss(the_code, for_distill_branch) * h.hubert_distill_lambda

        y_mel = mel_spectrogram(y, 
                                n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate=h.sampling_rate,
                                hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax)
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel).item() * 45

        loss_all = loss_mel + distill_loss + hubert_commit_and_codebook_loss

        self.log("validation/gen_mel", loss_mel, prog_bar=True, sync_dist=True)
        self.log("validation/gen_commit_codebook", hubert_commit_and_codebook_loss.item(), sync_dist=True)
        self.log("validation/gen_all", loss_all, prog_bar=True, sync_dist=True)

        # log audio?
        if batch_idx < 2:
            y = y.detach()
            y_g_hat = y_g_hat.detach()
            tb = self.logger.experiment
            tb.add_audio('gt/y_{}'.format(batch_idx), y[0], self.global_step, h.sampling_rate)
            tb.add_audio('generated/y_hat_{}'.format(batch_idx), y_g_hat[0], self.global_step, h.sampling_rate)

    def on_train_epoch_end(self):
        sched_g, sched_d = self.lr_schedulers()
        sched_g.step()
        sched_d.step()

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
        every_n_train_steps=10000 * 10 # we aint got that much space
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

if __name__=="__main__":

    a, h = init()

    pl.seed_everything(h.seed)

    models = get_models(h)
    optims, scheds = get_optims(h, models)

    h.batch_size = get_right_batch_size(h.batch_size)
    train_loader, valid_loader = get_dataset_loaders(h)

    module = LJDecoderDistill(models, optims, scheds, h)
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

    trainer.fit(module, train_loader, valid_loader, ckpt_path=None)