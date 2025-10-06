# Adapted from https://github.com/aleXiehta/Timed-Text-Regularization
from ...datasets.audio_text_bs1_dataset import AudioTextBS1Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import argparse
import json
import os

from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ...utils import build_env, AttrDict

from transformers import BertTokenizer, BertModel, WavLMModel

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='summarizer')
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

class SummarizerTransformer(pl.LightningModule):
    def __init__(
        self,
        h,
        num_layers=4,
        coeff1=1.0,
        coeff2=1.0,
        verbose=False,
        freeze_ssl = True # I am hoping this is true? Please???
    ):
        super().__init__()
        self.h = h
        self.d_model = 768
        self.verbose = verbose
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.translator = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.inter = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.coeff1, self.coeff2=coeff1, coeff2
        self.cls = nn.Parameter(torch.randn(1, 1, self.d_model))

        lm_type = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(lm_type)
        self.lm = BertModel.from_pretrained(lm_type)
        self.lm.eval()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm.eval()

    def pack_to_higher_granularity(self, 
                              lower_gran_f : torch.Tensor, 
                              lengths : torch.Tensor):
        """
            lower_gran_f: tensor of shape (B, ...)
            lengths: tensor of shape (BB,)

            output: tensor of shape (BB, max(lengths), ...)
        """
        # tuple with torch.Tensor[] of (lengths[i], 768)
        the_split = torch.split(lower_gran_f, tuple(lengths), dim=0)

        ret = pad_sequence(the_split, batch_first=True)

        return ret
    
    def pack_to_lower_granularity(self,
                                  higher_gran_f : torch.Tensor,
                                  lengths : torch.Tensor):
        """
            higher_gran_f: tensor of shape (BB, T, ...)
            lengths: tensor of shape (B,). sum(lengths) == T

            output: tensor of shape (B, ...)
        """
        return torch.cat([
            higher_gran_f[i, :lengths[i], ...]
            for i in range(lengths.shape[0])
        ], dim=0)
    
    def make_inter_subword_summary(self, bert, wavlm, subwords, subwords_l, subwords_r):
        """
            bert: (B, longest, 768)
            wavlm: (B, T, 768)
            subwords: (B,); how many subwords for each?
            subwords_l: (B, longest): what is the left boundary of subword, in wavlm index?
            subwords_r: (B, longest): what is the right boundary of subword, in wavlm index?

            output (B, longest, 768) 
        """
        # congregate wavlm to subword-level
        subword_diff = subwords_r - subwords_l
        # needs major fixing.
        seqs = []
        for i in range(subwords.shape[0]):
            for j in range(subwords[i]):
                seqs.append(wavlm[i, subwords_l[i, j]:subwords_r[i, j]+1])
        packed_by_subword = pad_sequence(
            seqs
        , batch_first=True) # [BB, ?, dim]
        # subword summary
        packed_by_subword = torch.cat([self.cls.repeat(packed_by_subword.shape[0],1,1), packed_by_subword], dim=1)
        subword_summary = self.translator(packed_by_subword)[:, 0, :] # like the CLS token in BERT

        # congregate subword summary to word-level
        packed_by_word = self.pack_to_higher_granularity(subword_summary, subwords)

        # sequence modeling on subwords
        inter_subword_summary = self.inter(packed_by_word)
        return inter_subword_summary
    
    def batched_pairwise_cosine_similarity(self, x, y):
        """
            x (B, T, D) - B is #utterance
            y (B, T, D) - B is #utterance

            output (B, T, T)
        """
        x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, p=2, dim=-1, keepdim=True) + 1e-8)
        
        return torch.matmul(x, y.transpose(-2, -1))

    def make_mask_for_mse(self, length): # XXX
        """
            length (B,)

            output (B, T, T)
        """
        T = length.max()
        ret = torch.zeros(length.shape[0], T, T, device=length.device)
        for i, n in enumerate(length):
            ret[i, :n, :n] = 1.0
        return ret
    
    def masked_mse_loss(self, x, y, length):
        mask = self.make_mask_for_mse(length)
        return F.mse_loss(x*mask, y*mask, reduction='sum') / mask.sum()

    def criterion(self, bert, summary, length):
        """
            bert (B, T, 768) - utterance level batching
            summary (B, T, 768) - utterance level batching
            length (B,)

            output scalar (loss)
        """
        
        b_pairwise = self.batched_pairwise_cosine_similarity(bert, bert)
        s_pairwise = self.batched_pairwise_cosine_similarity(summary, summary)
        cos_sim = F.cosine_similarity(
            self.pack_to_lower_granularity(bert, length),
            self.pack_to_lower_granularity(summary, length), 
            dim=-1)
        if self.verbose:
            pprint(cos_sim)
        cos_dist = 1 - cos_sim.mean()

        loss = self.coeff1 * cos_dist + \
               self.coeff2 * self.masked_mse_loss(b_pairwise, s_pairwise, length)
        
        return loss

    def _audio_pad(self, y):
        return torch.nn.functional.pad(y, (40,40), 'constant', value=0)

    def preprocess_inputs(self, batch):
        """
            batch: tuple of (ret, audio)
                ret: dict with keys
                    'words': List[str]
                    'partition': List[int]; number of subwords
                    'sentence': str
                    'span': List[tuple[float, float]]
                audio: torch.Size([1, T])

            output: 
                bert: (1, L, 768)
                wavlm: (1, T, 768)
                subwords: (1,) how many subwords for each i.e. return sum of partition
                subwords_l : (1, L) left boundary of subword in wavlm index
                subwords_r : (1, L) right boundary of subword in wavlm index
        """

        ret, audio = batch
        if self.verbose:
            print([e for e in ret['words'] if len(e)>0])
        
        with torch.no_grad():
            sentence = ret['sentence']
            tokens = self.tokenizer(sentence, return_tensors='pt')
            tokens = {
                k:v.to(self.lm.device) for k,v in tokens.items()
            }
            bert = self.lm(**tokens)['last_hidden_state'][:, 1:-1, :]

            # wavlm = self.wavlm(self._audio_pad(audio),)['last_hidden_state']
        wavlm = self.wavlm(self._audio_pad(audio), output_hidden_states=True)['hidden_states'][9+1]
            # print(bert.shape, wavlm.shape, audio.shape)
            # assert round( audio.shape[1] / 320) == wavlm.shape[1] # one every 20ms (strictly)

        subwords = torch.tensor(ret['partition'], device=bert.device)
        actual_spans = []
        for n, n_subword in enumerate(subwords):
            if n_subword == 0: continue
            l, r = ret['span'][n]
            divisions = torch.linspace(float(l), float(r), n_subword.item()+1, device=bert.device)
            actual_spans.extend([
                (before, after) for before, after in zip(divisions[:-1], divisions[1:])
            ])

        subwords_l, subwords_r = [],[]
        for n in actual_spans:
            l, r = n

            left_index = torch.round( (l - 0.01) / 0.02 )
            left_index = torch.maximum(left_index, torch.zeros_like(left_index))

            right_index = torch.round( (r - 0.01) / 0.02 )
            right_index = torch.minimum(right_index, torch.zeros_like(right_index) + wavlm.shape[1] )

            subwords_l.append(left_index)
            subwords_r.append(right_index)
            # subword_to_wavlm.append( (left_index, right_index) )
        subwords_l = torch.stack(subwords_l).unsqueeze(0).long()
        subwords_r = torch.stack(subwords_r).unsqueeze(0).long()

        subwords = subwords.sum().reshape(1)

        # print(bert.shape, wavlm.shape, subwords, subwords_l, subwords_r)
        return bert, wavlm, subwords, subwords_l, subwords_r

    def typical_forward(self, batch, batch_idx, logstring):
        batch = self.preprocess_inputs(batch)

        inter_subword_summary = self.make_inter_subword_summary(*batch)
        bert, _, subwords, _, _ = batch

        loss = self.criterion( bert, inter_subword_summary, subwords)
        if logstring is not None:
            self.log(logstring, loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.typical_forward(batch, batch_idx, "training/loss")
    
    def validation_step(self, batch, batch_idx):
        return self.typical_forward(batch, batch_idx, "validation/loss")
    
    def forward(self, batch):
        return self.typical_forward(batch, None, None)
    
    def configure_optimizers(self):
        h = self.h
        optimizer = torch.optim.Adam(params=self.parameters(), 
                                     lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=h.lr_decay, last_epoch=-1)
        return [optimizer], [scheduler]
 
def get_dataset_loaders(h):
    trainset = AudioTextBS1Dataset(
        h.wav_train_root,h.textgrid_train_root, mode='train'
    )
    validset = AudioTextBS1Dataset(
        h.wav_dev_root,h.textgrid_dev_root, mode='valid'
    )
    assert h.batch_size == 1
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)
    
    # output shape: tuple of (ret, audio)
    # ret: dict with keys
    #     'words': List[str]
    #     'partition': List[int]; number of subwords
    #     'sentence': str
    #     'span': List[tuple[float, float]]
    # audio: torch.Size([1, T])
    
    return train_loader, valid_loader
    
def get_optims(h, models):
    generator = models

    last_epoch = -1

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    
    return [optim_g], [sched_g]


def setup_training():
    a = get_args()
    h = get_h(a)

    build_env(a.config, 'config.json', a.checkpoint_path)

    pl.seed_everything(h.seed)

    # optims, scheds is done within the module

    device_count = torch.cuda.device_count()
    assert h.batch_size % device_count == 0
    h.batch_size = h.batch_size // device_count

    train_loader, valid_loader = get_dataset_loaders(h)

    module = SummarizerTransformer(h) # 4 layers. This is not subject to change.

    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(a.checkpoint_path, "logs")
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=a.checkpoint_path,
        filename="{step}",
        save_top_k=1,
        monitor="validation/loss"
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="validation/loss",
        patience=a.early_stopping_patience, # under default, we run at least 100k
    )

    trainer = pl.Trainer(
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],

        log_every_n_steps=a.summary_interval,

        # val_check_interval=a.validation_interval * h.accumulate_grad_batches, 
        val_check_interval=a.validation_interval, 
        check_val_every_n_epoch=None,

        max_epochs=-1,

        gradient_clip_val=1,
        gradient_clip_algorithm="norm",

        max_steps=1000000

        # accumulate_grad_batches=h.accumulate_grad_batches
    )

    trainer.fit(module, train_loader, valid_loader)
    # trainer.fit(module, train_loader, valid_loader, ckpt_path="checkpoints/LJSpeech_seq/summarizer_libri_bs1/step=242000.ckpt") # uncomment if you wish to continue
    

if __name__ == "__main__":
    setup_training()


