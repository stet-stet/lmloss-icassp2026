# adapted from https://github.com/facebookresearch/speech-resynthesis
import torch.nn as nn

from .jukebox import Encoder, Decoder
from .vq import Bottleneck

class HubertQuantizer(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.code_encoder_params)
        self.vq = Bottleneck(**h.code_vq_params)
        self.decoder = Decoder(**h.code_decoder_params)

    def forward(self, **kwargs):
        code_h = self.encoder(kwargs['code'])
        _, code_h_q, code_commit_losses, code_metrics = self.vq(code_h)
        code = self.decoder(code_h_q)

        return code, code_commit_losses, code_metrics