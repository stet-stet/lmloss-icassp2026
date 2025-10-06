# modified from https://github.com/facebookresearch/speech-resynthesis
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from ...models.jukebox import Encoder, Decoder
from ...utils import init_weights, get_padding, AttrDict

from ...models.hubert_quantizer import HubertQuantizer
from ...models.f0_quantizer import F0Quantizer

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                        padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                   padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                       padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)), k,
                                u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class LJSpeechBaselineDecoder(Generator):
    def __init__(self, h):
        super().__init__(h)
        one_speaker_token = torch.randn(1, h.embedding_dim, 1)
        self.spkr = nn.parameter.Parameter(data=one_speaker_token)

        assert 'f0_quantizer' in h
        self.f0_quantizer = F0Quantizer(AttrDict(h.f0_quantizer))
        self.f0_quantizer.eval()
        self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)
        # quantizer state loading must be done separately.
        
        self.hubert_dict = nn.Embedding(h.num_token_types, h.embedding_dim)
        # quantizer state loading must be done separately.

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):

        # kwargs['code'] # is (B, L), torch.long
        x = self.hubert_dict(kwargs['code']).transpose(1, 2) # now (B, 128, L)

        self.f0_quantizer.eval()
        f0_h = self.f0_quantizer.encoder(kwargs['f0'])
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.f0_quantizer.vq(f0_h)
        zs = [x.detach() for x in zs]
        f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
        kwargs['f0'] = f0_h_q # also [B, 128, L]

        # f0
        if x.shape[-1] < kwargs['f0'].shape[-1]:
            x = self._upsample(x, kwargs['f0'].shape[-1])
        else:
            kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
        x = torch.cat([x, kwargs['f0']], dim=1) # [B, 128 * 2, L]

        spkr = self.spkr.expand(x.shape[0], -1, -1)
        spkr = self._upsample(spkr, x.shape[-1])
        x = torch.cat([x, spkr], dim=1) # [B, 128 * 3, L]

        # and then upsamples to wav
        return super().forward(x)
    

class LJSpeechDecoder(Generator): # MODIFIED.
    def __init__(self, h):
        super().__init__(h)
        one_speaker_token = torch.randn(1, h.embedding_dim, 1)
        self.spkr = nn.parameter.Parameter(data=one_speaker_token)

        assert 'f0_quantizer' in h
        self.f0_quantizer = F0Quantizer(AttrDict(h.f0_quantizer))
        self.f0_quantizer.eval()
        self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)
        # quantizer state loading must be done separately.
        
        assert 'code_quantizer' in h
        self.hubert_quantizer = HubertQuantizer(AttrDict(h.code_quantizer))
        # self.hubert_quantizer.eval()
        self.hubert_dict = nn.Embedding(h.code_quantizer["code_vq_params"]['l_bins'], h.embedding_dim)
        # quantizer state loading must be done separately.
    
    def load_f0_quantizer(self, f0_state_dict):
        self.f0_quantizer.load_state_dict(f0_state_dict)

    def load_hubert_quantizer(self, hubert_state_dict):
        self.hubert_quantizer.load_state_dict(hubert_state_dict)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):

        # self.hubert_quantizer.eval()
        x_h = self.hubert_quantizer.encoder(kwargs['code'])
        x_h = [x for x in x_h]
        _, quantized_activations, hubert_commit_losses, hubert_code_metrics = self.hubert_quantizer.vq(x_h)
        for_distillation_branch = self.hubert_quantizer.decoder(quantized_activations)
        # q_h = [x.detach() for x in q_h]
        # x_h_q = self.hubert_dict(q_h[0].detach()).transpose(1, 2)
        x = quantized_activations[0] # this is [B, 128, 7]

        self.f0_quantizer.eval()
        f0_h = self.f0_quantizer.encoder(kwargs['f0'])
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.f0_quantizer.vq(f0_h)
        zs = [x.detach() for x in zs]
        f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
        kwargs['f0'] = f0_h_q # also [B, 128, L]

        # f0
        if x.shape[-1] < kwargs['f0'].shape[-1]:
            x = self._upsample(x, kwargs['f0'].shape[-1])
        else:
            kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
        x = torch.cat([x, kwargs['f0']], dim=1) # [B, 128 * 2, L]

        spkr = self.spkr.expand(x.shape[0], -1, -1)
        spkr = self._upsample(spkr, x.shape[-1])
        x = torch.cat([x, spkr], dim=1) # [B, 128 * 3, L]

        # and then upsamples to wav
        return super().forward(x), for_distillation_branch, hubert_commit_losses, hubert_code_metrics
    
class LJSpeechVariableRateDecoder(Generator):
    def __init__(self, h):
        super().__init__(h)
        one_speaker_token = torch.randn(1, h.embedding_dim, 1)
        self.spkr = nn.parameter.Parameter(data=one_speaker_token)

        assert 'f0_quantizer' in h
        self.f0_quantizer = F0Quantizer(AttrDict(h.f0_quantizer))
        self.f0_quantizer.eval()
        self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)
        # quantizer state loading must be done separately.
        
        assert 'code_quantizer' in h
        self.hubert_quantizer = HubertQuantizer(AttrDict(h.code_quantizer))
        self.hubert_quantizer.eval()
        self.hubert_dict = nn.Embedding(h.code_quantizer["code_vq_params"]['l_bins'], h.embedding_dim)
        # quantizer state loading must be done separately.

        self.the_mask = nn.Parameter(torch.zeros(h.embedding_dim*2, 1))
        self.inner_downsampling_factor = 2 ** int(AttrDict(h.inner_downsampler_params).downs_t[0])
        self.inner_downsampler = Encoder(
            **AttrDict(h.inner_downsampler_params)
        )
        self.linear_to_determine_mask_loc = nn.Linear(h.embedding_dim * 2, self.inner_downsampling_factor)
        self.softmax = nn.Softmax(dim=1)
    
    def load_f0_quantizer(self, f0_state_dict):
        self.f0_quantizer.load_state_dict(f0_state_dict)

    def load_hubert_quantizer(self, hubert_state_dict):
        self.hubert_quantizer.load_state_dict(hubert_state_dict)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal
    
    def mask(self, x, **kwargs):
        # x is [B, 128*3, L]
        B, three_times_embdim, L = x.shape
        # pad to correct length
        I = self.inner_downsampling_factor
        xc = torch.nn.functional.pad(x, (0, (I-L%I)%I), 'constant', 0) # [B, 128*3, L']
        x_ds = self.inner_downsampler(xc[:, :self.h.embedding_dim*2, :])[0] # [B, 128*2, L' //I]
        x_ds = self.linear_to_determine_mask_loc(x_ds.permute(0,2,1)).permute(0,2,1) # [B, I, L' //I]
        x_ds = self.softmax(x_ds) #[B, I, L'//I]
        maskdim = self.h.embedding_dim * 2
        if self.training:
            alpha = kwargs['step'] / 25000
            x_ds = torch.exp( - alpha * x_ds) # [B, I, L'//I], range in [0, 1]
            mask = x_ds.permute(0,2,1).reshape(B, 1, -1) # [B, 1, L'//I * I = L']
            # print(x.shape, x_ds.shape, xc.shape, mask.shape, self.the_mask.shape)
            ret = torch.cat((xc[:, :maskdim, :] * (1-mask) + (self.the_mask * mask), xc[:, maskdim:, :]), dim=1)
        else:
            the_indices = torch.argmax(x_ds, dim=1) # [B, L'//I]
            the_indices = the_indices + torch.arange(the_indices.shape[-1], device=x_ds.device)*I
            batch_indices = torch.arange(the_indices.shape[0], device=x_ds.device).reshape(-1, 1).expand(-1, the_indices.shape[-1]).reshape(-1)
            the_indices = the_indices.reshape(-1)
            xc[batch_indices, :maskdim, the_indices] = self.the_mask.squeeze()
            ret = xc
        return ret[:, :, :L]

    def forward(self, **kwargs):

        self.hubert_quantizer.eval()
        x_h = self.hubert_quantizer.encoder(kwargs['code'])
        x_h = [x.detach() for x in x_h]
        q_h, _, _, _ = self.hubert_quantizer.vq(x_h)
        q_h = [x.detach() for x in q_h]
        x_h_q = self.hubert_dict(q_h[0].detach()).transpose(1, 2)
        x = x_h_q # this is [B, 128, 7]

        self.f0_quantizer.eval()
        f0_h = self.f0_quantizer.encoder(kwargs['f0'])
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.f0_quantizer.vq(f0_h)
        zs = [x.detach() for x in zs]
        f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
        kwargs['f0'] = f0_h_q # also [B, 128, L]

        # f0
        if x.shape[-1] < kwargs['f0'].shape[-1]:
            x = self._upsample(x, kwargs['f0'].shape[-1])
        else:
            kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
        x = torch.cat([x, kwargs['f0']], dim=1) # [B, 128 * 2, L]

        spkr = self.spkr.expand(x.shape[0], -1, -1)
        spkr = self._upsample(spkr, x.shape[-1])
        x = torch.cat([x, spkr], dim=1) # [B, 128 * 3, L]

        # put through the network that downsamples by four & mask
        x = self.mask(x, **kwargs)


        # and then upsamples to wav
        return super().forward(x)