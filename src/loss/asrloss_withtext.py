import os 
import whisper
import torch
import random
from ..inference.normalize import EnglishTextNormalizer
from pprint import pprint

class ASRLossWithText(torch.nn.Module):
    def __init__(self, verbose=False):
        super(ASRLossWithText, self).__init__()
        self.verbose = verbose

        self.model = whisper.load_model("tiny")
        alignment_heads_dense = self.model.get_buffer("alignment_heads").to_dense()
        self.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)

        self.normalizer = EnglishTextNormalizer()

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True, 
            language="en",
            task="transcribe"
        )
        
        self.initial_tokens = torch.tensor(
            [50258, 50259, 50359, 50363], dtype=torch.long
        )

        self.eot = 50237
        self.celoss = torch.nn.CrossEntropyLoss()
        if self.verbose:
            self.celoss_print = torch.nn.CrossEntropyLoss(reduction='none')
        
    def _preprocess(self, x):
        if x.dim()>2:
            x = x.squeeze(1)
        x = whisper.pad_or_trim(x)
        mel = whisper.log_mel_spectrogram(x)
        return mel # [B, 80, 3000]
        

    def forward(self, txts, x_hat):
        with torch.no_grad():
            # normalized_txts = [self.normalizer(e) for e in txts]
            normalized_txts = txts
            tokenized_txt = [
                torch.tensor(self.tokenizer.encode(e), device=x_hat.device) for e in normalized_txts
            ]
            if self.verbose:
                # print(tokenized_txt)
                toprint = [self.tokenizer.decode([token.item()]) for token in tokenized_txt[0]]

        # encode the input x first:
        mel = self._preprocess(x_hat) # [B, 80, 3000]
        encoded = self.model.encoder(mel) # [B, 1500, 384]

        token_lengths = torch.tensor([len(e) for e in tokenized_txt], device=x_hat.device)
        max_length = max([len(e) for e in tokenized_txt])
        tokenized_txt_padded = [torch.nn.functional.pad(e, (0, max_length - len(e)), mode='constant', value=self.eot) for e in tokenized_txt]

        token_inputs_to_decoder = \
            torch.zeros(len(tokenized_txt), max_length + len(self.initial_tokens) , dtype=torch.long, device=x_hat.device)
        token_inputs_to_decoder[...] = self.eot
        token_inputs_to_decoder[:, :len(self.initial_tokens)] = self.initial_tokens
        for i, e in enumerate(tokenized_txt_padded):
            token_inputs_to_decoder[i, len(self.initial_tokens):len(self.initial_tokens)+len(e)] = e

        ground_truth = torch.cat([token_inputs_to_decoder[...,1:], token_inputs_to_decoder[...,-1:]], dim=-1) #[ B, ? ]
        decoder_out = self.model.decoder(token_inputs_to_decoder, encoded) #[ B, ?, 50257 ]

        # craft the inputs to CEloss
        ground_truth = torch.concatenate([e[len(self.initial_tokens)-1:len(self.initial_tokens)-1+l] for e, l in zip(ground_truth, token_lengths)], dim=0) # concat along the ?
        decoder_out = torch.concatenate([e[len(self.initial_tokens)-1:len(self.initial_tokens)-1+l] for e, l in zip(decoder_out, token_lengths)], dim=0) # concat along the ?, have [??, 50257]

        # print(token_inputs_to_decoder.shape)
        # print(encoded.shape)
        # print(ground_truth, ground_truth.shape)
        # print(decoder_out.argmax(-1), decoder_out.shape)
        # print(self.tokenizer.decode(decoder_out.argmax(-1).tolist()))
        
        loss = self.celoss(
            decoder_out, 
            ground_truth
        )

        if self.verbose:
            with torch.no_grad():
                loss_toprint = self.celoss_print(
                    decoder_out, 
                    ground_truth
                )
            pprint( list(zip(toprint, loss_toprint)) )

        return loss

if __name__=="__main__":
    loss = ASRLossWithText().to('cuda:0')
    import soundfile as sf
    y, sr = sf.read("local_preprocessed_data/LJSpeech-1.1/wavs_16khz/LJ001-0001.wav")
    txt1 = ["Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"]
    x1 = torch.tensor(y, device='cuda:0').float().unsqueeze(0)
    yy, sr = sf.read("local_preprocessed_data/LJSpeech-1.1/wavs_16khz/LJ001-0002.wav")
    txt2 = ["in being comparatively modern."]
    x2 = torch.tensor(yy, device='cuda:0').float().unsqueeze(0)

    print(loss(txt1, x1))
    print(loss(txt2, x2))
    print(loss(txt2, x1))
    print(loss(txt1, x2))

