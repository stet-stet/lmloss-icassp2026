import torch
import h5py
import random
import numpy as np
import soundfile as sf
import os
from librosa.util import normalize
from tqdm import tqdm
from .textgrid import TextGrid
from transformers import BertTokenizer

MAX_WAV_VALUE = 32768.0

class CachedEverythingBS1Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 f0_h5_fn, f0_stats_fn,
                 hubert_h5_fn,
                 return_text=False,
                 split_textgrid_folder=None,
                 mode='train'):
        self.f0_h5 = h5py.File(f0_h5_fn, 'r') # note that this has to be a cache file for the entire thing, not just a segment.

        self.f0_stats_file = dict(np.load(f0_stats_fn, allow_pickle=True)) # if no "dict", training crashes with OSError "Invalid Argument".

        self.hubert_h5 = h5py.File(hubert_h5_fn, 'r')

        self.segment_len_min = 30 * 16000

        self.return_text = return_text
        if return_text:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.split_textgrid_folder = split_textgrid_folder

        self.keys = sorted(list(self.f0_h5.keys()))
        self.keys = [k for k in self.keys if not self._exclude_crit(k)]
        self.valid_seeds = self._valid_seeds(self.keys)

        self.mode = mode

    def _valid_seeds(self, keys):      
        """
        try concatenating the wavlm segments, and see if it can reach >30s without going "out-of-the-bounds" of the passage.
        Currently supporting LJ only
        """
        passed = []
        for n,key in tqdm(enumerate(keys)):
            verdict = None
            hubert_data = self.hubert_h5[key][:]
            hubert_len = hubert_data.shape[0]
            for m in range(n+1, len(keys)):
                if key.split('-')[0] != keys[m].split('-')[0]:
                    break
                hubert_len += self.hubert_h5[keys[m]][:].shape[0]
                if hubert_len > self.segment_len_min / 320:
                    verdict=True
                    break
            if verdict:
                passed.append(key)

        return passed

    def _exclude_crit(self, key):
        return (key in ['p282_133', 'p345_321', 'p341_101', 'p345_292', 'p239_336'])
        
    def _key_to_spkr(self, key):
        if key.startswith("LJ"):
            return "only"
        else:
            return key.split('_')[0]
        
    def _speaker_embedding(self, key):
        return None
        
    def _key_to_filename(self, key):
        return os.path.join("local_preprocessed_data",
                            "LJSpeech-1.1",
                            "wavs_16khz",
                            key+".wav")
    
    def _key_to_textfn(self, key):
        return os.path.join("rawtext", "LJSpeech", key+".txt")
        
    def _key_to_textgrid_tier(self, key):
        spkr = self._key_to_spkr(key)
        if spkr == "only":
            textgrid_fn = os.path.join(self.split_textgrid_folder, key + ".TextGrid")
        else:
            textgrid_fn = os.path.join(self.split_textgrid_folder, spkr, key + ".TextGrid")

        with open(textgrid_fn) as file:
            textgrid = TextGrid(file.read())
        
        the_tier = None
        for tier in textgrid:
            if tier.nameid == "words":
                the_tier = tier
        return the_tier

    def get_token_count(self, word):
        if len(word.strip()) == 0: return 0
        tokens = self.tokenizer(word, return_tensors='pt')
        return tokens['input_ids'].shape[-1] - 2

    def _load_audio(self, fn):
        y, _ = sf.read(fn, dtype='int16')
        y = y / MAX_WAV_VALUE
        y = normalize(y) * 0.95
        # no padding or anything.
        return y.astype(np.float32)
    
    def _load_text(self, fn):
        with open(fn) as f:
            return f.read().strip()
    
    def _load_one(self, idx):
        origidx = idx
        key = self.keys[idx]

        audio = self._load_audio(self._key_to_filename(key))

        f0 = self.f0_h5[key][:].squeeze(1) # ( N_segments, 1, L/80 )
        speaker = self._key_to_spkr(key)
        mean, std = self.f0_stats_file[speaker]
        indices_to_affect = (f0 != 0)
        f0[indices_to_affect] = (f0[indices_to_affect] - mean) / std

        code = np.array(self.hubert_h5[key]).T # (768, T)
        
        ret = {
            "f0": f0, # ()
            "code": code,
        }
        
        if self.return_text:
            ret['text'] = self._load_text(self._key_to_textfn(key))
            tier = self._key_to_textgrid_tier(key)

            ret['words'] = [utt for _, _, utt in tier.simple_transcript]
            ret['partition'] = [self.get_token_count(word) for word in ret['words']]
            ret['sentence'] = " ".join([e for e in ret['words'] if len(e)>0])
            ret['span'] = [(float(time1),float(time2)) for time1,time2,_ in tier.simple_transcript]
        
        return ret, audio

    def __len__(self):
        if self.mode == "train":
            return len(self.valid_seeds)
        else:
            return 100 # made by concatenating utterances with i%100 == idx
        # return len(self.keys)

    def _index_sampler(self, index_family=None, previous_index=None):
        if previous_index is None: 
            if self.mode == "train":
                return self.keys.index(self.valid_seeds[int(index_family)])
            else:
                return self.keys.index(self.valid_seeds[int(index_family*5)]) # magic constant, decided by the size of the validation set for LJ
        else: 
            return previous_index + 1
        
    def __getitem__(self, idx):
        sampled = self._index_sampler(index_family=idx, previous_index=None)
        ret, audio = self._load_one(sampled)
        previous_index = sampled

        while audio.shape[0] < self.segment_len_min:
            sampled = self._index_sampler(index_family=idx, previous_index=previous_index)
            previous_index = sampled
            ret2, audio2 = self._load_one(sampled)

            audio = np.concatenate([audio, audio2], axis=-1)

            orig_hubert_len = ret['code'].shape[-1]
            for key in ret:
                # print(key, ret[key].shape, ret2[key].shape)
                if key == "subwords":
                    ret[key] += ret2[key]
                elif key == "subword_to_wavlm_l" or key == "subword_to_wavlm_r":
                    ret2[key] += orig_hubert_len
                    ret[key] = np.concatenate([ret[key], ret2[key]], axis=-1).astype(np.int32)
                elif key == "text":
                    ret[key] = ret[key] + " " + ret2[key]
                elif key == "partition" or key=="words":
                    ret[key].extend(ret2[key])
                elif key == "span":
                    orig_audio_len = orig_hubert_len * 320 # SIGHHHHH
                    ret['span'].extend([
                        (float(a)+orig_audio_len/16000, float(b)+orig_audio_len/16000) for a,b in ret2['span']
                    ])
                elif key == "sentence":
                    ret[key] = ret[key] + " " + ret2[key]
                else: # HuBERT(code), f0
                    ret[key] = np.concatenate([ret[key], ret2[key]], axis=-1).astype(np.float32)


        return ret, audio

if __name__=="__main__":
    from torch.utils.data import DataLoader
    def do_print(r):
        ret, audio = r
        for key in ret:
            try:
                ret[key].shape
                if ret[key].dim()<3:
                    print(key, ':', ret[key])
                else:
                    print(key, ':', ret[key].shape)
            except:
                print(key, ':', ret[key])
        print("audio: ", audio.shape)

    dset = CachedEverythingBS1Dataset(
        f0_h5_fn="f0_cache_whole_seq/LJSpeech/val.h5",
        f0_stats_fn="f0_stats_seq/LJSpeech/train.npz",
        hubert_h5_fn="hubert_cache_nonpad_seq/LJSpeech/val.h5",
        return_text=True
    )

    do_print(dset[0])

    loader = DataLoader(dset, batch_size=1, shuffle=False)
    for n,batch in enumerate(tqdm(loader)):
        if n % 200 == 0:
            print("====",n)
            do_print(batch)
        


        