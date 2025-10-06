import numpy as np
import h5py
import random as random
import os
import json
from torch.utils.data import Dataset
from .textgrid import TextGrid
from transformers import BertTokenizer

import soundfile as sf
import os
from librosa.util import normalize
import tqdm 

MAX_WAV_VALUE = 32768.0

class AudioTextBS1Dataset(Dataset):
    """
        Dataset class for training the Subword-lvl alignment (SLA) module.
        Note that due to the massive size of the librispeech dataset, we can't pre-cache the data.
        All we can do is to load and pass everything so that the model will handle the rest.
    """
    def __init__(
        self, split_folder, split_textgrid_folder, mode="train", find_mispronounced=False
    ):
        self.split_folder = split_folder
        self.split_textgrid_folder = split_textgrid_folder
        self.mode = mode

        self.segment_len_min = 30 * 16000

        self.keys = self.get_keys(split_folder)
        self.valid_seeds = self._valid_seeds(self.keys)

        self.find_mispronounced = find_mispronounced

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_keys(self, split_folder):
        keys = []
        for root, dirs, files in os.walk(split_folder):
            if len(files) == 0 : continue
            if ".txt" in files[0] or ".wav" in files[0]:
                the_set = set([os.path.basename(file).split('.')[0] for file in files])
                keys += list(the_set)
        return sorted(keys)
    
    def _key_to_spkr(self, key):
        if '-' in key:
            return key.split('-')[0]
        else:
            return key.split('_')[0]

    def _key_to_book_and_spkr(self, key):
        if '-' in key:
            return '-'.join(key.split('-')[0:2])
        else:
            return key.split('_')[0]
    
    def _key_to_filename(self, key):
        spkr = self._key_to_spkr(key)
        return os.path.join(self.split_folder, spkr, key + ".wav")
    
    def _key_to_textgrid_tier(self, key):
        spkr = self._key_to_spkr(key)
        textgrid_fn = os.path.join(self.split_textgrid_folder, spkr, key + ".TextGrid")
        with open(textgrid_fn) as file:
            textgrid = TextGrid(file.read())
        the_tier = None
        for tier in textgrid:
            # if key == "THV_arctic_b0415": <- be sure to check this guy and replace tab with spaces.
            #     print(tier.nameid)
            if tier.nameid == "words":
                the_tier = tier
        return the_tier
    
    def _key_to_mispronounced_intervals(self, key):
        spkr = self._key_to_spkr(key)
        textgrid_fn = os.path.join(self.split_textgrid_folder, spkr, key + ".TextGrid")
        with open(textgrid_fn) as file:
            textgrid = TextGrid(file.read())
        
        the_tier = None
        for tier in textgrid:
            if tier.nameid == "phones":
                the_tier = tier

        intervals = []
        for time1, time2, phone in the_tier.simple_transcript:
            if "," in phone:
                intervals.append((float(time1), float(time2)))

        return intervals
    
    def get_token_count(self, word):
        if len(word.strip()) == 0: return 0
        tokens = self.tokenizer(word, return_tensors='pt')
        return tokens['input_ids'].shape[-1] - 2
    
    def _load_audio(self, fn):
        y, sr = sf.read(fn, dtype='int16')
        assert sr == 16000, f"Sample rate {sr} is not 16000"
        y = y / MAX_WAV_VALUE
        y = normalize(y) * 0.95
        # no padding or anything.
        return y.astype(np.float32)
    
    def _load_one(self, idx):
        ret = {}
        key = self.keys[idx]
        tier = self._key_to_textgrid_tier(key) # is tier of words

        ret['words'] = [utt for _, _, utt in tier.simple_transcript]
        ret['partition'] = [self.get_token_count(word) for word in ret['words']]
        ret['sentence'] = " ".join([e for e in ret['words'] if len(e)>0])
        ret['span'] = [(float(time1),float(time2)) for time1,time2,_ in tier.simple_transcript]
        assert len(ret['words']) == len(ret['span'])

        if self.find_mispronounced:
            ret['mispronounced'] = self._key_to_mispronounced_intervals(key)

        audio = self._load_audio(self._key_to_filename(key))

        return ret, audio

    def _valid_seeds(self, keys):
        passed = []
        for n,key in tqdm.tqdm(enumerate(keys)):
            verdict = None
            audio = self._load_audio(self._key_to_filename(key))
            audiolen = audio.shape[0]
            for m in range(n+1, len(keys)):
                if self._key_to_book_and_spkr(key) != self._key_to_book_and_spkr(keys[m]):
                    break
                audio2 = self._load_audio(self._key_to_filename(keys[m]))
                audiolen += audio2.shape[0]
                if audiolen > self.segment_len_min:
                    verdict=True
                    break
            if verdict:
                passed.append(key)
        return passed

    def _index_sampler(self, index_family=None, previous_index=None):
        if previous_index is None: 
            if self.mode == "train":
                return self.keys.index(self.valid_seeds[int(index_family)])
            else:
                return self.keys.index(self.valid_seeds[int(index_family*5)]) # magic constant, decided by the size of the validation set for LJ
        else: 
            return previous_index + 1
        
    def __len__(self):
        if self.mode == "train":
            return len(self.valid_seeds)
        else:
            if self.find_mispronounced: # l2-arctic
                return 28
            else:
                return 100 
        # return len(self.valid_seeds)

    def __getitem__(self, idx):
        sampled = self._index_sampler(index_family=idx, previous_index=None)
        ret, audio = self._load_one(sampled)
        # print(ret['mispronounced'])
        previous_index = sampled

        while audio.shape[0] < self.segment_len_min:
            sampled = self._index_sampler(index_family=idx, previous_index=previous_index)
            previous_index = sampled
            ret2, audio2 = self._load_one(sampled)

            orig_audio_len = audio.shape[0]

            audio = np.concatenate([audio, audio2], axis=0).astype(np.float32)
            ret['words'].extend(ret2['words'])
            ret['partition'].extend(ret2['partition'])
            ret['sentence'] += (" " + ret2['sentence'])
            ret['span'].extend([
                (float(a)+orig_audio_len/16000, float(b)+orig_audio_len/16000) for a,b in ret2['span']
            ])
            if self.find_mispronounced:
                ret['mispronounced'].extend([
                    (float(a)+orig_audio_len/16000, float(b)+orig_audio_len/16000) for a,b in ret2['mispronounced']
                ])

        # Finally, pad audio to the nearest multiple of 20ms;
        audio = np.pad(audio, (0, 320 - audio.shape[0] % 320), 'constant', constant_values=0)

        return ret, audio

if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    def do_print(r):
        ret, audio = r
        for key in ret:
            try: 
                print(f"{key} : shape ", ret[key].shape)
            except AttributeError:
                print(f"{key} : ", ret[key])
        print("audio : shape ", audio.shape)

    # dset = AudioTextBS1Dataset("/mnt/train1/jayeonyi/libri_toalign/dev/",
    #                            "/mnt/train1/jayeonyi/libri_aligned/dev/", 
    #                            mode="train")

    # dset = AudioTextBS1Dataset("/home/stetstet/dset/l2arctic_links/train/wav/",
    #                            "/home/stetstet/dset/l2arctic_links/train/annotation/", 
    #                            mode="train", find_mispronounced=True)
    
    dset = AudioTextBS1Dataset("/home/stetstet/dset/l2arctic_links/valid/wav/",
                               "/home/stetstet/dset/l2arctic_links/valid/annotation/", 
                               mode="valid", find_mispronounced=True)

    # for i in range(2368, 2375):
    #     print("====",i)
    #     do_print(dset[i])
    do_print(dset[0])

    # from tqdm import tqdm
    # loader = DataLoader(dset, batch_size=1, shuffle=False)
    # for n,batch in enumerate(tqdm(loader)):
    #     if n % 3000 == 0:
    #         print("====",n)
    #         do_print(batch)