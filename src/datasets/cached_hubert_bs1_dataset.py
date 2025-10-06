import torch
import h5py
import random
import numpy as np
import tqdm

class HubertCodeBS1Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_fn, mode):
        self.h5file = h5py.File(h5_fn)

        self.sampling_rate=16000
        self.hop=320
        self.mode = mode

        self.segment_len_min = 30 * 16000 / 320

        self.sorted_indexes = sorted(list(self.h5file.keys()))
        self.valid_seeds = self._valid_seeds(self.sorted_indexes)

    def _valid_seeds(self, keys):      
        """
        try concatenating the hubert segments, and see if it can reach >30s without going "out-of-the-bounds" of the passage.
        Currently supporting LJ only
        """
        passed = []
        for n,key in tqdm.tqdm(enumerate(keys)):
            verdict = None
            hubert_data = self.h5file[key][:]
            for m in range(n+1, len(keys)):
                if key.split('-')[0] != keys[m].split('-')[0]:
                    break
                hubert_data = np.concatenate([hubert_data, self.h5file[keys[m]][:]], axis=0).astype(np.float32)
                if hubert_data.shape[0] > self.segment_len_min:
                    verdict=True
                    break
            if verdict:
                passed.append(key)

        return passed
    
    def _load_one(self, idx):
        key = self.sorted_indexes[idx]
        code = np.array(self.h5file[key]).T # (768, T)

        return code

    def _index_sampler(self, index_family=None, previous_index=None):
        if previous_index is None: 
            if self.mode == "train":
                return int(index_family)
            else:
                return int(index_family*5)
        else: 
            return previous_index + 1

    def __len__(self):
        if self.mode == "train":
            return len(self.valid_seeds)
        else:
            return 100
    
    def __getitem__(self, idx):
        sampled = self._index_sampler(index_family=idx, previous_index=None)
        ret = self._load_one(sampled)
        previous_index = sampled

        while ret.shape[1] < self.segment_len_min:
            sampled = self._index_sampler(index_family=idx, previous_index=previous_index)
            previous_index = sampled
            code2 = self._load_one(sampled)

            ret = np.concatenate([ret, code2], axis=-1).astype(np.float32)

        return ret
