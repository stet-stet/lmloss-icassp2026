import torch
import h5py
import random
import tqdm
import numpy as np

class F0BS1Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_file, stats_file, mode):
        self.h5_file = h5_file
        self.h5 = h5py.File(h5_file, 'r')
        self.keys = sorted(list(self.h5.keys()))
        self.stats_file = dict(np.load(stats_file, allow_pickle=True)) # if no "dict", training crashes with OSError "Invalid Argument".
    
        self.segment_len_min = 30 * 16000 / 80 # one pitch value per 5ms = 80 samples
        self.valid_seeds = self._valid_seeds(self.keys)
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.valid_seeds)
        else:
            return 100
    
    def _valid_seeds(self, keys):
        """
        try concatenating the f0 segments, and see if it can reach >30s without going "out-of-the-bounds" of the passage.
        Supports LJ only for now
        """
        passed = []
        for n,key in tqdm.tqdm(enumerate(keys)):
            verdict = None
            f0_data = self.h5[key][:].squeeze(1) # (1, T/80)
            for m in range(n+1, len(keys)):
                if key.split('-')[0] != keys[m].split('-')[0]:
                    break
                f0_data = np.concatenate([f0_data, self.h5[keys[m]][:].squeeze(1)], axis=-1).astype(np.float32)
                if f0_data.shape[-1] > self.segment_len_min:
                    verdict=True
                    break
            if verdict:
                passed.append(key)

        return passed
    
    def _index_sampler(self, index_family=None, previous_index=None):
        if previous_index is None: 
            if self.mode == "train":
                return int(index_family)
            else:
                return int(index_family*5)
        else: 
            return previous_index + 1

    def _speaker(self, key):
        if key.startswith("LJ"):
            return "only"
        else:
            return key.split('_')[0]
        
    def _load_one(self, idx):
        key = self.keys[idx]
        f0 = np.array(self.h5[key][:]).squeeze(1) # (1, T/80)

        speaker = self._speaker(key)
        mean, std = self.stats_file[speaker]
        indices_to_affect = (f0 != 0)
        f0[indices_to_affect] = (f0[indices_to_affect] - mean) / std
        return f0 

    def __getitem__(self, idx):
        sampled = self._index_sampler(index_family=idx, previous_index=None)
        f0 = self._load_one(sampled)
        previous_index = sampled

        while f0.shape[-1] < self.segment_len_min:
            sampled = self._index_sampler(index_family=idx, previous_index=previous_index)
            previous_index = sampled
            f0 = np.concatenate([f0, self._load_one(sampled)], axis=-1).astype(np.float32)

        return f0 
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader 
    def do_print(r):
        print(r.shape)

    dset = F0BS1Dataset("f0_cache_whole_seq/LJSpeech/val.h5", "f0_stats_seq/LJSpeech/train.npz", mode="train")
    do_print(dset[0])

    from tqdm import tqdm
    loader = DataLoader(dset, batch_size=1, shuffle=False)
    for n,batch in enumerate(tqdm(loader)):
        if n % 200 == 0:
            print("====",n)
            do_print(batch)