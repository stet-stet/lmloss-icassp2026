import numpy as np
import os
import tqdm
import json
import sys
import h5py

def combine_shards(shard_fns, output_fn):
    os.makedirs(os.path.basename(output_fn), exist_ok=True)
    output_file = h5py.File(output_fn, 'w')

    for shard_fn in shard_fns:
        shard = np.load(shard_fn, allow_pickle=True)
        for key in shard:
            output_file[key] = shard[key]

    output_file.close()

def make_lists():
    root = "f0_cache_whole"
    ret = []
    for split in ["train","test", "val"]:
        try:
            caches = os.listdir(root)
            caches = [os.path.join(root, cache) for cache in caches if cache.startswith(f"{split}")]
            ret.append(caches)
        except Exception as e:
            ret.append("")
    
    LJSpeech_tr, LJSpeech_ts, LJSpeech_v = ret
    return LJSpeech_tr, LJSpeech_ts, LJSpeech_v


if __name__=="__main__":
    LJSpeech_tr, LJSpeech_ts, LJSpeech_v = make_lists()
    combine_shards(
        LJSpeech_tr, "f0_cache_whole/train.h5"
    )
    combine_shards(
        LJSpeech_ts, "f0_cache_whole/test.h5"
    )
    combine_shards(
        LJSpeech_v, "f0_cache_whole/val.h5"
    )