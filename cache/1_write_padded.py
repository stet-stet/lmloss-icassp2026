import os
import sys
import numpy as np
import soundfile as sf
import tqdm
import tqdm.contrib
import tqdm.contrib.concurrent

def do_one_file(arg):
    src, dst, at_least = arg 
    y, sr = sf.read(src)
    if y.shape[0] < at_least:
        y = np.pad(y, (0, at_least-y.shape[0]), mode='constant', constant_values=0)
    y = np.pad(y, (40, 40), mode='constant',
                          constant_values=0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, y, sr)

def do_all(src, dst, at_least=32000):
    src_fns, dst_fns, al = [], [], []
    
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                fullfn = os.path.join(root,file)
                src_fns.append(fullfn)
                destination = os.path.join(dst, os.path.relpath(fullfn, src))
                dst_fns.append(destination)
                al.append(int(at_least))

    args = list(zip(src_fns, dst_fns, al))
    tqdm.contrib.concurrent.process_map(do_one_file, args, max_workers=8, chunksize=4)

if __name__=="__main__":
    do_all("local_preprocessed_data/", "local_preprocessed_data_40/", 0)

