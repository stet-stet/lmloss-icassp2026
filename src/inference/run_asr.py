import os
import subprocess
import sys
import whisperx
import pathlib
from tqdm import tqdm 
import numpy as np
import soundfile as sf
import torch
import argparse

device = "cuda" 
compute_type='float16'
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

def read(fn):
    audio, sr = sf.read(fn)
    # print(audio.shape)
    right_length = audio.shape[-1] // 1280 * 1280
    audio = audio[:right_length]
    # print(audio.shape)
    return audio.astype(np.float32)
    # return torch.tensor(audio,device=device).float()

def transcribe(audiofn):
    audio = read(audiofn)
    result = model.transcribe(audio, batch_size=16)
    ret = ''
    segs = result['segments']
    for seg in result['segments']:
        ret += seg['text']
    print(ret)
    return ret

def do_one_file(src, dst):
    txt = transcribe(src)
    with open(dst,'w') as file:
        file.write(txt)

def do_recursive(src, dst):
    actionlist = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                fullfile = os.path.join(root, file)
                rel = os.path.relpath(fullfile, src)
                destfile = os.path.join(dst, rel)

                actionlist.append((fullfile, destfile.replace(".wav",".txt").replace(".flac",".txt").replace("_mic2",'')))

    print(actionlist[:10])

    folders_to_make = set()
    for _, dst in actionlist:
        folders_to_make.add(os.path.dirname(dst))

    for paths in folders_to_make:
        os.makedirs(paths, exist_ok=True)

    for src, dst in tqdm(actionlist):
        do_one_file(src, dst)

def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--src", type=str, required=True)
    a.add_argument("--dst", type=str, required=True)
    return a.parse_args()

if __name__=="__main__":
    args = get_args()
    do_recursive(args.src, args.dst)
    