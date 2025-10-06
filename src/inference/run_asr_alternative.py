import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import os
from tqdm import tqdm
import argparse 
        
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to('cuda:0')

def read(fn):
    audio, sr = sf.read(fn)
    # print(audio.shape)
    right_length = audio.shape[-1] // 1280 * 1280
    audio = audio[:right_length]
    # print(audio.shape)
    return audio.astype(np.float32), sr
    # return torch.tensor(audio,device=device).float()

def transcribe(audiofn):
    audio, sr = read(audiofn)
    input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values.to('cuda:0')).logits
        predicted_ids = torch.argmax(logits, dim=-1).detach().cpu()
    transcription = processor.decode(predicted_ids[0])
    return str(transcription)

def do_one_file(src, dst):
    txt = transcribe(src)
    with open(dst,'w') as file:
        file.write(txt)

def do_recursive(src, dst):
    actionlist = []
    for root, dirs, files in os.walk(src):
        for file in files:
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
