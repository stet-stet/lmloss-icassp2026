import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

import soundfile as sf
import tqdm.contrib
import tqdm.contrib.concurrent
MAX_WAV_VALUE = 32768.0
from librosa.util import normalize
import numpy as np
import os
import tqdm
import json
import sys

def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]

    f0 = np.vstack(f0s)
    return f0

def fn_to_key(fn):
    return os.path.basename(fn).split('.')[0].replace("_mic2","")

def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    return data, sampling_rate

def get_speaker(name):
    if name.startswith("LJ"):
        return "only"
    else:
        return name.split("_")[0]

def one_file(fn):

    y, sr = load_audio(fn)
    y = y / MAX_WAV_VALUE
    y = normalize(y) * 0.95

    try:
        f0 = get_yaapt_f0(y[np.newaxis, :], rate=sr, interp=False)
        f0 = f0.flatten()
        nonzeros = f0[f0 != 0]
        return (fn_to_key(fn), nonzeros)
    except Exception as e:
        return (fn_to_key(fn), np.array([]))

def stat_all(dataset_fn, cache_fn):
    with open(dataset_fn) as file:
        lines = file.readlines()
        audio_files = [line.strip() for line in lines if line.strip()]
        # audio_files = ["local_preprocessed_data",fn for fn in audio_files]

    cache_primary = [one_file(fn) for fn in tqdm.tqdm(audio_files)]

    # cache_primary = tqdm.contrib.concurrent.process_map(one_file, audio_files, max_workers=2, chunksize=4) # this doesn't work

    cache = dict(cache_primary)

    by_speaker = {
        get_speaker(k) : [] for k in set(cache.keys())
    }

    for key in cache:
        speaker = get_speaker(key)
        by_speaker[speaker].append(cache[key])

    for speaker in by_speaker:
        by_speaker[speaker] = np.concatenate(by_speaker[speaker])

    mean_std = {
        spkr: [by_speaker[spkr].mean(), by_speaker[spkr].std()] for spkr in by_speaker
    }

    os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
    # print(mean_std)
    np.savez(cache_fn, **mean_std)

if __name__=="__main__":
    dataset = sys.argv[1]
    split = sys.argv[2]
        
    stat_all(
        f"split/{split}.txt",
        f"f0_stats/{split}.npz"
    )