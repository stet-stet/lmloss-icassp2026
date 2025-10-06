
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import soundfile as sf
import os
import sys
import tqdm
import json
from librosa.util import normalize
import tqdm.contrib
import tqdm.contrib.concurrent

MAX_WAV_VALUE = 32768.0

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

def load_audio(full_path):
    data, sampling_rate = sf.read(full_path, dtype='int16')
    data = data / MAX_WAV_VALUE
    data = normalize(data) * 0.95
    return data, sampling_rate

def fn_to_key(fn):
    return os.path.basename(fn).split('.')[0]

def one_file(args):
    fn = args 
    y, sr = load_audio(fn)

    try:
        f0 = get_yaapt_f0(y[np.newaxis, :], rate=sr, interp=False)
    except Exception as e:
        f0 = np.zeros((1, 1, y.shape[-1] // 80))
    
    return (fn_to_key(fn), f0)

def cache_all(dataset_fn, cache_fn, shard=None):
    with open(dataset_fn) as file:
        lines = file.readlines()
        audio_files = [line.strip() for line in lines if line.strip()]
        # lines = [json.loads(line.strip().replace("'",'"')) for line in lines]
        # audio_files = [line['audio'] for line in lines]
        # audio_files = ["local_preprocessed_data",fn for fn in audio_files]

    if shard is not None:
        audio_files = audio_files[shard[0] : shard[1]]

    args = list(audio_files)
    cache_primary = [one_file(e) for e in tqdm.tqdm(args)]
    # cache_primary = tqdm.contrib.concurrent.process_map(one_file, args, max_workers=2, chunksize=1) # this doesn't work

    cache = dict(cache_primary)
    os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
    np.savez(cache_fn, **cache)

if __name__ == "__main__":
    dataset = sys.argv[1]
    split = sys.argv[2]
    shard = [int(sys.argv[3]), int(sys.argv[4])]

    cache_all(
        f"split/{split}.txt",
        f"f0_cache_whole/{split}_{shard[0]}to{shard[1]}.npz",
        shard
    )

