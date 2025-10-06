import torch 
import torch.nn.functional as F
import h5py
import numpy as np

from ..trainscripts.env import get_h
# Here, we are not going to make any datasets; instead we directly load the caches and do everything raw.

import argparse
import re 
import os
import soundfile as sf 
from tqdm import tqdm

from ..models.ljspeech_decoder import LJSpeechDecoder

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output_dir', required=True)
    a = parser.parse_args()
    return a

def find_right_checkpoint(a, h):
    folder = a.checkpoint_path
    keyword = "_isbest"

    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".ckpt") and keyword in f]
    if len(files) > 1:
        steps = [int(re.findall(r"step=(\d+)div2", f)[0]) for f in files]
        max_step = max(steps)
        files = [f"step={max_step}div2_isbest.ckpt"]

    return os.path.join(folder, files[0])

def get_arguments():
    a = get_command_line_args()
    h = get_h(a)
    return a, h

class Helper(torch.nn.Module):
    def __init__(self, model):
        super(Helper, self).__init__()
        self.generator = model

    def forward(self, x):
        pass

    def load_into(self, fn):
        the_state_dict = torch.load(fn)["state_dict"]
        the_state_dict = {k.replace("_orig_mod.",""): v for k, v in the_state_dict.items() if k.startswith("generator.")}
        self.load_state_dict(the_state_dict)

def get_model(a, h):
    # see config and determine type.
    model = LJSpeechDecoder(h)

    load_helper = Helper(model)
    load_helper.load_into(find_right_checkpoint(a, h))
    return model.to(device)

def get_caches(h):
    ret = {}
    if "f0_h5" in h:
        ret["f0_h5"] = h5py.File(h.f0_h5)
    if "kmeans_h5" in h:
        ret["kmeans_h5"] = h5py.File(h.kmeans_h5)
    if "hubert_h5" in h:
        ret['hubert_h5'] = h5py.File(h.hubert_h5)

    assert not ('hubert_h5' in ret and 'kmeans_h5' in ret)
    assert 'hubert_h5' in ret or 'kmeans_h5' in ret

    return ret

def pack_into_tensor(t):
    t = torch.tensor(t, device=device)
    if t.shape[0] != 1:
        t = t.unsqueeze(0)
    if t.dtype == torch.float64:
        return t.to(torch.float32).to(device)
    else:
        return t.to(device)

def cut_to_right_size(f0, secondary):
    if secondary.dim() == 3 and secondary.shape[-1] == 768:
        secondary = secondary.permute(0, 2, 1)
    the_right_size = secondary.shape[-1] // 4 * 4
    the_right_f0_size = the_right_size * 4
    if f0.shape[-1]<the_right_f0_size:
        f0 = torch.nn.functional.pad(f0, (0, the_right_f0_size - f0.shape[-1]))
    return f0[..., :the_right_f0_size], secondary[..., :the_right_size]

def normalize_f0(f0, stats):
    mean, std = stats['only']
    indices_to_affect = (f0 != 0)
    f0[indices_to_affect] = (f0[indices_to_affect] - mean) / std
    return f0

def setup_inference():
    a, h = get_arguments()
    
    generator = get_model(a, h)
    caches = get_caches(h)

    f0_cache = caches["f0_h5"]
    secondary_cache = caches["hubert_h5"]

    keys = list(f0_cache.keys())
    inference_output_dir = a.output_dir
    os.makedirs(inference_output_dir, exist_ok=True)

    stats = dict(np.load(h.f0_stats, allow_pickle=True))
    generator.eval()

    for key in tqdm(keys):
        f0 = f0_cache[key]
        f0 = pack_into_tensor(f0)
        f0 = normalize_f0(f0, stats)
        secondary = secondary_cache[key]
        secondary = pack_into_tensor(secondary)
        f0, secondary = cut_to_right_size(f0, secondary)

        with torch.no_grad():
            try:
                y = generator(f0=f0, code=secondary)
                y = y.squeeze().cpu().numpy() # this is the waveform
            except NotImplementedError as e:
                print(f"Error: {e}; skipping {key}; shapes {f0.shape} {secondary.shape}")
                continue

        output_fn = os.path.join(inference_output_dir, f"{key}.wav")
        sf.write(output_fn, y, 16000)

if __name__=="__main__":
    setup_inference()

        


    




    
    
