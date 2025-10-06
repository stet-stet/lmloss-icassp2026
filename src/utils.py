# Adapted from https://github.com/facebookresearch/speech-resynthesis/, which in turn adapted from https://github.com/jik876/hifi-gan

import glob
import os
import shutil

import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def parse_manifest(manifest):
    audio_files = []
    codes = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                if 'cpc_km100' in sample:
                    k = 'cpc_km100'
                elif 'vqvae256' in sample:
                    k = 'vqvae256'
                elif 'hubert' in sample:
                    k = 'hubert'
                else:
                    k = 'codes'

                codes += [torch.LongTensor(
                    [int(x) for x in sample[k].split(' ')]
                ).numpy()]
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files, codes


def get_dataset_filelist(h):
    training_files, training_codes = parse_manifest(h.input_training_file)
    validation_files, validation_codes = parse_manifest(h.input_validation_file)

    return (training_files, training_codes), (validation_files, validation_codes)