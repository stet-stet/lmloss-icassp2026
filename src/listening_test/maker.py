# script for making a pseudo listening-test, semantic and acoustic

# IMPORTANT: Run this on the root directory of the repo.
import soundfile as sf
import os
import random 
import numpy as np
import pyloudnorm as pyln

random.seed(42)

def copy_renormalize(src, dst, length=None): # length in samples
    y, sr = sf.read(src)
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(y)
    loudness_normalized_audio = pyln.normalize.loudness(y, loudness, -24.0)
    if length is not None:
        if len(y) > length:
            y = y[:length]
        else:
            y = np.pad(y, (0, length - len(y)), 'constant')
    sf.write(dst, loudness_normalized_audio, sr)
    return len(y)

def make_semantic_listening_test(count, output="z_final_listening_test/semantic/", variation_folders=[
    "inferred/phase2/2_5_bs1",
    "local_preprocessed_data/LJSpeech-1.1/wavs_16khz",
    "inferred/phase3/2_5_distill_bs1",
    "inferred/phase3/2_5_asr_bs1",
    "inferred/phase3/2_5_ttr_bs1",
    # and then for the low anchor pick another sentence 
]):
    os.makedirs(output, exist_ok=True)

    filelist = os.listdir(variation_folders[0])
    samples = random.sample(filelist, count)
    for sample in samples:
        for var_folder in variation_folders:
            name = os.path.basename(var_folder)
            src = os.path.join(var_folder, sample)
            dst = os.path.join(output, name, f"{sample}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            l = copy_renormalize(src, dst)

        original_text_fn = os.path.join("rawtext/LJSpeech", sample.replace(".wav", ".txt"))
        with open(original_text_fn) as t:
            ttt = t.read().strip()
        outfile = os.path.join(output, "text", f"{sample.replace('.wav', '.txt')}")
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as t:
            t.write(ttt)

def make_acoustic_listening_test(count, output="z_final_listening_test/acoustic/", variation_folders=[
    "inferred/phase2/2_5_bs1",
    "local_preprocessed_data/LJSpeech-1.1/wavs_16khz",
    "inferred/phase3/2_5_distill_bs1",
    "inferred/phase3/2_5_asr_bs1",
    "inferred/phase3/2_5_ttr_bs1",
]):
    os.makedirs(output, exist_ok=True)

    filelist = os.listdir(variation_folders[0])
    samples = random.sample(filelist, count)
    for sample in samples:
        for var_folder in variation_folders:
            name = os.path.basename(var_folder)
            src = os.path.join(var_folder, sample)
            dst = os.path.join(output, name, f"{sample}")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copy_renormalize(src, dst)

if __name__ == "__main__":
    make_semantic_listening_test(15)
    make_acoustic_listening_test(10)