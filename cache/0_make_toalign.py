import os
import shutil
import soundfile as sf
import tqdm
from tqdm.contrib.concurrent import process_map


LJSpeech_text = "rawtext/LJSpeech"
LJSpeech_wav = "local_preprocessed_data/LJSpeech-1.1/wavs_16khz"

def make_copy(list_of_srcs, destdir):
    for fn in list_of_srcs:
        shutil.copy(fn, os.path.join(destdir, os.path.basename(fn)))
        # os.symlink(fn, os.path.join(destdir, os.path.basename(fn)))

def reencode_one(args):
    src, dst = args
    y, sr = sf.read(src)
    sf.write(dst, y, sr)

def reencode(list_of_flacs, dests):
    a = tqdm.tqdm(dests)
    process_map(reencode_one, 
                [(src, dst) for src, dst in zip(list_of_flacs, dests)],
                max_workers=8,
                chunksize=1)
    
def do_LJ():
    dest = "bert_toalign/LJSpeech"
    os.makedirs(dest, exist_ok=True)
    wav_files = os.listdir(LJSpeech_wav)
    wav_files = [os.path.join(LJSpeech_wav, e) for e in wav_files]

    txt_files = os.listdir(LJSpeech_text)
    txt_files = [os.path.join(LJSpeech_text, e) for e in txt_files]

    make_copy(wav_files, dest)
    make_copy(txt_files, dest)

if __name__=="__main__":
    do_LJ()
    