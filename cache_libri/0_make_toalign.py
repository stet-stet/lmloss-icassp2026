import os
import shutil
import soundfile as sf
import tqdm
from tqdm.contrib.concurrent import process_map

"""
traverse all directory
process train set folder by folder; 
- open and keep the text file
- keep tab of which speaker we are dealing with
- copy flac files (as .wav) to directory for alignment. Use directory 

"""

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
    
def one_folder(dst_root, folder):
    b = os.listdir(folder)
    to_write_audio = []
    to_write_text = []

    txtfile = [os.path.join(folder, e) for e in b if e.endswith('.txt')][0]
    print(txtfile)
    with open(txtfile) as f:
        txt = f.readlines()
        txt = [t.strip() for t in txt if t.strip() != '']

    for line in txt:
        key = line.split()[0]
        spkr = key.split('-')[0]
        text = " ".join(line.split()[1:])
        audio_fn = os.path.join(folder, key + ".flac")
        to_write_audio.append((audio_fn, os.path.join(dst_root, spkr, key + ".wav")))
        to_write_text.append((text, os.path.join(dst_root, spkr, key + ".txt")))

    os.makedirs(os.path.join(dst_root, spkr), exist_ok=True)
    reencode([e[0] for e in to_write_audio], [e[1] for e in to_write_audio])
    for text, fn in to_write_text:
        with open(fn, 'w') as f:
            f.write(text)
    
def traverse(data_root, dst_root, condition="train-"):
    list_of_folders = []
    for root, dirs, files in os.walk(data_root):
        if condition in root:
            list_of_folders.append(root)
    
    for folder in tqdm.tqdm(list_of_folders):
        try:
            one_folder(dst_root, folder)
        except IndexError:
            print("Error in folder", folder)
    
if __name__=="__main__":
    dst_roots = [
        os.path.join("data", "libri_toalign", e) for e in [
            "train", "train", "train", "dev", "test"
        ]
    ]
    data_roots = [
        os.path.join("data", e) for e in [
            "train-clean-100", "train-clean-300", "train-other-500",
            "dev-clean", "test-clean"
        ]
    ]
    conds = [
        "train-", "train-", "train-",
        "dev-clean", "test-clean"
    ]
    for data_root, dst_root, cond in zip(data_roots, dst_roots, conds):
        traverse(data_root, dst_root, condition=cond)
