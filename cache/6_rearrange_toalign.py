import os 
import argparse 
import shutil 
from tqdm import tqdm

def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--orig-cache-dir", type=str, required=True)
    a.add_argument("--new-cache-dir", type=str, required=True)
    return a.parse_args()

def id_picker(key):
    if "25-" in key or "26-" in key or "27-" in key:
        return 1
    elif "21-" in key or "22-" in key or "23-" in key or "24-" in key:
        return 2
    else: 
        return 0
    
foldermap = {
    0: 'train',
    1: 'val',
    2: 'test'
}

def resplit(a):
    foldermap = {
        0: 'train',
        1: 'val',
        2: 'test'
    }
    debug = [0,0,0]
    for k,v in foldermap.items():
        os.makedirs(os.path.join(a.new_cache_dir, v), exist_ok=True)
    
    old_cache_files = os.listdir(a.orig_cache_dir)
    old_cache_files = [os.path.join(a.orig_cache_dir, f) for f in old_cache_files]
    for file in tqdm(old_cache_files):
        split_id = id_picker(file)
        new_file = os.path.join(a.new_cache_dir, foldermap[split_id], os.path.basename(file))
        # if debug[split_id] < 2:
        #     print(file, new_file)
        # debug[split_id] += 1
        print(file, new_file)
        shutil.copyfile(file, new_file)

if __name__ == "__main__":
    a = get_args()
    resplit(a)