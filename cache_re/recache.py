import h5py
import os 
import argparse 

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

def resplit(a):
    """
    for all indexes in the old cache,
    we allocate passages 25, 26, 27 to valid
    and 21, 22, 23, 24 to test
    """
    os.makedirs(a.new_cache_dir, exist_ok=True)
    old_cache_contents = [
        h5py.File(os.path.join(a.orig_cache_dir, f"{x}.h5"), 'r') for x in ['train', 'val', 'test']
    ]
    new_cache_contents = [
        h5py.File(os.path.join(a.new_cache_dir, f"{x}.h5"), 'w') for x in ['train', 'val', 'test']
    ]
    if "partition" in old_cache_contents[0]:
        for i in range(3):
            new_cache_contents[i].create_group('partition')
            new_cache_contents[i].create_group('span')
            new_cache_contents[i].create_group('representation')
        all_keys, where = [],[]
        for c, name in zip(old_cache_contents, ['train', 'val', 'test']):
            all_keys.extend(list(c['partition'].keys()))
            where.extend([name for _ in list(c['partition'].keys())])    

        print(len(all_keys), len(where))

        for key, name in zip(all_keys, where):
            old_id = ['train', 'val', 'test'].index(name)
            new_id = id_picker(key)
            for g in ['partition', 'span', 'representation']:
                old_cache_contents[old_id][g][key][:]
                new_cache_contents[new_id][g][key] = old_cache_contents[old_id][g][key][:]
    else:
        all_keys, where = [],[]
        for c, name in zip(old_cache_contents, ['train', 'val', 'test']):
            all_keys.extend(list(c.keys()))
            where.extend([name for _ in list(c.keys())])    
        for key, name in zip(all_keys, where):
            old_id = ['train', 'val', 'test'].index(name)
            new_id = id_picker(key)
            new_cache_contents[new_id][key] = old_cache_contents[old_id][key][:]

if __name__ == "__main__":
    resplit(get_args())
    
    