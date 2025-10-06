import os
import sys
import soundfile
import json

# first line goes the root
# and then rest is relative to that root

def do_one_file(root, fn, outfile):
    assert outfile.endswith(".tsv")

    with open(fn) as file:
        lines = file.readlines()
        lines = [e.strip() for e in lines]

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as dest:
        print(root, file=dest)
        for line in lines:
            name = line.replace("local_preprocessed_data", "local_preprocessed_data_40")
            frames = soundfile.info(name).frames
            print(
                "{}\t{}".format(os.path.relpath(name, root), frames), file=dest
            )

if __name__=="__main__":        
    for split in ["test", "val", "train"]:
        do_one_file("local_preprocessed_data_40/LJSpeech-1.1/wavs_16khz", 
                    f"split/{split}.txt",
                    f"manifests_nonpad/LJSpeech/{split}.tsv")