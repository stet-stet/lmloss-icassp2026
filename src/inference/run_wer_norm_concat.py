import sys
import os
from jiwer import wer
from .normalize import EnglishTextNormalizer
from jiwer import wer_standardize, wer_standardize_contiguous
import numpy as np

stan = EnglishTextNormalizer()
def eval_wer(ft, toeval):
    """
    Evaluate WER between two list of text files
    """
    assert len(ft) == len(toeval), "File lists must be of the same length"
    string_pairs = []
    for fn1, fn2 in zip(ft, toeval):
        with open(fn1) as f1, open(fn2) as f2:
            t1 = stan(f1.read().strip())
            t2 = stan(f2.read().strip())
            string_pairs.append((t1, t2))

    s1 = " ".join([pair[0] for pair in string_pairs])
    s2 = " ".join([pair[1] for pair in string_pairs])
    return wer(
        s1, s2,
        reference_transform=wer_standardize,
        hypothesis_transform=wer_standardize
    )
        
def print_content(ft, toeval):
    with open(ft) as f1:
        with open(toeval) as f2:
            t1 = f1.read().strip()
            print(t1, "|", stan(t1))
            t2 = f2.read().strip()
            print(t2, "|", stan(t2))

def do_one_folder(gt, toeval):
    actionlist = []
    files = os.listdir(toeval)
    for file in files:
        fullfile = os.path.join(toeval, file)
        gtfile = os.path.join(gt, file)
        actionlist.append( (gtfile, fullfile) )

    wer = eval_wer([a for a, _ in actionlist], [b for _, b in actionlist])
    return wer
    
def do_all(folder):
    gt = "rawtext/LJSpeech"
    others = os.listdir(folder)
    others = sorted(others)
    for name in others:
        if name == "GT": continue
        print(name)
        print(f"total wer mean for {name}:", do_one_folder(gt, os.path.join(folder, name)))

if __name__=="__main__":
    transcription_folder_name = (sys.argv[1])
    print("====LJSpeech Results (whisper)====")
    print("== gt")
    do_all(f"{transcription_folder_name}/gt")
    print("== phase2")
    do_all(f"{transcription_folder_name}/phase2")
    print("== phase3")
    do_all(f"{transcription_folder_name}/phase3")
    print("====LJSpeech Results (wav2vec2.0)====")
    print("== gt")
    do_all(f"{transcription_folder_name}_alter/gt")
    print("== phase2")
    do_all(f"{transcription_folder_name}_alter/phase2")
    print("== phase3")
    do_all(f"{transcription_folder_name}_alter/phase3")