from warpq.core import warpqMetric
from pesq import pesq
from pystoi import stoi
import soundfile as sf
import os
import sys
# from tqdm import tqdm 

mode = None
model = None
def set_mode(modestring):
    global model
    global mode
    mode = modestring
    if "warpq" in modestring:
        model = warpqMetric()

def one_file(ref, sig):
    if "warpq" in mode:
        if "raw" in mode:
            return model.evaluate(ref, sig)['raw_warpq_score']
        else:
            return model.evaluate(ref, sig)['normalized_warpq_score']
    elif mode == "pesq":
        yr, _ = sf.read(ref)
        ys, _ = sf.read(sig)
        yr = yr[:ys.shape[0]]
        ys = ys[:yr.shape[0]]
        return pesq(16000, yr, ys)
    elif mode == "stoi":
        yr, _ = sf.read(ref)
        ys, _ = sf.read(sig)
        yr = yr[:ys.shape[0]]
        ys = ys[:yr.shape[0]]
        return stoi(yr, ys, 16000)
    
def do_one_folder(gt, toeval):
    actionlist = []
    files = os.listdir(toeval)
    for file in files:
        evalfile = os.path.join(toeval, file)
        reffile = os.path.join(gt, file)
        actionlist.append( (reffile, evalfile) )

    mets = [one_file(a,b) for a,b in actionlist]
    return sum(mets)/ len(mets)

def do_all(folder):
    gtpath = "local_preprocessed_data/LJSpeech-1.1/wavs_16khz"

    others = sorted(os.listdir(folder))

    for name in others:
        print(name)
        print(do_one_folder(gtpath, os.path.join(folder, name)))

if __name__=="__main__":
    set_mode(sys.argv[1])
    inference_folder_name = (sys.argv[2])
    print("== gt ==")
    do_all(f"{inference_folder_name}/gt")
    print("== phase2 ==")
    do_all(f"{inference_folder_name}/phase2")
    print("== phase3 ==")
    do_all(f"{inference_folder_name}/phase3")