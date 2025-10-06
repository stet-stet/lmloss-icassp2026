import os 
import soundfile as sf
import numpy as np 
# a function to recursively make all the files dual-mono from a mono wav
def make_dualmono(folder):
    filelist = []
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            filelist.append(os.path.join(root, filename))

    for fn in filelist:
        if fn.endswith(".wav"):
            fullfn = fn
            y, sr = sf.read(fullfn)
            if len(y.shape) == 1:
                y = np.stack([y, y], axis=1)
                sf.write(fullfn, y, sr)
                print(f"Made dual-mono: {fullfn}")
            else:
                print(f"Already dual-mono: {fullfn}")

if __name__ == "__main__":
    make_dualmono("z_final_listening_test/")
