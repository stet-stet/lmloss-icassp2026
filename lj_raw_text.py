import os

with open("data/LJSpeech-1.1/metadata.csv") as file:
    lines = file.readlines()
    files = [line.split("|")[0].strip() for line in lines]
    texts = [line.split("|")[-1].strip() for line in lines]
os.makedirs("rawtext/LJSpeech", exist_ok=True)

for fn, text in zip(files, texts):
    with open(os.path.join("rawtext/LJSpeech", fn+".txt"), "w") as f:
        f.write(text)