# From Hallucination to Articulation: Language Model-Driven Losses for Ultra Low-Bitrate Neural Speech Coding

This repository complements the paper submitted to *ICASSP 2026*.

* For the listening samples and MUSHRA test configs we used, see `z_final_listening_test`.
* For replication of training etc., follow the steps below.
* To test inference, follow preprocessing steps below, download the checkpoints folder [(link)](https://drive.google.com/file/d/1N5VuWm3gR-G9MOjVXIu1exvnyPv-hZxa/view?usp=sharing) and place the whole thing in `checkpoints`, check that `checkpoints/phase3/decoder_2_5_ttr_bs1/step=692000div2_isbest.ckpt` exists.

## Preprocessing Dataset

**Prepare a venv that has editable fairseq installed. Put your editable fairseq installation directory in line 13 of `cache/features_hubert.sh`**. 

```bash
conda create -n codec python==3.10 
conda activate codec
pip install -r requirements.txt

cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2

cd ..
python lj_raw_text.py

python preprocess.py \
   --srcdir data/LJSpeech-1.1/wavs/ \
   --outdir local_preprocessed_data/LJSpeech-1.1/wavs_16khz --pad

# cache f0
bash cache/4_cache_in_shards_whole.sh
python cache/5_congregate.py
# bash cache/7_generate_statistics.sh # 2h. This comes with the repository.
```

While that runs, you can pull up another window and run the below.

```bash
# while that's running, let's start making stuff with mfa.
python cache/0_make_toalign.py
python cache/1_write_padded.py
python cache/2_manifest.py

conda activate fairseq
bash cache/3_features_ljspeech_hubert.sh
conda activate codec 

conda create -n aligner -c conda-forge montreal-forced-aligner
conda activate aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align ./bert_toalign/LJSpeech english_us_arpa english_us_arpa ./bert_aligned/LJSpeech

bash cache/6_rearrange_toalign.sh
```

### For LibriSpeech (if you wish to pretrain TTR)

```bash
cd data
wget https://openslr.trmal.net/resources/12/train-clean-100.tar.gz 
wget https://openslr.trmal.net/resources/12/train-clean-360.tar.gz 
wget https://openslr.trmal.net/resources/12/train-other-500.tar.gz
wget https://openslr.trmal.net/resources/12/test-clean.tar.gz
wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz 

for setname in "train-clean-100" "train-clean-360" "train-other-500" "test-clean" "dev-clean"
do
   mkdir $setname
   wget https://openslr.trmal.net/resources/12/${setname}.tar.gz
   tar -zxf ${setname}.tar.gz --strip-components=1 -C ${setname}
done

cd ..
python cache_libri/0_make_toalign.py

conda activate aligner
mfa align data/libri_toalign/train english_us_arpa english_us_arpa data/libri_aligned/train
mfa align data/libri_toalign/dev english_us_arpa english_us_arpa data/libri_aligned/dev
mfa align data/libri_toalign/test english_us_arpa english_us_arpa data/libri_aligned/test
```

## Pretrain TTR (Optional)

This training took about a week for us, so I recommend you to just take our checkpoint (download from the above).

```bash
scripts/ttr/summarizer_bs1.sh
```

## Phase1

Train the hubert quantizers:
```bash
bash scripts/phase1/hubert_vq_vanilla.sh 2 5
bash scripts/phase1/hubert_vq_vanilla.sh 2 6
```

and the f0 quantizer:
```bash
bash scripts/phase1/f0_vq_bs1.sh
```

Neither of these should take more than 12 hours each.

## Phase2

For each, sit back and enjoy for 200000~400000 steps.

```bash
bash scripts/phase2/decoder_2_5_bs1.sh
bash scripts/phase2/decoder_2_6_bs1.sh
```

## Phase3

For each, sit back and enjoy for 400000~600000 steps. This took a few days for us. 

By the way, the step-counting mechanism in `lightning` does not count a single call to `training_step` as one training step, which is why all the step numbers on the uploaded checkpoints are roughtly twice of this.

```bash
for brand in "asr" "ttr" "distill"
do
   mkdir -p checkpoints/phase3/decoder_2_5_${brand}_bs1
   cp checkpoints/phase2/decoder_2_5_bs1/*_isbest.ckpt checkpoints/phase3/decoder_2_5_${brand}_bs1
   bash scripts/phase3/decoder_2_5_${brand}_bs1.sh
done 

for brand in "asr" "ttr" "distill"
do
   mkdir -p checkpoints/phase3/decoder_2_6_${brand}_bs1
   cp checkpoints/phase2/decoder_2_6_bs1/*_isbest.ckpt checkpoints/phase3/decoder_2_6_${brand}_bs1
   bash scripts/phase3/decoder_2_6_${brand}_bs1.sh
done 
```

## Inference

```bash
bash scripts/inference/all.sh
bash scripts/inference/gt_maker.sh 

conda create -n whisperx python==3.10
pip install -r requirements_whisperx.txt
bash scripts/infrence/all_asr.sh

conda create -n huggingface python==3.10
pip install -r requirements_huggingface.txt
bash scripts/inference/all_asr_alternative.sh

bash scripts/inference/all_metrics.sh  
```

The audio, transcriptions and metrics go in the following locations:
* Audio: `inferred`
* Whisper-large-v3 transcriptions: `transcribed`
* wav2vec2.0 transcriptions: `transcribed_alter`
* metrics: `inferred/metrics_test`

## Listening Test

The samples and configs used are in `z_final_listening_test`. These were randomly generated by

```bash
python -m src.listening_test.maker
python -m src.listening_test.dualmono
python -m src.listening_test.confmaker_acoustic
python -m src.listening_test.confmaker_semantic
```


