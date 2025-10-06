bash scripts/inference/phase2/ljspeech_asr_alternative.sh 2_5_bs1
bash scripts/inference/phase2/ljspeech_asr_alternative.sh 2_6_bs1

for brand in "asr" "ttr" "distill"
do
    bash scripts/inference/phase3/ljspeech_asr_alternative.sh 2_5_${brand}_bs1
    bash scripts/inference/phase3/ljspeech_asr_alternative.sh 2_6_${brand}_bs1
done

python -m src.inference.run_asr_alternative \
    --src inferred/gt/$1/ \
    --dst transcribed_alter/gt/$1/