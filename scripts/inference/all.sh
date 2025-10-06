# bash scripts/inference/phase2/ljspeech_template.sh 2_5 2_5_bs1
# bash scripts/inference/phase2/ljspeech_template.sh 2_6 2_6_bs1

# for brand in "asr" "ttr" "distill"
for brand in "asr" "ttr"
do
    bash scripts/inference/phase3/ljspeech_template.sh 2_5 2_5_${brand}_bs1
    bash scripts/inference/phase3/ljspeech_template.sh 2_6 2_6_${brand}_bs1
done