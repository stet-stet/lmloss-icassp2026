x=$1
bit=$2

python -m src.trainscripts.phase1.hubert_vq_bs1 \
  --config configs/phase1/hubert_${x}_${bit}.json \
  --checkpoint_path checkpoints/phase1/hubert_${x}_${bit}