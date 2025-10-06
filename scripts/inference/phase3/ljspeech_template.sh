python -m src.inference.ljspeech_infer_phase3 \
    --config configs/inference/$1.json \
    --checkpoint_path checkpoints/phase3/decoder_$2/ \
    --output_dir inferred/phase3/$2/