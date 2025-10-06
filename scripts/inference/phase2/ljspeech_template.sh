python -m src.inference.ljspeech_infer_phase2 \
    --config configs/inference/$1.json \
    --checkpoint_path checkpoints/phase2/decoder_$2/ \
    --output_dir inferred/phase2/$2/