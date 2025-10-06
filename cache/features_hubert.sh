
TYPE=hubert
CKPT_PATH=$1
LAYER=6
KM_MODEL_PATH=/

MANIFEST=$2
OUT_FILE=$3

# TODO: put your editably-installed fairseq path below. for me this was: 
PYTHONPATH=/home/jayeonyi/editables/fairseq/ \
python cache/nonquantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_file_path $OUT_FILE \