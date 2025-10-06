if [ ! -f cache/hubert_base_ls960.pt ]; then
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -O cache/hubert_base_ls960.pt
fi

bash cache/features_hubert.sh cache/hubert_base_ls960.pt manifests_nonpad/LJSpeech/test.tsv hubert_cache_nonpad/LJSpeech/test.h5
bash cache/features_hubert.sh cache/hubert_base_ls960.pt manifests_nonpad/LJSpeech/val.tsv hubert_cache_nonpad/LJSpeech/val.h5
bash cache/features_hubert.sh cache/hubert_base_ls960.pt manifests_nonpad/LJSpeech/train.tsv hubert_cache_nonpad/LJSpeech/train.h5