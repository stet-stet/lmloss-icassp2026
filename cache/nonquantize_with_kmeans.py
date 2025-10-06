# Adapted from fairseq.

import argparse
import logging
import os

import numpy as np

import joblib
from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_features, get_feature_iterator
)

import h5py
from tqdm import tqdm


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_file_path",
        required=True,
        type=str,
        help="File path of output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    parser.add_argument(
        "--channel_id",
        choices=['1', '2'],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    return parser


def main(args, logger):
    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        generator, num_files = get_feature_iterator(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            sample_pct=1.0,
            channel_id=int(args.channel_id) if args.channel_id else None,
        )
        # features_batch = get_features(
        #     feature_type=args.feature_type,
        #     checkpoint_path=args.acoustic_model_path,
        #     layer=args.layer,
        #     manifest_path=args.manifest_path,
        #     sample_pct=1.0,
        #     flatten=False,
        #     channel_id=int(args.channel_id) if args.channel_id else None,
        # )
        logger.info(
            f"Iterator made for {num_files} utterances.\n"
        )

    _, fnames, _ = get_audio_files(args.manifest_path)

    os.makedirs(os.path.dirname(args.out_file_path), exist_ok=True)
    print(f"Writing the self-supervised representations to {args.out_file_path}")

    if args.channel_id is not None:
        raise NotImplementedError("we only treat channel_id == None. Sorry!")

    features_batch = generator()
    os.makedirs(os.path.dirname(args.out_file_path), exist_ok=True)
    with h5py.File(args.out_file_path, 'w') as h5file:
        for i, feats in tqdm(enumerate(features_batch), total=num_files):
            base_fname = os.path.basename(fnames[i]).split('.')[0].replace("_mic2","")
            h5file.create_dataset(base_fname, data=feats)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
