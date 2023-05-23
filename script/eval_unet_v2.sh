#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES

ARCH=$1 # unet, msunet
SRC_DIR=$2 # "${DATA_DIR}/input/test"
TGT_DIR=$3 # "${DATA_DIR}/output/test"
CKPT_FILE=$4
ADDITIONAL_FLAGS=$5 # "--load_from_pl --save_results_dir [DIR] --n_saves 10

python eval_unet.py \
  --arch $ARCH --src-dir $SRC_DIR --tgt-dir $TGT_DIR \
  --img-size 1024 --ckpt-file $CKPT_FILE $ADDITIONAL_FLAGS

