#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES

SRC_DIR=$1 # "${DATA_DIR}/input/test"
TGT_DIR=$2 # "${DATA_DIR}/output/test"
CKPT_FILE=$3
ADDITIONAL_FLAGS=$4 # "--load_from_pl --save_results_dir [DIR]

python eval_unet.py \
  --src-dir $SRC_DIR --tgt-dir $TGT_DIR \
  --img-size 1024 --ckpt-file $CKPT_FILE $ADDITIONAL_FLAGS

