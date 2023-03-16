#!/bin/bash
export CUDA_VISIBLE_DEVICES

set -e

INPUT=$1 # PC_AI, BF
OUTPUT=$2 # Nucleus, TUFM, CD29
CKPT_FILE=$3
ADDITIONAL_FLAGS=$4 # "--load_from_pl --save_results_dir [DIR]
SPLIT="test"

DATA_DIR="/home/andrewbai/data/cell_img2img/${OUTPUT}_MSC_20x_${INPUT}"
SRC_DIR="${DATA_DIR}/input/${SPLIT}"
TGT_DIR="${DATA_DIR}/output/${SPLIT}"

python eval_unet.py \
  --src-dir $SRC_DIR --tgt-dir $TGT_DIR \
  --img-size 1024 --ckpt-file $CKPT_FILE $ADDITIONAL_FLAGS

