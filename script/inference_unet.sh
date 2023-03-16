#!/bin/bash
export CUDA_VISIBLE_DEVICES

set -e

INPUT=$1 # PC_AI, BF
OUTPUT=$2 # Nucleus, TUFM
CKPT_FILE="./checkpoints/$3"
RESULT_DIR="./results/$3/${INPUT}"

mkdir -p $RESULT_DIR

DATA_DIR="/home/andrewbai/data/cell_img2img/${OUTPUT}_MSC_20x_${INPUT}"
SRC_DIR="${DATA_DIR}/input/test"
TGT_DIR="${DATA_DIR}/output/test"
BSIZE="8"

python -m BNNBench.inference.ensemble_inference \
  --result-dir $RESULT_DIR \
  --test-src-dir $SRC_DIR --test-tgt-dir $TGT_DIR \
  --batch-size $BSIZE --img-size 1024 \
  --ckpt-files $CKPT_FILE

