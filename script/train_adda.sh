#/bin/bash
export CUDA_VISIBLE_DEVICES

set -e

SRC_A=$1
SRC_B=$2
TGT=$3
KWARGS=$4 # --max_epochs 200 --save_every_n_epochs 20, --max_B_size 20, --name ntrn-100

UNET_PATH="./checkpoints/baseline_${TGT}_${SRC_A}.ckpt"
DIR_A="/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_A}"
DIR_B="/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_B}"
LOG_DIR="./adda_logs/${SRC_B}_${TGT}"

mkdir -p $LOG_DIR

python train_adda.py --accelerator 'gpu' --auto_select_gpus 'True' \
  --devices 1 \
  --data_dir_A $DIR_A --data_dir_B $DIR_B \
  --pretrained_unet_path $UNET_PATH --save_dir $LOG_DIR \
  $KWARGS

