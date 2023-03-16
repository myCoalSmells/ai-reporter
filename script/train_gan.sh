 #/bin/bash
export CUDA_VISIBLE_DEVICES

set -e

SRC_DIR=$1 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC}/input"
TGT_DIR=$2 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC}/output"
LOAD_UNET_PATH=$3 # "./checkpoints/baseline_CD29_PC.ckpt"
LOG_DIR=$4 # "./gan_logs/${SRC}_${TGT}"
KWARGS=$5 # --max_epochs 200 --save_every_n_epochs 20, --max_B_size 20, --name ntrn-100

mkdir -p $LOG_DIR

python train_gan.py --accelerator 'gpu' \
  --auto_select_gpus 'True' --devices 1 \
  --data_src_dir $SRC_DIR --data_tgt_dir $TGT_DIR \
  --pretrained_unet_path $LOAD_UNET_PATH --save_dir $LOG_DIR \
  $KWARGS

