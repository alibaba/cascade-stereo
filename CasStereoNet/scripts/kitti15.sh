#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path=$2

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="./data/data_scene_flow/"

python -m torch.distributed.launch --nproc_per_node=$1 main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --test_datapath $DATAPATH --test_dataset kitti \
    --epochs 300 --lrepochs "200:10" \
    --crop_width 512  --crop_height 256 --test_crop_width 1248  --test_crop_height 384 \
    --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
    --model gwcnet-c --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt