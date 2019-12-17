#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"

save_path=$2

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="./data/sceneflow/"

python -m torch.distributed.launch --nproc_per_node=$1 main.py --dataset sceneflow \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --test_datapath $DATAPATH --test_dataset sceneflow \
    --epochs 16 --lrepochs "10,12,14,16:2" \
    --crop_width 512  --crop_height 256 --test_crop_width 960  --test_crop_height 512 --using_ns --ns_size 3 \
    --model gwcnet-c --logdir $save_path  ${@:3} | tee -a  $save_path/log.txt