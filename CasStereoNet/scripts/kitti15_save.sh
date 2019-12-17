#!/usr/bin/env bash
#set -x
export PYTHONWARNINGS="ignore"

save_path=$1

if [ ! -d $save_path ];then
    mkdir -p $save_path
fi

DATAPATH="./data/data_scene_flow/"
python save_disp.py --test_dataset kitti  --test_datapath $DATAPATH --testlist ./filenames/kitti15_test.txt \
               --model gwcnet-c  --loadckpt $2  \
               --logdir $save_path \
               --test_crop_width 1248  --test_crop_height 384 \
               --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
               ${@:3} | tee -a  $save_path/log.txt