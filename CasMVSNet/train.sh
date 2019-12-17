#!/usr/bin/env bash
MVS_TRAINING="./data/DTU/mvs_training/dtu/"

LOG_DIR=$2
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=$1 train.py --logdir $LOG_DIR --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 ${@:3} | tee -a $LOG_DIR/log.txt