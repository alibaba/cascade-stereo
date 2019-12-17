loadckpt=$1
save_path=$2

save_path_stage1=$save_path/stage1/256-512_800-400-10/psment_48-24_4-1_range-ns3
save_path_stage2=$save_path/stage2/256-1248_8/psment_48-24_4-1_range-ns3
save_path_stage12=$save_path/stage12/test

./scripts/kitti15.sh 8  $save_path_stage1
                       --dataset kitti --datapath ./data/data_scene_flow_all/ --crop_height 256 --crop_width 512 \
                       --test_dataset kitti --test_datapath ./data/data_scene_flow_all/   --test_crop_height 384  --test_crop_width 1248 \
                       --trainlist ./filenames/kitti15_train_more.txt  --testlist ./filenames/kitti15_val.txt \
                       --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
                       --save_freq 50 --eval_freq 25 --batch_size 2 --test_batch_size 2  --epochs 800 --lrepochs "400:10" \
                       --loadckpt $1


./scripts/kitti15.sh 8  $save_path_stage2\
                       --dataset kitti --datapath ./data/data_scene_flow_all/ --crop_height 256 --crop_width 1184 \
                       --test_dataset sceneflow --test_datapath ./data/sceneflow/  --test_crop_height 512 --test_crop_width 960 \
                       --trainlist ./filenames/kitti15_train_more.txt  --testlist ./filenames/sceneflow_test_select.txt \
                       --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
                       --save_freq 1 --eval_freq 1 --batch_size 1 --test_batch_size 1 --epochs 8 --lrepochs "400:10"  \
                       --loadckpt $save_path_stage1/checkpoint_000799.ckpt


./scripts/kitti15.sh 8  $save_path_stage12 \
                       --dataset kitti --datapath ./data/data_scene_flow_all/ --crop_height 256 --crop_width 1184 \
                       --test_dataset kitti --test_datapath ./data/data_scene_flow_all/  --test_crop_height 384 --test_crop_width 1248 \
                       --trainlist ./filenames/kitti15_trainval.txt  --testlist ./filenames/kitti15_trainval.txt \
                       --ndisp "48,24" --disp_inter_r "4,1" --dlossw "0.5,2.0"  --using_ns --ns_size 3 \
                       --save_freq 1 --eval_freq 1 --batch_size 1 --test_batch_size 1 --epochs 8 --lrepochs "400:10"  \
                       --loadckpt $save_path_stage2/checkpoint_best.ckpt \
                       --mode test