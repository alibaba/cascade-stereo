# Trianing and testing on DTU datset.
## Training
* Dowload [DTU dataset](https://roboimagedata.compute.dtu.dk/). For convenience, can download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.

```                
├── Cameras    
├── Depths
├── Depths_raw   
├── Rectified
├── Cameras                               
             
```

* In ``train.sh``, set ``MVS_TRAINING`` as $MVS_TRANING
* Train CasMVSNet (Multi-GPU training): 

```
export NGPUS=8
export save_results_dir="./checkpoints"
./train.sh $NGPUS $save_results_dir  --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3
```

If apex is installed, you can enable sync_bn in training:
```
export NGPUS=8
export $save_results_dir="./checkpoints"
./train.sh $NGPUS $save_results_dir  --ndepths "48,32,8"  --depth_inter_r "4,2,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3  --using_apex  --sync_bn
```

## Testing and Fusion
* Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the $TESTPATH folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* In ``test.sh``, set ``TESTPATH`` as $TESTPATH.
* Set ``CKPT_FILE``  as your checkpoint file, you also can download my [pretrained model](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/48_32_8-4-2-1_dlossw-0.5-1.0-2.0/casmvsnet.ckpt).
* Test CasMVSNet and Fusion(default is provided by [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch)): 
```
export save_results_dir="./outputs"
./test.sh  $CKPT_FILE --outdir $save_results_dir  --interval_scale 1.06
```
* We also support Gipuma to fusion(need to install [fusibile](https://github.com/YoYo000/fusibile)) . the script is borrowed from [MVSNet](https://github.com/YoYo000/MVSNet). 
```
export save_results_dir="./outputs"
./test.sh  $CKPT_FILE --outdir $save_results_dir  --interval_scale 1.06  --filter_method gipuma
```

## Results on DTU
|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| MVSNet(D=256)         | 0.396  | 0.527  | 0.462    |
| CasMVSNet(D=48,32,8)  | 0.325  | 0.385  | 0.355    |

## Results on Tanks and Temples benchmark

| Mean   | Family | Francis | Horse  | Lighthouse | M60    | Panther | Playground | Train |
|--------|--------|---------|--------|------------|--------|---------|------------|-------|
| 56.42  | 76.36  | 58.45   | 46.20  | 55.53	  | 56.11  | 54.02   | 58.17	  | 46.56 |

Please refer to [leaderboard](https://www.tanksandtemples.org/details/691/).

# CasMVSNet input from COLMAP SfM
We use a script provided by [MVSNet](https://github.com/YoYo000/MVSNet) to convert COLMAP SfM 
result to CasMVSNet input. After recovering SfM result and undistorting all images, 
COLMAP should generate a dense folder COLMAP/dense/ containing an undistorted image folder 
COLMAP/dense/images/ and a undistorted camera folder COLMAP/dense/sparse/. Then, you can use the following command to generate the CasMVSNet input and dense point cloud:

```
export $save_results_dir="outputs/colmap"
python colmap2mvsnet.py --dense_folder COLMAP/dense  --save_folder $save_scene_result/casmvsnet
./test.sh  $CKPT_FILE  --testpath_single_scene $save_results_dir/casmvsnet  --testlist all --outdir $save_results_dir/ply --interval_scale 1.06  
```

