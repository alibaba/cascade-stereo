# Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)


# Scene Flow Datasets
Please update `DATAPATH` in the ``sceneflow.sh`` file as your training data path.
## Training
Multi-GPU training
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/sceneflow.sh $NGPUS $save_results_dir  --ndisps "48,24"  --disp_inter_r "4,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3  --model gwcnet-c
```
You can set ``--mdoel gwcnet`` to using GwcNet-gc as baseline model. ``gwcnet-c`` refer to only using concatenation cost volume which similar to PSMNet.

## Evaluation
* Set ``CKPT_FILE`` as your checkpoint file, you also can download my pretrained model 
[cas-gwcnet-c](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasStereoNet/gwcnet-c/casgwcnet-c-2.ckpt) and
[cas-gwcnet-gc](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasStereoNet/gwcnet/casgwcnet.ckpt) 
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/sceneflow.sh $NGPUS $save_results_dir --loadckpt $CKPT_FILE- --ndisps "48,24"  --disp_inter_r "4,1"   --batch_size 2 --mode test  --model gwcnet-c
```


## Results on SceneFlow datsets
|                       | >1px.  | >2px.  | >3px.  | EPE.   |
|-----------------------|--------|--------|--------|--------|
| GwcNet-gc             | 8.03   | 4.47   | 3.30   | 0.765  |
| CasGwcNet-gc(D=48,24) | 7.47   | 4.16   | 3.05   | 0.649  |
| GwcNet-c              | 8.41   | 4.63   | 3.41   | 0.808  | 
| CasGwcNet-c(D=48,24)  | 7.49   | 4.16   | 3.04   | 0.650  |


# KITTI 2012 / 2015
Please update `DATAPATH` in the ``kitti15.sh`` file as your training data path.

## Training
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/kitti15.sh $NGPUS $save_results_dir  --ndisps "48,24"  --disp_inter_r "4,1"   --dlossw "0.5,1.0,2.0"  --batch_size 2 --eval_freq 3  --model gwcnet-c
```

## Evaluation
* Set ``CKPT_FILE`` as your checkpoint file, you also can download my pretrained model 
[cas-gwcnet-c](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasStereoNet/kitti/casgwcnet-c/kitti2015.ckpt) for KITTI 2015. 
```
export NGPUS=8
export save_results_dir="./checkpoints"
./scripts/kitti15.sh $NGPUS $save_results_dir  --loadckpt $CKPT_FILE- --ndisps "48,24"  --disp_inter_r "4,1"   --batch_size 2 --mode test  --model gwcnet-c
```

## Save Disps
```
export save_path="./outputs/predictions"
./scripts/kitti15_save.sh $save_results_dir  --loadckpt $CKPT_FILE --ndisps "48,24"  --disp_inter_r "4,1"   --batch_size 2 --model gwcnet-c
```