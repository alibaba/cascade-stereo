# Casacde-Stereo(CVPR2020 Oral)
Official source code of paper Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching, [arxiv](https://arxiv.org/pdf/1912.06378.pdf)

# Installation
## Requirements
* python 3.6
* Pytorch == 1.2 （The default interpolation of the high version is different from the low version, which will cause the results to be different）
* CUDA >= 9.0

```
pip install -r requirements.txt  --user
```

## Option: Apex 
install apex to enable synchronized batch normalization 
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# CasMVSNet(Multi View Stereo)
Refer to [MVS-README.md](CasMVSNet/README.md)
# CasStereoNet(Stereo Matching)
Refer to [Stereo-Matching-README.md](CasStereoNet/README.md)

# Citation
If you find this code useful in your research, please cite:

```
@inproceedings{gu2020cascade,
  title={Cascade cost volume for high-resolution multi-view stereo and stereo matching},
  author={Gu, Xiaodong and Fan, Zhiwen and Zhu, Siyu and Dai, Zuozhuo and Tan, Feitong and Tan, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2495--2504},
  year={2020}
}
```

# Acknowledgements
Thanks to Xiaoyang Guo for opening source of his excellent works [GwcNet](https://github.com/xy-guo/GwcNet)
and [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch). Thanks to Yao Yao for opening source of 
his excellent work [MVSNet](https://github.com/YoYo000/MVSNet).
