# Casacde-Stereo
Official source code of paper Cascade Cost Volume for High-Resolution Deep Stereo and Multi-View Stereo, [arxiv](https://arxiv.org/pdf/1912.06378.pdf)

# Installation
## Requirements
* python 3.6
* Pytorch >= 1.0.0
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
refer to [MVS-README.md](CasMVSNet/README.md)
# CasStereoNet(Stereo Matching)
refer to [Stereo-Matching-README.md](CasStereoNet/README.md)

# Citation
If you find this code useful in your research, please cite:

```
@inproceedings{gu2019cas,
  title={Cascade Cost Volume for High-Resolution Deep Stereo and Multi-View Stereo},
  author={Gu, Xiaodong and Fan, Zhiwen and Zhu, Siyu and Dai, Zuozhuo and Tan, Feitong and Tan, Ping},
  journal={arxiv preprint arXiv:1912.06378},
  year={2019}
}
```

# Acknowledgements
Thanks to Xiaoyang Guo for opening source of his excellent works [GwcNet](https://github.com/xy-guo/GwcNet)
and [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch). Thanks to Yao Yao for opening source of 
his excellent work [MVSNet](https://github.com/YoYo000/MVSNet).