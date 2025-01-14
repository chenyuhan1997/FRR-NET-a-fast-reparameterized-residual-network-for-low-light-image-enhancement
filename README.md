<p align="center">
   <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&width=435&lines=FRR NET&center=true&size=27" />
  <img src="https://github.com/chenyuhan1997/PED/blob/main/assets/1.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>
   <a href="https://github.com/chenyuhan1997/PED/"><img src="https://img.shields.io/badge/Project-PED-black" /></a>&emsp;
</p>

# :fire: FRR-NET: A Fast Reparameterized Residual Network for Low-light Image Enhancement

- :star: - [PAPER](https://link.springer.com/article/10.1007/s11760-024-03127-y)
- :star: - [DATESET  LOL](https://daooshee.github.io/BMVC2018website/)  [DATESET  FIVEK](https://data.csail.mit.edu/graphics/fivek/) 
- :star: - Pretrained Models `Train_result/channel_48/Encoder_weight.pkl` `Train_result/channel_30/Encoder_weight.pkl`

<img width="200%" src="https://github.com/chenyuhan1997/chenyuhan1997/blob/main/assets/hr.gif" />

## Introduction

<img src="https://github.com/chenyuhan1997/FRR-NET-a-fast-reparameterized-residual-network-for-low-light-image-enhancement/blob/main/img/1.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>

Low-light image enhancement algorithm is an important branch in the ﬁeld of image enhancement algorithms. To solve the problem of severe feature degradation in enhanced images after brightness enhancement, much work has been devoted to the construction of multi-scale feature extraction modules. However, this type of research often results in a huge number of parameters, which prevents the work from being generalized. To solve the above problems, this paper proposes a fast repara-metric residual network (FRR-NET) for low-light image enhancement. It achieves results beyond comparable multi-scale fusion modules. By designing a light-weight fast reparametric residual block and a transformer-based brightness enhancement module. The network in this paper has only 0.012 M parameters. Extensive experimental validation shows that the algorithm in this paper is more saturated in color reproduction, while appropriately increasing brightness. FRR-NET performs well on subjective vision tests and image quality tests with fewer parameters compared to existing methods.

<img width="200%" src="https://github.com/chenyuhan1997/chenyuhan1997/blob/main/assets/hr.gif" />

## Getting Started

### Installation

1. Clone FRR-NET.
```bash
git clone --recursive https://github.com/chenyuhan1997/FRR-NET-a-fast-reparameterized-residual-network-for-low-light-image-enhancement
cd FRR-NET
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n FRR-NET python=3.11
conda activate FRR-NET
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install opencv, kornia, pytorch_msssim, matplotlib, PIL, scikit-image, scipy, einops, math, typing
```

### Train & Test

1. Train
```bash
python train.py
```

1. Test
```bash
python test.py
```

<img width="200%" src="https://github.com/chenyuhan1997/chenyuhan1997/blob/main/assets/hr.gif" />

## Results on Low-light Image Enhancement

<img src="https://github.com/chenyuhan1997/FRR-NET-a-fast-reparameterized-residual-network-for-low-light-image-enhancement/blob/main/img/2.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>

<img src="https://github.com/chenyuhan1997/FRR-NET-a-fast-reparameterized-residual-network-for-low-light-image-enhancement/blob/main/img/3.png" alt="my" width="1000" style="display: block; margin: 0 auto;"/>

<img width="200%" src="https://github.com/chenyuhan1997/chenyuhan1997/blob/main/assets/hr.gif" />

## Citations

If you find this project helpful, please consider citing the following papers:

```
@article{chen2024frr,
  title={FRR-NET: a fast reparameterized residual network for low-light image enhancement},
  author={Chen, Yuhan and Zhu, Ge and Wang, Xianquan and Yang, Huan},
  journal={Signal, Image and Video Processing},
  pages={1--10},
  year={2024},
  publisher={Springer}
}
```

