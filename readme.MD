# 1. 代码结构
## 1.1. 文件结构
easydiffusion\
    gaussian_diffusion.py
    unet.py
    train.py
    data.zip

## 1.2. 依赖关系
train.py
    ├─ gaussian_diffusion.py
    └─ unet.py

# 2. 代码运行
## 2.1. 环境搭建
`conda create -n dp python=3.11`  
main packge: `torch`

## 2.2. 解压示例数据集
`unzip data.zip` 

## 2.3. 训练
`python train.py`

-----------------
# 3. Diffusion 原理
图片加噪过程及变量`x, beta`的定义，`beta`从零到一递增，一般不会增到1，大概0.02左右合适

<img src="imgs_readme/def_x_beta.png" alt="logo" width="500"/>

变量`alpha`的定义，时间步定义为`T-1`，即全部加噪加噪过程步数

<img src="imgs_readme/def_alpha.png" alt="" width="500"/>

从原图推理到第`t`时间步的公式

<img src="imgs_readme/inference_xt.png" alt="" width="820"/>

变量`alpha_bar`定义为从该位置往前的所有`alpha`相乘

<img src="imgs_readme/def_alpha_bar.png" alt="" width="370"/>

从当前图片`x_t`推理前一步图片`x_t_minus_1`的公式

<img src="imgs_readme/inference_xt-1.png" alt="" width="850"/>

使用推理出的图片替换上图中的`x_0`

<img src="imgs_readme/inference_xt-1_from_inf_orig.png" alt="" width="1000"/>


----------------

