# 3DCADFusion
This is a python/CUDA implementation of the ["3DCADFusion: Tracking and 3D Reconstruction of Dynamic Objects without External Motion Information"](https://ieeexplore.ieee.org/abstract/document/9658735?casa_token=BNbHLtWnLc0AAAAA:Rvnj4xUqPKrXFsTtxBM2ET4-nn-xm0XHPSZB3R_JmZPbE1bBjkQoKwE7QKY47gjtd-2N66Q) paper. The system is capable of producing dense reconstructions of objects by tracking and reconstructing their incrementally improving 3D geometry in a sequence of RGBD-D or luminance-depth frames. The provided demo is executed on a frame sequence of a robotic gripper containing luminance and depth images.
<br/> <br/> 
![Demo Image](https://github.com/4ri4Slr/3DCADFusion/blob/58a3b2ce1ea968e7cf88947ed6a5ea55d824ce2a/demo-im.png) 
<br/> 
## Environment Setup

The code was built and tested on Ubuntu 18.04.5 LTS with Python 3.8, PyTorch 1.7, and CUDA 10.1. Versions for other packages can be found in `3DCAD.yml`
1. Clone the repo: 
```
git clone https://github.com/4ri4Slr/3DCADFusion.git
cd 3DCADFusion
```
2. Create a conda environment named 3DCAD with necessary dependencies specified in 3DCAD.yml.
```
conda env create -f 3DCAD.yml
```

## Demo

1. Download the demo dataset and copy it to the `data/` directory.

```
mkdir data 
cd data
wget https://drive.google.com/file/d/1misOgyLk8Z_lpvU-wcvIskE-hcp1XSqs/view?usp=sharing
tar -xf data.tar.gz
rm data.tar.gz
cd ..
```

2. Download the pretrained model `best_model.pth`.

```
wget https://drive.google.com/file/d/1qF6BN9Sdrsarlu-Cj9Mv_eF4h3YSdvQ1/view?usp=sharing
```

3. The demo can be executed as follows: 

```
conda activate 3DCAD
python run.py
```

## Training 

You can use the following script to download the GTEA hand segmentation dataset and train the hand segmentation network. 
    
```
mkdir Training_Data

wget https://www.dropbox.com/s/ysi2jv8qr9xvzli/hand14k.zip?dl=0
unzip hand14k.zip?dl=0 -d Training_Data

python train.py
```

    
## References

```
@inproceedings{zeng20163dmatch, 
    title={3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions}, 
    author={Zeng, Andy and Song, Shuran and Nie{\ss}ner, Matthias and Fisher, Matthew and Xiao, Jianxiong and Funkhouser, Thomas}, 
    booktitle={CVPR}, 
    year={2017} 
}


```
