#!/usr/bin/env bash

mkdir Training_Data
wget https://www.dropbox.com/s/ysi2jv8qr9xvzli/hand14k.zip?dl=0
unzip hand14k.zip?dl=0 -d Training_Data

conda activate 3DCAD
python train.py

conda deactivate
