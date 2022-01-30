Prerequisites:
1. setup nvidia docker: https://github.com/NVIDIA/nvidia-docker
2. download data-folder to $PROJECT_ROOT/data
3. download best_model.pth to $PROJECT_ROOT

Usage:
1. run build-image.sh to build image
2. run run-container.sh to run container
3. result-ply files are written to data-folder

Notes:
1. data and model folders are bind mounted into the container, see run-container.sh
2. models are symlinked to right folders
3. pretrained_model is downloaded on build-step, see build-image.sh
4. best_model.pth is copied to model-folder on build step
5. changes to dependencies from conda to requirements.txt:

100c100
< setuptools==49.6.0.post20210108
---
> setuptools==49.6.0
110c110
< torchaudio==0.7.0a0+ac17b64
---
> torchaudio==0.7.0

