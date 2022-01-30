FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /3DCADFusion
WORKDIR /3DCADFusion
RUN apt-get update && apt-get install -y python3.8 python3-pip python3.8-dev libgl1 libusb-1.0-0
RUN python3.8 -m pip install --upgrade pip
COPY requirements.txt /3DCADFusion/requirements.txt
RUN python3.8 -m pip install -r /3DCADFusion/requirements.txt
COPY *.py /3DCADFusion/
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && ln -s /3DCADFusion/model/se_resnext50_32x4d-a260b3a4.pth /root/.cache/torch/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth
RUN ln -s model/best_model.pth best_model.pth
CMD python3.8 run.py && mv *.ply data
