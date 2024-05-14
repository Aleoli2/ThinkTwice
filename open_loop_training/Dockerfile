ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all
RUN pip install torchvision==0.13.1 torchaudio==0.12.1

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS="1"

# Install MMCV
RUN git clone https://github.com/open-mmlab/mmcv.git -b v1.7.0 && cd mmcv && \
pip install -r requirements.txt && pip install -e .  && cd ..

RUN git clone https://github.com/open-mmlab/mmdetection.git -b v2.28.2 && cd mmdetection && \
pip install -r requirements/build.txt &&  pip install -v -e . && cd ..

# Install MMSegmentation
RUN git clone -b v0.30.0 https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
RUN cd /mmsegmentation && pip install -r requirements.txt && \
pip install --no-cache-dir -e . && cd ../

RUN git clone https://github.com/open-mmlab/mmdetection3d.git -b 1.0 && cd mmdetection3d && \
pip install cumm-cu113 && pip install spconv-cu113 && pip install -v -e . && cd ..

WORKDIR /

RUN pip install imgaug
