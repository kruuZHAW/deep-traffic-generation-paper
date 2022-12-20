# hash:sha256:a657c600963ab3bca485f1ec8d9d4f3011844a208fd669aad5180f3a45fd77b4
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++=4:9.3.0-1ubuntu2 \
        gcc=4:9.3.0-1ubuntu2 \
        libgeos-dev=3.8.0-1build1 \
        libproj-dev=6.3.1-1 \
        libstdc++6=10.3.0-1ubuntu1~20.04 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=conda-forge \
        cartopy==0.19.0 \
        cudatoolkit==11.7.0 \
        jupyter==1.0.0 \
        numba==0.55.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    altair==4.2.0 \
    altair-saver==0.5.0 \
    matplotlib==3.5.2 \
    numpy==1.21.0 \
    pandas==1.5.2 \
    pytorch-lightning==1.8.5 \
    scikit-learn==1.2.0 \
    scipy==1.9.3 \
    shapely==1.8.0 \
    tk==0.1.0 \
    torch==1.13.1 \
    torchmetrics==0.11.0 \
    torchvision==0.14.1 \
    traffic==2.8.0

COPY postInstall /
RUN /postInstall
