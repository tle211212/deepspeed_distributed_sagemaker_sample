
BUILDER_IMAGE="ubuntu:22.04"
RUNTIME_IMAGE="ubuntu:22.04"

PUSH_IMAGE="905418053260.dkr.ecr.us-east-1.amazonaws.com/dinhnn/sagemaker/training/huggingface-pytorch-training:1.13-transformers4.41-gpu-py39-cu117-ubuntu20.04-v0.0.2"

export PIP_DEFAULT_TIMEOUT=100

DOCKER_BUILDKIT=0 docker build \
--build-arg BUILDER_IMAGE=$BUILDER_IMAGE \
--build-arg RUNTIME_IMAGE=$RUNTIME_IMAGE \
--build-arg PYTORCH_DOWNLOAD_URL=https://download.pytorch.org/whl/cu117/torch_stable.html \
--build-arg PYTORCH_VERSION=1.13.1 \
--build-arg PYTORCH_VERSION_SUFFIX=+cu117 \
--build-arg TORCHAUDIO_VERSION=0.13.1 \
--build-arg TORCHAUDIO_VERSION_SUFFIX=+cu117 \
--build-arg TORCHVISION_VERSION=0.14.1 \
--build-arg TORCHVISION_VERSION_SUFFIX=+cu117 \
--build-arg DATASETS_VERSION=2.16.1 \
--build-arg DIFFUSERS_VERSION=0.27.2 \
--build-arg PT_TORCHAUDIO_URL=https://download.pytorch.org/whl/cu117/torchaudio-0.13.1%2Bcu117-cp39-cp39-linux_x86_64.whl \
--build-arg TRANSFORMERS_VERSION=4.26.0 \
--build-arg HOME_DIR=root \
--build-arg FLASH_ATTN_VERSION=2.5.9.post1 \
--build-arg PT_SM_TRAINING_URL=https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp39-cp39-linux_x86_64.whl \
--build-arg PT_TORCHAUDIO_URL=https://download.pytorch.org/whl/cu117/torchaudio-0.13.1%2Bcu117-cp39-cp39-linux_x86_64.whl \
--build-arg PT_TORCHDATA_URL=https://download.pytorch.org/whl/test/torchdata-0.5.1-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
--build-arg PT_TORCHVISION_URL=https://download.pytorch.org/whl/cu117/torchvision-0.14.1%2Bcu117-cp39-cp39-linux_x86_64.whl \
--build-arg PYTHON=python3 \
--build-arg PYTHON_SHORT_VERSION=3.9 \
--build-arg PYTHON_VERSION=3.9.19 \
--build-arg RMM_VERSION=0.18.0 \
--build-arg SMDEBUG_VERSION=1.0.34 \
--build-arg SMD_DATA_PARALLEL_URL=https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.13.1/cu117/2023-01-09/smdistributed_dataparallel-1.7.0-cp39-cp39-linux_x86_64.whl \
--build-arg SMD_MODEL_PARALLEL_URL=https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/pytorch-1.13.1/build-artifacts/2023-04-17-15-49/smdistributed_modelparallel-1.15.0-cp39-cp39-linux_x86_64.whl \
--build-arg SMPPY_BINARY=smppy-0.3.325-cp39-cp39-linux_x86_64.whl \
--build-arg TRITON_VERSION=2.0.0.dev20221202 \
--build-arg ZERO_2D_URL=https://aws-deepspeed-zero-2d-binaries.s3.us-west-2.amazonaws.com/r1.13.1/20230412-070309/4c3ff1af262257f1636b4aafd497e0c332a1bc1d/deepspeed-0.6.1%2B4c3ff1a-py3-none-any.whl \
--build-arg TARGETARCH=amd64 \
--build-arg CUBLAS_VERSION=11.10.3.66 \
--build-arg CUDA_HOME=/usr/local/cuda \
--build-arg EFA_PATH=/opt/amazon/efa \
--build-arg MAMBA_VERSION=22.11.1-2 \
--build-arg BUILDKIT_INLINE_CACHE=1 \
-t $PUSH_IMAGE .
