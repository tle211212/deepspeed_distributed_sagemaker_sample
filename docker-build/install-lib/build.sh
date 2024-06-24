

BASE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13-transformers4.26-gpu-py39-cu117-ubuntu20.04"

PUSH_IMAGE="905418053260.dkr.ecr.us-east-1.amazonaws.com/dinhnn/sagemaker/training/huggingface-pytorch-training:1.13-transformers4.41-gpu-py39-cu117-ubuntu20.04-v0.0.1"

export PIP_DEFAULT_TIMEOUT=100

DOCKER_BUILDKIT=1 docker build \
--build-arg BASE_IMAGE=$BASE_IMAGE \
--build-arg BUILDKIT_INLINE_CACHE=1 \
-t $PUSH_IMAGE .

