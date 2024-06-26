REPO="${1:-905418053260.dkr.ecr.us-east-1.amazonaws.com}"

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $REPO
docker image push $REPO/dinhnn/sagemaker/training/huggingface-pytorch-training:1.13-transformers4.41-gpu-py39-cu117-ubuntu20.04-v0.0.1
