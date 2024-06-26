my_huggingface_token = 'YOUR HUGGING FACE TOKEN';

# Deepspeed distributed training needs a shared storage for saving checkpoints
security_group_ids = ['sg-??????', 'sg-??????', 'sg-?????']
subnets = ['subnet-?????????']
efs_id = 'fs-????????'

training_image_uri = '905418053260.dkr.ecr.us-east-1.amazonaws.com/dinhnn/sagemaker/training/huggingface-pytorch-training:1.13-transformers4.41-gpu-py39-cu117-ubuntu20.04-v0.0.1'
