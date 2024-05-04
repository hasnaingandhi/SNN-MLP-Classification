# CS-525 Brain Inspired Computing

The aim of the project is to to improve MLP performance by integrating brain inspired neural networks such as Spiking Neural Network. Hence, we incorporated the mechanism of LIF neurons into the MLP models, to improve upon accuracy and energy.

## Dataset

- We used ImageNet 1k dataset consisting of 244 x 244 sized images. 
- Hugging Face ImageNet: https://huggingface.co/datasets/imagenet-1k
- Download and store the dataset under imagenet folder

## Requirements
```bash
torch
torchvision
timm
cupy-cuda101
PyYAML
termcolor
yacs
```

## Run commands

### Train

- Tiny
```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 \
 main.py --cfg configs/snnmlp_tiny_patch4_lif4_224.yaml --data-path /imagenet \
--batch-size 128 --cache-mode no --accumulation-steps 0  --output output --lif 4
```

- Small
```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 \
 main.py --cfg configs/snnmlp_small_patch4_lif4_224.yaml --data-path /imagenet \
--batch-size 128 --cache-mode no --accumulation-steps 0  --output output --lif 4
```

- Base
```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 \
 main.py --cfg configs/snnmlp_base_patch4_lif4_224.yaml --data-path /imagenet \
--batch-size 128 --cache-mode no --accumulation-steps 0  --output output --lif 4
```

## Results

- Tiny
  - Accuracy: 81.88%
  - [[log]](https://github.com/hasnaingandhi/SNN-MLP-Classification/blob/main/output/snnmlp_tiny.log)
- Small
  - Accuracy: 83.30%
  - [[log]](https://github.com/hasnaingandhi/SNN-MLP-Classification/blob/main/output/snnmlp_small.log)
- Base
  - Accuracy: 83.59%
  - [[log]](https://github.com/hasnaingandhi/SNN-MLP-Classification/blob/main/output/snnmlp_base.log)
