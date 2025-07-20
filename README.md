# Swin Transformer on CIFAR-10 with RandAugment
This project implements and experiments with the **Swin Transformer**—a hierarchical vision transformer with shifted window attention—for image classification on the **CIFAR-10 dataset**. The code uses Berniwal's PyTorch-based Swin implementation and incorporates **RandAugment-based preprocessing** for better generalization.

## Project Overview
The **Swin Transformer** introduces linear-complexity self-attention within non-overlapping shifted windows, making it an efficient and scalable alternative to traditional ViT architectures. This project focuses on:
- Understanding and reproducing the Swin architecture using PyTorch
- Comparing Swin with ViT in terms of architectural improvements
- Training Swin on CIFAR-10 using `randomaug.py`-based data augmentation
- Tracking training/testing accuracy and loss across epochs

## What’s Implemented
- Swin Transformer fine-tuning on CIFAR-10
- Custom RandAugment transformations (`randomaug.py`)
- Modular data loaders for CIFAR-10 and CIFAR-100 with augmentation (`Utils.py`)
- Performance tracking with training/test metrics exported as CSV
- Summary memo and presentation of the work done

## Repository Contents
- `Project.ipynb`: Main notebook with model training, checkpointing, metric plotting, and visualization
- `randomaug.py`: Custom augmentation operators used in training
- `Utils.py`: CIFAR data loading functions with optional RandAugment insertion
- `memo.pdf`: Brief report summarizing project details and observations
- `presentation.pdf`: Slide deck explaining the Swin Transformer and its comparison with Vision Transformer (ViT)

## Running the Notebook
Run `Project.ipynb` from top to bottom. It:
- Loads CIFAR-10 with augmentations  
- Initializes and trains the Swin model  
- Evaluates performance and saves plots  
- Optionally loads a pretrained checkpoint for predictions  

> This project was completed as part of academic coursework exploring the efficiency of hierarchical vision transformers using shifted windows on low-resolution datasets.
