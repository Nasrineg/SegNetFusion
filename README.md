
# Image Segmentation using U-Net and Swin Transformer UNet

This repository contains two deep learning models for medical image segmentation: **U-Net** and **Swin Transformer UNet**. Both models are implemented using TensorFlow/Keras and are designed to perform pixel-wise classification, often used for tasks like medical imaging, autonomous driving, or any other segmentation-related problems.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
    - [U-Net Model](#u-net-model)
    - [Swin Transformer UNet](#swin-transformer-unet)
- [Data Augmentation](#data-augmentation)
- [Usage](#usage)
    - [Requirements](#requirements)
    - [Training](#training)
    - [Evaluation](#evaluation)
- [References](#references)

## Introduction
Image segmentation is crucial in various fields, such as medical imaging and autonomous driving. The two models implemented here—**U-Net** and **Swin Transformer UNet**—offer powerful solutions for segmentation tasks:

- **U-Net**: A convolutional neural network architecture that uses an encoder-decoder structure with skip connections for efficient feature extraction and image reconstruction.
- **Swin Transformer UNet**: A transformer-based architecture, utilizing the Swin Transformer as the backbone. It provides excellent long-range context understanding due to its self-attention mechanism.

## Models

### U-Net Model
The U-Net model is based on the popular encoder-decoder structure, which consists of a contracting path (downsampling) followed by an expansive path (upsampling). The key feature of U-Net is its **skip connections**, which help preserve high-resolution features and improve the model's segmentation capabilities.

![U-Net Architecture](unet.png)

Key components:
- **Encoder**: Extracts features from the image using a series of convolutional layers followed by max-pooling.
- **Bottleneck**: Contains the highest-level features before upsampling begins.
- **Decoder**: Upsamples the features to match the original input resolution.
- **Skip Connections**: Directly connect encoder layers to decoder layers to retain spatial information.

### Swin Transformer UNet
The **Swin Transformer UNet** utilizes the Swin Transformer, which is based on self-attention mechanisms. Swin Transformers capture long-range dependencies and have strong performance on various vision tasks, including image segmentation. In the Swin Transformer UNet, the image is processed in windows of patches and features are extracted using attention layers.

Key components:
- **Self-Attention**: Captures global information in images through attention mechanisms.
- **Patch-Based Processing**: The image is divided into patches, and transformations are applied to capture both local and global features.
- **Decoder**: Similar to U-Net, but it integrates the transformer layers.

## Data Augmentation
The training dataset undergoes several augmentations to improve the model's generalization capabilities. The augmentations include:
- **Flipping**: Random horizontal/vertical flips.
- **Rotation**: Random rotations within a specified range.
- **Scaling**: Random scaling of the image.
- **Translation**: Random pixel shifts.
- **Noise Injection**: Adding random noise to the image for robustness.

These transformations are applied using a custom function with the `tf.raw_ops.ImageProjectiveTransformV2` operation.

## Usage

### Requirements
To run the models, you need to have the following libraries installed:
- TensorFlow (>= 2.0)
- Keras
- Matplotlib
- Python 3.x

Install the required packages with:
```bash
pip install tensorflow keras matplotlib
```

### Training
To train the models, the dataset should be structured into two folders:
- **frames/**: Contains the input images.
- **masks/**: Contains the corresponding binary masks for each image.

Run the script to train either the U-Net or Swin Transformer UNet model:
```bash
python train.py --model unet --data_dir /path/to/data/
```

For Swin Transformer UNet:
```bash
python train.py --model swin --data_dir /path/to/data/
```

### Evaluation
The model can be evaluated using a separate script. After training, evaluate the model on a validation dataset:
```bash
python evaluate.py --model /path/to/saved_model --data_dir /path/to/validation_data/
```

The model's performance is measured using:
- **Dice Coefficient**: A common evaluation metric for image segmentation.
- **Accuracy**: Pixel-wise classification accuracy.

## References
- **U-Net Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **Swin Transformer Paper**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
