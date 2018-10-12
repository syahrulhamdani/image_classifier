# Image Classifier

This is a project from Udacity [Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). An __Image Classifier__ that will classify flowers name from images.

## Overview

In this project, I built a classifier comprises pretrained ImageNet model with new defined classifier architecture.

The classifier architecture used here is 3 hidden layers with units for each of them are `4096`, `2048`, and `512` respectively. This user-defined architecture gives good results classifying images with 85% accuracy only in `15` epochs and `0.0003` for the learning rate. Thanks to transfer learning and pretrained model provided by PyTorch.

## Table of Contents

1. [Getting Started](/##getting-started)
    1. [Requirements](/###requirements)
    2. [Installation](/###installation)
    3. [Usage](/###usage)
2. [Experiment Results](/##experiment-results)
3. [References](/##references)

## Getting Started

These instruction show what requirements that need to be installed and how to start or inferencing.

### Requirements

To be able to use this repository, you need to have:

1. PyTorch
2. Pillow
3. Numpy
4. Matplotlib (if you want to visualize the image)

Also, you must have downloaded the flowers dataset provided [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). If you have downloaded the dataset, you need to make your dataset to match PyTorch data loader format.

### Usage

- First, make sure you have the dataset in a proper directory structure
- You can create and train your model using `train.py` with some optional arguments, for example:

```bash
python train.py flowers --arch 'vgg16' -hunits 4096 -hunits 2048 -hunits 512 --learning_rate 0.0003 -saved models
```

- The optional arguments are:

```bash
usage: train.py [-h] [--save_dir CPOINT_DIR] [--gpu]
                [--hidden_units HIDDEN_SIZES] [--epochs EPOCHS]
                [--learning_rate LR] [--arch ARCH]
                dataset

positional arguments:
  dataset

optional arguments:
  -h, --help            show this help message and exit
  --save_dir CPOINT_DIR, -saved CPOINT_DIR
                        save model checkpoints to desired directory
  --gpu                 set gpu on/off, default off
  --hidden_units HIDDEN_SIZES, -hunits HIDDEN_SIZES
                        list of hidden unit with sizes
  --epochs EPOCHS       number of epochs
  --learning_rate LR    learning rate
  --arch ARCH           architecture of pretrained model to be loaded
```

## References
