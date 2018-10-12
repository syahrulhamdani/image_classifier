# Image Classifier

This is a project from Udacity [Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). An __Image Classifier__ that will classify flowers name from images.

## Overview

In this project, I built a classifier comprises pretrained ImageNet model with new defined classifier architecture.

The classifier architecture used here is 3 hidden layers with units for each of them are `4096`, `2048`, and `512` respectively. This user-defined architecture gives good results classifying images with 85% accuracy only in `15` epochs and `0.0003` for the learning rate. Thanks to transfer learning and pretrained model provided by PyTorch.

## Table of Contents

1. [Getting Started](https://github.com/syahrulhamdani/image_classifier#getting-started)
    1. [Requirements](https://github.com/syahrulhamdani/image_classifier#requirements)
    2. [Installation](https://github.com/syahrulhamdani/image_classifier#installation)
    3. [Usage](https://github.com/syahrulhamdani/image_classifier#usage)
2. [Experiment Results](https://github.com/syahrulhamdani/image_classifier#getting-startedexperiment-results)
3. [References](https://github.com/syahrulhamdani/image_classifier#getting-startedreferences)

## Getting Started

These instruction show what requirements that need to be installed and how to start or inferencing.

### Requirements

To be able to use this repository, you need to have:

1. PyTorch (torch and torchvision)
2. Pillow
3. Numpy
4. Matplotlib (if you want to visualize the image)

Also, you must have downloaded the flowers dataset provided [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). If you have downloaded the dataset, you need to make your dataset to match PyTorch data loader format.

### Usage

- First, make sure you have the dataset in a proper directory structure
- There are optional arguments that can be passed as user input: `--gpu`,
  `-hunits`, `--epochs`, `--learning_rate`, `--arch`, and `--save_dir`.

    - `--save_dir` pass directory where you want to save the trained model,
    - `--gpu` pass a boolean whether activate the gpu or not,
    - `--hidden_units` or `-hunits` pass a list of hidden unit sizes to be the classifier of the model,
    - `--learning_rate` needs you to pass the learning rate, and
    - `--arch` is your choice of pretrained model architecture given in the `torchvision.models`

- `dataset` is required as positional argument as dataset directory for training the model.
- You can create and train your model using `train.py` with some optional arguments above, for example:

```bash
python train.py flowers --arch 'vgg16' -hunits 4096 -hunits 2048 -hunits 512 --learning_rate 0.0003 -saved models --gpu
```

If you run the above command, you will get a pretrained model of `vgg16` with self-defined classifier with hidden units `[4096, 2048, 512]` that will revise predefined classifier in the pretrained model, learning rate `0.0003`, train the network with gpu (the program will check the availability first), and save the trained model in folder named `models`.

## References
