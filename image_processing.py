import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def process_data(datapath, transformation, batch_sizes=[64, 32, 32],
                 shuffles=[True, False, False]):
    """
    load data to train the model.

    parameters
    ----------
    datapath: str path to dataset;
        This function expect 3 subfolders inside root folder of the data
        `train` `valid` and `test`.
    transformation: list of dict or nested dict;
        a dictionary comprises all image transformation for each `train`,
        `valid`, and `test` with the same key names

    returns
    -------
    image_dataset: dict of ImageFolder dataloaders: DataLoader
    """
    train_dir = os.path.join(datapath, 'train')
    valid_dir = os.path.join(datapath, 'valid')
    test_dir = os.path.join(datapath, 'test')

    image_dataset = {
        'train': datasets.ImageFolder(
            train_dir, transform=transformation['train']
        ),
        'valid': datasets.ImageFolder(
            valid_dir, transform=transformation['valid']
        ),
        'test': datasets.ImageFolder(
            test_dir, transform=transformation['test']
        )
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_dataset['train'],
            batch_size=batch_sizes[0],
            shuffle=shuffles[0]
        ),
        'valid': torch.utils.data.DataLoader(
            image_dataset['valid'],
            batch_size=batch_sizes[1],
            shuffle=shuffles[1]
        ),
        'test': torch.utils.data.DataLoader(
            image_dataset['test'],
            batch_size=batch_sizes[2],
            shuffle=shuffles[2]
        )
    }

    return image_dataset, dataloaders


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when
    # displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize([256, 256])
    left, right = (im.width - 224)/2, (im.width + 224)/2
    top, bottom = (im.height - 224)/2, (im.height + 224)/2
    im = im.crop([left, top, right, bottom])
    np_im = np.array(im)
    np_im[:, :, 0] = (np_im[:, :, 0] - np_im[:, :, 0].mean())
    / np_im[:, :, 0].std()
    np_im[:, :, 1] = (np_im[:, :, 1] - np_im[:, :, 1].mean())
    / np_im[:, :, 1].std()
    np_im[:, :, 2] = (np_im[:, :, 2] - np_im[:, :, 2].mean())
    / np_im[:, :, 2].std()

    return np_im.transpose()
