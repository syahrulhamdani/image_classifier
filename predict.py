import os
import argparse
import json
import torch
import image_processing as imp
from model import load_pretrained
from torchvision import transforms


def get_argument():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "image_path", action='store', type=str
    )
    parser.add_argument(
        "checkpoint", action='store', type=str
    )
    parser.add_argument(
        '--top_k', dest='topk', action='store', type=int,
        help='number of top probabilities to show', default=5
    )
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help='gpu on/off'
    )
    parser.add_argument(
        '--category_names', dest='cat_names', action='store', type=str,
        help='use mapping categories', default=None
    )
    argument = parser.parse_args()

    if argument.gpu:
        argument.with_gpu = 'cuda'
    else:
        argument.with_gpu = 'cpu'

    return argument


def load_model(filepath):
    """Load saved model after trained.

    parameters
    ----------
    filepath(str): path to saved model.
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optim']
    optimizer.state_dict = checkpoint['optim_state']
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def predict(model, image_path, top_k=5, cat_class=None, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(imp.process_image(image_path))
        output = model(image.type(torch.float32).to(device).unsqueeze_(0))
        ps = torch.exp(output)
    prob = ps.topk(topk_k)[0].cpu().numpy()
    idx = ps.topk(top_k)[1].cpu().numpy()
    class_name = []
    for key, value in model.class_to_idx.items():
        if value in idx[0]:
            class_name.append(key)
    if cat_class is not None:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        flower_name = cat_to_name[class_name[0]]

    return class_name, prob[0][0]


if __name__ == "__main__":
    argument = get_argument()
    # load model, optimizer, and epochs from checkpoint
    print('loading saved model checkpoint..')
    model, optimizer, epochs = load_model(argument.checkpoint)
    print('[DONE]')
    # predict the image
    print('Predicting {}..'.format(argument.image_path))
    flower_name, prob = predict(
        model, argument.image_path, argument.topk,
        argument.cat_names, argument.with_gpu
    )
    print('[DONE]')
    print('flower name: {}.. with probability: {}'.format(flower_name, prob))
