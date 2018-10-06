import argparse
import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import load_pretrained
import image_processing as imp


def get_argparse():
    """Get any arguments come from command line. Return a list of
        necessary arguments for building the model.
        """
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument(
        "--save_dir",
        "-saved",
        dest="cpoint_dir",
        action="store",
        help="save model checkpoints to desired directory",
        type=str
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="set gpu on/off, default off"
    )
    parser.add_argument(
        "--hidden_units",
        "-hunits",
        dest="hidden_sizes",
        action="append",
        help="list of hidden unit with sizes",
        type=int
    )
    parser.add_argument(
        "--epochs",
        type=int,
        action="store",
        help="number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, action="store", dest="lr",
        help="learning rate"
    )
    parser.add_argument(
        "--arch",
        type=str,
        action="store",
        help="architecture of pretrained model to be loaded",
    )

    arguments = parser.parse_args()

    # condition for gpu
    if arguments.gpu:
        arguments.with_gpu = "cuda"
    else:
        arguments.with_gpu = "cpu"

    return arguments


def validation(model, dataloaders, criterion, device):
    """validate the model on validatio set using the same criterion with
    training the model. Return validation loss and validation accuracy

    parameters
    ----------
    model: pretrained model with modified classifier from `model.Model` class
    dataloader: torch.utils.data.DataLoader for validation set
    criterion: criterion to compute the loss of the model
    device: gpu device to use to do training process

    returns
    -------
    validation_loss: float. Item of `loss` from criterion
    validation_acc: float. accuracy of the model implemented on validation set
                        compare to the original label
    """
    model.to(device)
    model.eval()
    validation_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataloaders:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            validation_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            is_equal = labels.data == ps.max(dim=1)[1]
            accuracy += is_equal.type(torch.FloatTensor).mean()

    return validation_loss, accuracy


def deep_learning(
    model,
    criterion,
    optimizer,
    trainloader,
    validloader,
    epochs=5,
    print_every=40,
    device="cpu",
):
    """train the pretrained model using user input architecture.

    parameters
    ----------
    model: pretrained model with modified classifier from `model.Model` class
    trainloader: torch.utils.data.DataLoader for training set
    validloader: torch.utils.data.DataLoader for validation set
    criterion: criterion to compute loss of the model
    optimizer: optimizer for updating model parameters
    epochs: int. number of epoch
    print_every: int. limit to print
    device: gpu device to use to do training process
    """
    epochs = epochs
    printed = print_every
    steps = 0
    epoch_loss = 0

    model.to(device)
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            # forward pass
            output = model.forward(images)
            loss = criterion(output, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss
            epoch_loss += loss.item()

            if steps % printed == 0:
                model.eval()

                valid_loss, valid_accuracy = validation(
                    model, validloader, criterion, device
                )
                print(
                    "Epoch: {}/{}..".format(e + 1, epochs),
                    "Training loss: {:.3f}".format(epoch_loss / printed),
                    "Validation loss: {:.3f}".format(
                        valid_loss / len(validloader)),
                    "Validation accuracy: {:.3f}".format(
                        valid_accuracy / len(validloader)
                    )
                )

                epoch_loss = 0

                model.train()


def save_model(model, epochs, optimizer, save_dir):
    """Save the trained model in desired directory based on user input directory.

    parameters
    ----------
    model: trained model to be saved
    state_dict: model.state_dict()
    epochs: int. Number of epochs
    optimizer: torch.optim. Optimizer
    save_dir: string. Directory where model is saved
    """
    checkpoint = {
        'epoch': epochs,
        'model': model,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optim_state': optimizer.state_dict()
    }
    filename = "model.pth"
    savedir = os.path.join(save_dir, filename)
    print("saving the model..")
    torch.save(checkpoint, savedir)
    print("[DONE] Saved!")


# TODO:
# 1. function load_pretrained with classifier and pretrained model name
#       (define directly in model.py)
# 2. function deep_learning to train the model (v)
# 3. function validatation to validate the model used in deep_learning
#    function (v)
# 4. function save_model to save the model to desired directory

if __name__ == "__main__":
    # get arguments input from command line
    argument = get_argparse()
    # load image
    print('Creating data loaders..')
    image_dataset, dataloaders = imp.process_data(argument.dataset)
    # instantiate the model with user input architecture
    print('Loading pre-trained network..')
    model = load_pretrained(
        argument.arch, argument.hidden_sizes, 102, drop_p=0.6
    )
    print('[DONE]')
    print('Model overview:')
    print(model)
    model.class_to_idx = image_dataset['train'].class_to_idx
    # define criterion and optimizer
    print('Define criterion and optimizer..')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=argument.lr
    )
    print('[DONE]')
    # train the model
    print('\n\nTraining the network..')
    deep_learning(
        model, criterion, optimizer, epochs=argument.epochs,
        device=argument.with_gpu, trainloader=dataloaders['train'],
        validloader=dataloaders['valid']
    )
    # save the model
    save_model(model, argument.epochs, optimizer, save_dir=argument.cpoint_dir)
