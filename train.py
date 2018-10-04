import argparse
import sys


def get_argparse():
        """Get any arguments come from command line. Return a list of
        necessary arguments for building the model.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--save_dir', '-saved', dest='cpoint_dir', action='store',
            help='save model checkpoints to desired directory', type=str
        )
        parser.add_argument(
            '--gpu', action='store_true', default=False,
            help='set gpu on/off, default off'
        )
        parser.add_argument(
            '--hidden_units', '-hunits', dest='hidden_sizes',
            action='append', help='list of hidden unit with sizes', type=int
        )
        parser.add_argument(
            '--epochs', type=int, action='store', help='number of epochs'
        )
        parser.add_argument(
            '--learning_rate', type=float, action='store', dest='lr',
            help='learning rate'
        )
        parser.add_argument(
            '--arch', type=str, action='store',
            help='architecture of pretrained model to be loaded'
        )

        arguments = parser.parse_args()

        return arguments


# TODO:
# 1. function load_pretrained with classifier and pretrained model name
# 2. function deep_learning to train the model
# 3. function validate_model to validate the model used in deep_learning
#    function
# 4. function test_model to test the model after being trained

if __name__ == '__name__':
    argument = get_argparse()
