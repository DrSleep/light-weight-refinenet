import sys
import torch

import densetorch as dt

from arguments import get_arguments
from train import setup_network


def test_setup_network():
    # NOTE: Removing any sys.argv to get default arguments
    sys.argv = [""]
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter, training_loss, validation_loss = setup_network(args, device)
    assert isinstance(segmenter, torch.nn.Module)
    assert isinstance(training_loss, torch.nn.CrossEntropyLoss)
    assert isinstance(validation_loss, dt.engine.MeanIoU)
