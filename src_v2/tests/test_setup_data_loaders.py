import sys
import torch
from torchvision.datasets import FakeData

import train
from arguments import get_arguments


def get_fake_datasets(**kwargs):
    return FakeData(), FakeData()


def test_setup_data_loaders(mocker):
    # NOTE: Removing any sys.argv to get default arguments
    sys.argv = [""]
    args = get_arguments()
    mocker.patch.object(train, "get_datasets", side_effect=get_fake_datasets)
    train_loaders, val_loader = train.setup_data_loaders(args)
    assert len(train_loaders) == args.num_stages
    for train_loader in train_loaders:
        assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
