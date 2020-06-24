import sys
import torch

from arguments import get_arguments
from train import setup_optimisers_and_schedulers


class DummyEncDecModel(torch.nn.Module):
    def __init__(self):
        super(DummyEncDecModel, self).__init__()
        self.layer1 = torch.nn.Parameter(torch.FloatTensor(1, 2))
        self.dec1 = torch.nn.Parameter(torch.FloatTensor(1, 2))


def test_setup_optimisers_and_schedulers():
    # NOTE: Removing any sys.argv to get default arguments
    sys.argv = [""]
    args = get_arguments()
    model = DummyEncDecModel()
    optimisers, schedulers = setup_optimisers_and_schedulers(args, model)
    assert len(optimisers) == 2
    assert len(schedulers) == 2
    for optimiser in optimisers:
        assert isinstance(optimiser, torch.optim.Optimizer)
        assert hasattr(optimiser, "state_dict")
        assert hasattr(optimiser, "load_state_dict")
        assert hasattr(optimiser, "step")
        assert hasattr(optimiser, "zero_grad")
    for scheduler in schedulers:
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        assert hasattr(scheduler, "state_dict")
        assert hasattr(scheduler, "load_state_dict")
        assert hasattr(scheduler, "step")
