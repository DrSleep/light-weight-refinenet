import logging
import numpy as np
import re
from torch.optim.lr_scheduler import MultiStepLR

import densetorch as dt


def get_lr_schedulers(
    enc_optim, dec_optim, enc_lr_gamma, dec_lr_gamma, epochs_per_stage,
):
    milestones = np.cumsum(epochs_per_stage)
    schedulers = [
        MultiStepLR(enc_optim, milestones=milestones, gamma=enc_lr_gamma),
        MultiStepLR(dec_optim, milestones=milestones, gamma=dec_lr_gamma),
    ]
    return schedulers


def get_optimisers(
    model,
    enc_optim_type,
    enc_lr,
    enc_weight_decay,
    enc_momentum,
    dec_optim_type,
    dec_lr,
    dec_weight_decay,
    dec_momentum,
):
    logger = logging.getLogger(__name__)
    # Filter parameters of encoder / decoder
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            enc_params.append(v)
            logger.info(" Enc. parameter: {}".format(k))
        else:
            dec_params.append(v)
            logger.info(" Dec. parameter: {}".format(k))
    # Create optimisers
    optimisers = [
        dt.misc.create_optim(
            optim_type=enc_optim_type,
            parameters=enc_params,
            lr=enc_lr,
            weight_decay=enc_weight_decay,
            momentum=enc_momentum,
        ),
        dt.misc.create_optim(
            optim_type=dec_optim_type,
            parameters=dec_params,
            lr=dec_lr,
            weight_decay=dec_weight_decay,
            momentum=dec_momentum,
        ),
    ]
    return optimisers
