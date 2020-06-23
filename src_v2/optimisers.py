import numpy as np

import densetorch as dt

from network import get_encoder_and_decoder_params


def get_lr_schedulers(
    enc_optim,
    dec_optim,
    enc_lr_gamma,
    dec_lr_gamma,
    enc_scheduler_type,
    dec_scheduler_type,
    epochs_per_stage,
):
    milestones = np.cumsum(epochs_per_stage)
    max_epochs = milestones[-1]
    schedulers = [
        dt.misc.create_scheduler(
            scheduler_type=enc_scheduler_type,
            optim=enc_optim,
            gamma=enc_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
        dt.misc.create_scheduler(
            scheduler_type=dec_scheduler_type,
            optim=dec_optim,
            gamma=dec_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
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
    enc_params, dec_params = get_encoder_and_decoder_params(model)
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
