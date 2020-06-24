# general libs
import logging
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# densetorch wrapper
import densetorch as dt

# configuration for light-weight refinenet
from arguments import get_arguments
from data import get_datasets, get_transforms
from network import get_segmenter
from optimisers import get_optimisers, get_lr_schedulers


def setup_network(args, device):
    logger = logging.getLogger(__name__)
    segmenter = get_segmenter(
        enc_backbone=args.enc_backbone,
        enc_pretrained=args.enc_pretrained,
        num_classes=args.num_classes,
    ).to(device)
    if device == "cuda":
        segmenter = nn.DataParallel(segmenter)
    logger.info(
        " Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M".format(
            args.enc_backbone,
            args.enc_pretrained,
            dt.misc.compute_params(segmenter) / 1e6,
        )
    )
    training_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    validation_loss = dt.engine.MeanIoU(num_classes=args.num_classes)
    return segmenter, training_loss, validation_loss


def setup_checkpoint_and_maybe_restore(args, model, optimisers, schedulers):
    saver = dt.misc.Saver(
        args=vars(args),
        ckpt_dir=args.ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score
    (
        epoch_start,
        _,
        model_state_dict,
        optims_state_dict,
        scheds_state_dict,
    ) = saver.maybe_load(
        ckpt_path=args.ckpt_path,
        keys_to_load=["epoch", "best_val", "model", "optimisers", "schedulers"],
    )
    if epoch_start is None:
        epoch_start = 0
    dt.misc.load_state_dict(model, model_state_dict)
    if optims_state_dict is not None:
        for optim, optim_state_dict in zip(optimisers, optims_state_dict):
            optim.load_state_dict(optim_state_dict)
    if scheds_state_dict is not None:
        for sched, sched_state_dict in zip(schedulers, scheds_state_dict):
            sched.load_state_dict(sched_state_dict)
    return saver, epoch_start


def setup_data_loaders(args):
    train_transforms, val_transforms = get_transforms(
        crop_size=args.crop_size,
        shorter_side=args.shorter_side,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        img_mean=args.img_mean,
        img_std=args.img_std,
        img_scale=args.img_scale,
        ignore_label=args.ignore_label,
        num_stages=args.num_stages,
        augmentations_type=args.augmentations_type,
        dataset_type=args.dataset_type,
    )
    train_sets, val_set = get_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        masks_names=("segm",),
        dataset_type=args.dataset_type,
        stage_names=args.stage_names,
        train_download=args.train_download,
        val_download=args.val_download,
    )
    train_loaders, val_loader = dt.data.get_loaders(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_set=train_sets,
        val_set=val_set,
        num_stages=args.num_stages,
    )
    return train_loaders, val_loader


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type=args.enc_optim_type,
        enc_lr=args.enc_lr,
        enc_weight_decay=args.enc_weight_decay,
        enc_momentum=args.enc_momentum,
        dec_optim_type=args.dec_optim_type,
        dec_lr=args.dec_lr,
        dec_weight_decay=args.dec_weight_decay,
        dec_momentum=args.dec_momentum,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=args.enc_lr_gamma,
        dec_lr_gamma=args.dec_lr_gamma,
        enc_scheduler_type=args.enc_scheduler_type,
        dec_scheduler_type=args.dec_scheduler_type,
        epochs_per_stage=args.epochs_per_stage,
    )
    return optimisers, schedulers


def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.deterministic = True
    dt.misc.set_seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Network
    segmenter, training_loss, validation_loss = setup_network(args, device=device)
    # Data
    train_loaders, val_loader = setup_data_loaders(args)
    # Optimisers
    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=segmenter)
    # Checkpoint
    saver, restart_epoch = setup_checkpoint_and_maybe_restore(
        args, model=segmenter, optimisers=optimisers, schedulers=schedulers,
    )
    # Calculate from which stage and which epoch to restart the training
    total_epoch = restart_epoch
    all_epochs = np.cumsum(args.epochs_per_stage)
    restart_stage = sum(restart_epoch >= all_epochs)
    if restart_stage > 0:
        restart_epoch -= all_epochs[restart_stage - 1]
    for stage in range(restart_stage, args.num_stages):
        if stage > restart_stage:
            restart_epoch = 0
        for epoch in range(restart_epoch, args.epochs_per_stage[stage]):
            logger.info(f"Training: stage {stage} epoch {epoch}")
            dt.engine.train(
                model=segmenter,
                opts=optimisers,
                crits=training_loss,
                dataloader=train_loaders[stage],
                freeze_bn=args.freeze_bn[stage],
                grad_norm=args.grad_norm[stage],
            )
            total_epoch += 1
            for scheduler in schedulers:
                scheduler.step(total_epoch)
            if (epoch + 1) % args.val_every[stage] == 0:
                logger.info(f"Validation: stage {stage} epoch {epoch}")
                vals = dt.engine.validate(
                    model=segmenter, metrics=validation_loss, dataloader=val_loader,
                )
                saver.maybe_save(
                    new_val=vals,
                    dict_to_save={
                        "model": segmenter.state_dict(),
                        "epoch": total_epoch,
                        "optimisers": [
                            optimiser.state_dict() for optimiser in optimisers
                        ],
                        "schedulers": [
                            scheduler.state_dict() for scheduler in schedulers
                        ],
                    },
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
