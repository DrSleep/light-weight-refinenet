# general libs
import logging

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


def setup_checkpoint_and_maybe_restore(args, model):
    saver = dt.misc.Saver(
        args=vars(args),
        ckpt_dir=args.ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score
    epoch_start, _, state_dict = saver.load(
        ckpt_path=args.ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],
    )
    if epoch_start is None:
        epoch_start = 0
    dt.misc.load_state_dict(model, state_dict)
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
    )
    train_sets, val_set = get_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        masks_names=("segm",),
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
    enc_optim, dec_optim = get_optimisers(
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
        enc_optim=enc_optim,
        dec_optim=dec_optim,
        enc_lr_gamma=args.enc_lr_gamma,
        dec_lr_gamma=args.dec_lr_gamma,
        epochs_per_stage=args.epochs_per_stage,
    )
    return [enc_optim, dec_optim], schedulers


def main():
    args = get_arguments()
    torch.backends.cudnn.deterministic = True
    dt.misc.set_seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Network
    segmenter, training_loss, validation_loss = setup_network(args, device=device)
    # Checkpoint
    saver, epoch_start = setup_checkpoint_and_maybe_restore(args, model=segmenter)
    # Data
    train_loaders, val_loader = setup_data_loaders(args)
    # Optimisers
    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=segmenter)

    total_epoch = epoch_start
    for stage, num_epochs in enumerate(args.epochs_per_stage):
        if stage > 0:
            epoch_start = 0
        for epoch in range(epoch_start, num_epochs):
            dt.engine.train(
                model=segmenter,
                opts=optimisers,
                crits=training_loss,
                dataloader=train_loaders[stage],
                freeze_bn=args.freeze_bn[stage],
            )
            total_epoch += 1
            for scheduler in schedulers:
                scheduler.step(total_epoch)
            if (epoch + 1) % args.val_every[stage] == 0:
                vals = dt.engine.validate(
                    model=segmenter, metrics=validation_loss, dataloader=val_loader,
                )
                saver.maybe_save(
                    new_val=vals,
                    dict_to_save={
                        "state_dict": segmenter.state_dict(),
                        "epoch": total_epoch,
                    },
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
