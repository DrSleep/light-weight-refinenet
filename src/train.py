"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# general libs
import argparse
import logging
import os
import random
import re
import time

# misc
import cv2
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# custom libs
from config import *
from miou_utils import compute_iu, fast_cm
from util import *


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument(
        "--train-dir",
        type=str,
        default=TRAIN_DIR,
        help="Path to the training set directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=VAL_DIR,
        help="Path to the validation set directory.",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        nargs="+",
        default=TRAIN_LIST,
        help="Path to the training set list.",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        nargs="+",
        default=VAL_LIST,
        help="Path to the validation set list.",
    )
    parser.add_argument(
        "--shorter-side",
        type=int,
        nargs="+",
        default=SHORTER_SIDE,
        help="Shorter side transformation.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs="+",
        default=CROP_SIZE,
        help="Crop size for training,",
    )
    parser.add_argument(
        "--normalise-params",
        type=list,
        default=NORMALISE_PARAMS,
        help="Normalisation parameters [scale, mean, std],",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=BATCH_SIZE,
        help="Batch size to train the segmenter model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for pytorch's dataloader.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="+",
        default=NUM_CLASSES,
        help="Number of output classes for each task.",
    )
    parser.add_argument(
        "--low-scale",
        type=float,
        nargs="+",
        default=LOW_SCALE,
        help="Lower bound for random scale",
    )
    parser.add_argument(
        "--high-scale",
        type=float,
        nargs="+",
        default=HIGH_SCALE,
        help="Upper bound for random scale",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=IGNORE_LABEL,
        help="Label to ignore during training",
    )

    # Encoder
    parser.add_argument("--enc", type=str, default=ENC, help="Encoder net type.")
    parser.add_argument(
        "--enc-pretrained",
        type=bool,
        default=ENC_PRETRAINED,
        help="Whether to init with imagenet weights.",
    )
    # General
    parser.add_argument(
        "--evaluate",
        type=bool,
        default=EVALUATE,
        help="If true, only validate segmentation.",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        nargs="+",
        default=FREEZE_BN,
        help="Whether to keep batch norm statistics intact.",
    )
    parser.add_argument(
        "--num-segm-epochs",
        type=int,
        nargs="+",
        default=NUM_SEGM_EPOCHS,
        help="Number of epochs to train for segmentation network.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY,
        help="Print information every often.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=SNAPSHOT_DIR,
        help="Path to directory for storing checkpoints.",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=CKPT_PATH, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--val-every",
        nargs="+",
        type=int,
        default=VAL_EVERY,
        help="How often to validate current architecture.",
    )

    # Optimisers
    parser.add_argument(
        "--lr-enc",
        type=float,
        nargs="+",
        default=LR_ENC,
        help="Learning rate for encoder.",
    )
    parser.add_argument(
        "--lr-dec",
        type=float,
        nargs="+",
        default=LR_DEC,
        help="Learning rate for decoder.",
    )
    parser.add_argument(
        "--mom-enc",
        type=float,
        nargs="+",
        default=MOM_ENC,
        help="Momentum for encoder.",
    )
    parser.add_argument(
        "--mom-dec",
        type=float,
        nargs="+",
        default=MOM_DEC,
        help="Momentum for decoder.",
    )
    parser.add_argument(
        "--wd-enc",
        type=float,
        nargs="+",
        default=WD_ENC,
        help="Weight decay for encoder.",
    )
    parser.add_argument(
        "--wd-dec",
        type=float,
        nargs="+",
        default=WD_DEC,
        help="Weight decay for decoder.",
    )
    parser.add_argument(
        "--optim-dec",
        type=str,
        default=OPTIM_DEC,
        help="Optimiser algorithm for decoder.",
    )
    return parser.parse_args()


def create_segmenter(net, pretrained, num_classes):
    """Create Encoder; for now only ResNet [50,101,152]"""
    from models.resnet import rf_lw50, rf_lw101, rf_lw152

    if str(net) == "50":
        return rf_lw50(num_classes, imagenet=pretrained)
    elif str(net) == "101":
        return rf_lw101(num_classes, imagenet=pretrained)
    elif str(net) == "152":
        return rf_lw152(num_classes, imagenet=pretrained)
    else:
        raise ValueError("{} is not supported".format(str(net)))


def create_loaders(
    train_dir,
    val_dir,
    train_list,
    val_list,
    shorter_side,
    crop_size,
    low_scale,
    high_scale,
    normalise_params,
    batch_size,
    num_workers,
    ignore_label,
):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Custom libraries
    from datasets import NYUDataset as Dataset
    from datasets import (
        Pad,
        RandomCrop,
        RandomMirror,
        ResizeShorterScale,
        ToTensor,
        Normalise,
    )

    ## Transformations during training ##
    composed_trn = transforms.Compose(
        [
            ResizeShorterScale(shorter_side, low_scale, high_scale),
            Pad(crop_size, [123.675, 116.28, 103.53], ignore_label),
            RandomMirror(),
            RandomCrop(crop_size),
            Normalise(*normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose([Normalise(*normalise_params), ToTensor()])
    ## Training and validation sets ##
    trainset = Dataset(
        data_file=train_list,
        data_dir=train_dir,
        transform_trn=composed_trn,
        transform_val=composed_val,
    )

    valset = Dataset(
        data_file=val_list,
        data_dir=val_dir,
        transform_trn=None,
        transform_val=composed_val,
    )
    logger.info(
        " Created train set = {} examples, val set = {} examples".format(
            len(trainset), len(valset)
        )
    )
    ## Training and validation loaders ##
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def create_optimisers(
    lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec
):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(
        param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc
    )
    if optim_dec == "sgd":
        optim_dec = torch.optim.SGD(
            param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec
        )
    elif optim_dec == "adam":
        optim_dec = torch.optim.Adam(
            param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3
        )
    return optim_enc, optim_dec


def load_ckpt(ckpt_path, ckpt_dict):
    best_val = epoch_start = 0
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(ckpt_path)
        for (k, v) in ckpt_dict.items():
            if k in ckpt:
                v.load_state_dict(ckpt[k])
        best_val = ckpt.get("best_val", 0)
        epoch_start = ckpt.get("epoch_start", 0)
        logger.info(
            " Found checkpoint at {} with best_val {:.4f} at epoch {}".format(
                ckpt_path, best_val, epoch_start
            )
        )
    return best_val, epoch_start


def train_segmenter(
    segmenter, train_loader, optim_enc, optim_dec, epoch, segm_crit, freeze_bn
):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage("train")
    segmenter.train()
    if freeze_bn:
        for m in segmenter.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    for i, sample in enumerate(train_loader):
        start = time.time()
        input = sample["image"].cuda()
        target = sample["mask"].cuda()
        input_var = torch.autograd.Variable(input).float()
        target_var = torch.autograd.Variable(target).long()
        # Compute output
        output = segmenter(input_var)
        output = nn.functional.interpolate(
            output, size=target_var.size()[1:], mode="bilinear", align_corners=False
        )
        soft_output = nn.LogSoftmax()(output)
        # Compute loss and backpropagate
        loss = segm_crit(soft_output, target_var)
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        if i % args.print_every == 0:
            logger.info(
                " Train epoch: {} [{}/{}]\t"
                "Avg. Loss: {:.3f}\t"
                "Avg. Time: {:.3f}".format(
                    epoch, i, len(train_loader), losses.avg, batch_time.avg
                )
            )


def validate(segmenter, val_loader, epoch, num_classes=-1):
    """Validate segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      val_loader (DataLoader) : training data iterator
      epoch (int) : current epoch
      num_classes (int) : number of classes to consider

    Returns:
      Mean IoU (float)
    """
    val_loader.dataset.set_stage("val")
    segmenter.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input = sample["image"]
            target = sample["mask"]
            input_var = torch.autograd.Variable(input).float().cuda()
            # Compute output
            output = segmenter(input_var)
            output = (
                cv2.resize(
                    output[0, :num_classes].data.cpu().numpy().transpose(1, 2, 0),
                    target.size()[1:][::-1],
                    interpolation=cv2.INTER_CUBIC,
                )
                .argmax(axis=2)
                .astype(np.uint8)
            )
            # Compute IoU
            gt = target[0].data.cpu().numpy().astype(np.uint8)
            gt_idx = (
                gt < num_classes
            )  # Ignore every class index larger than the number of classes
            cm += fast_cm(output[gt_idx], gt[gt_idx], num_classes)

            if i % args.print_every == 0:
                logger.info(
                    " Val epoch: {} [{}/{}]\t"
                    "Mean IoU: {:.3f}".format(
                        epoch, i, len(val_loader), compute_iu(cm).mean()
                    )
                )

    ious = compute_iu(cm)
    logger.info(" IoUs: {}".format(ious))
    miou = np.mean(ious)
    logger.info(" Val epoch: {}\tMean IoU: {:.3f}".format(epoch, miou))
    return miou


def main():
    global args, logger
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Add args ##
    args.num_stages = len(args.num_classes)
    ## Set random seeds ##
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    ## Generate Segmenter ##
    segmenter = nn.DataParallel(
        create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0])
    ).cuda()
    logger.info(
        " Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M".format(
            args.enc, args.enc_pretrained, compute_params(segmenter) / 1e6
        )
    )
    ## Restore if any ##
    best_val, epoch_start = load_ckpt(args.ckpt_path, {"segmenter": segmenter})
    ## Criterion ##
    segm_crit = nn.NLLLoss2d(ignore_index=args.ignore_label).cuda()

    ## Saver ##
    saver = Saver(
        args=vars(args),
        ckpt_dir=args.snapshot_dir,
        best_val=best_val,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score

    logger.info(" Training Process Starts")
    for task_idx in range(args.num_stages):
        start = time.time()
        torch.cuda.empty_cache()
        ## Create dataloaders ##
        train_loader, val_loader = create_loaders(
            args.train_dir,
            args.val_dir,
            args.train_list[task_idx],
            args.val_list[task_idx],
            args.shorter_side[task_idx],
            args.crop_size[task_idx],
            args.low_scale[task_idx],
            args.high_scale[task_idx],
            args.normalise_params,
            args.batch_size[task_idx],
            args.num_workers,
            args.ignore_label,
        )
        if args.evaluate:
            return validate(
                segmenter, val_loader, 0, num_classes=args.num_classes[task_idx]
            )

        logger.info(" Training Stage {}".format(str(task_idx)))
        ## Optimisers ##
        enc_params = []
        dec_params = []
        for k, v in segmenter.named_parameters():
            if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
                enc_params.append(v)
                logger.info(" Enc. parameter: {}".format(k))
            else:
                dec_params.append(v)
                logger.info(" Dec. parameter: {}".format(k))
        optim_enc, optim_dec = create_optimisers(
            args.lr_enc[task_idx],
            args.lr_dec[task_idx],
            args.mom_enc[task_idx],
            args.mom_dec[task_idx],
            args.wd_enc[task_idx],
            args.wd_dec[task_idx],
            enc_params,
            dec_params,
            args.optim_dec,
        )
        for epoch in range(args.num_segm_epochs[task_idx]):
            train_segmenter(
                segmenter,
                train_loader,
                optim_enc,
                optim_dec,
                epoch_start,
                segm_crit,
                args.freeze_bn[task_idx],
            )
            if (epoch + 1) % (args.val_every[task_idx]) == 0:
                miou = validate(
                    segmenter, val_loader, epoch_start, args.num_classes[task_idx]
                )
                saver.save(
                    miou,
                    {"segmenter": segmenter.state_dict(), "epoch_start": epoch_start},
                    logger,
                )
            epoch_start += 1
        logger.info(
            "Stage {} finished, time spent {:.3f}min".format(
                task_idx, (time.time() - start) / 60.0
            )
        )
    logger.info(
        "All stages are now finished. Best Val is {:.3f}".format(saver.best_val)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
