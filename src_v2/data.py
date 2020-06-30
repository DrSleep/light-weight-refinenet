import cv2
import numpy as np
import torch


def albumentations2torchvision(transforms):
    """Wrap albumentations transformation so that they can be used in torchvision dataset"""
    from albumentations import Compose

    def wrapper_func(image, target):
        keys = ["image", "mask"]
        np_dtypes = [np.float32, np.uint8]
        torch_dtypes = [torch.float32, torch.long]
        sample_dict = {
            key: np.array(value, dtype=dtype)
            for key, value, dtype in zip(keys, [image, target], np_dtypes)
        }
        output = Compose(transforms)(**sample_dict)
        return [output[key].to(dtype) for key, dtype in zip(keys, torch_dtypes)]

    return wrapper_func


def albumentations_transforms(
    crop_size,
    shorter_side,
    low_scale,
    high_scale,
    img_mean,
    img_std,
    img_scale,
    ignore_label,
    num_stages,
    dataset_type,
):
    from albumentations import (
        Normalize,
        HorizontalFlip,
        RandomCrop,
        PadIfNeeded,
        RandomScale,
        LongestMaxSize,
        SmallestMaxSize,
        OneOf,
    )
    from albumentations.pytorch import ToTensorV2 as ToTensor
    from densetorch.data import albumentations2densetorch

    if dataset_type == "densetorch":
        wrapper = albumentations2densetorch
    elif dataset_type == "torchvision":
        wrapper = albumentations2torchvision
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    common_transformations = [
        Normalize(max_pixel_value=1.0 / img_scale, mean=img_mean, std=img_std),
        ToTensor(),
    ]
    train_transforms = []
    for stage in range(num_stages):
        train_transforms.append(
            wrapper(
                [
                    OneOf(
                        [
                            RandomScale(
                                scale_limit=(low_scale[stage], high_scale[stage])
                            ),
                            LongestMaxSize(max_size=shorter_side[stage]),
                            SmallestMaxSize(max_size=shorter_side[stage]),
                        ]
                    ),
                    PadIfNeeded(
                        min_height=crop_size[stage],
                        min_width=crop_size[stage],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=np.array(img_mean) / img_scale,
                        mask_value=ignore_label,
                    ),
                    HorizontalFlip(p=0.5,),
                    RandomCrop(height=crop_size[stage], width=crop_size[stage],),
                ]
                + common_transformations
            )
        )
    val_transforms = wrapper(common_transformations)
    return train_transforms, val_transforms


def densetorch_transforms(
    crop_size,
    shorter_side,
    low_scale,
    high_scale,
    img_mean,
    img_std,
    img_scale,
    ignore_label,
    num_stages,
    dataset_type,
):
    from torchvision.transforms import Compose
    from densetorch.data import (
        Pad,
        RandomCrop,
        RandomMirror,
        ResizeAndScale,
        ToTensor,
        Normalise,
        densetorch2torchvision,
    )

    if dataset_type == "densetorch":
        wrapper = Compose
    elif dataset_type == "torchvision":
        wrapper = densetorch2torchvision
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    common_transformations = [
        Normalise(scale=img_scale, mean=img_mean, std=img_std),
        ToTensor(),
    ]
    train_transforms = []
    for stage in range(num_stages):
        train_transforms.append(
            wrapper(
                [
                    ResizeAndScale(
                        shorter_side[stage], low_scale[stage], high_scale[stage]
                    ),
                    Pad(crop_size[stage], img_mean, ignore_label),
                    RandomMirror(),
                    RandomCrop(crop_size[stage]),
                ]
                + common_transformations
            )
        )
    val_transforms = wrapper(common_transformations)
    return train_transforms, val_transforms


def get_transforms(
    crop_size,
    shorter_side,
    low_scale,
    high_scale,
    img_mean,
    img_std,
    img_scale,
    ignore_label,
    num_stages,
    augmentations_type,
    dataset_type,
):
    """
    Args:

      crop_size (int) : square crop to apply during the training.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      img_mean (list of float) : image mean.
      img_std (list of float) : image standard deviation
      img_scale (list of float) : image scale.
      ignore_label (int) : label to pad segmentation masks with.
      num_stages (int): how many train_transforms to create.
      augmentations_type (str): whether to use densetorch augmentations or albumentations.
      dataset_type (str): whether to use densetorch or torchvision dataset;
                            needed to correctly wrap transformations.

    Returns:
      train_transforms, val_transforms

    """
    if augmentations_type == "densetorch":
        func = densetorch_transforms
    elif augmentations_type == "albumentations":
        func = albumentations_transforms
    else:
        raise ValueError(f"Unknown augmentations type {augmentations_type}")
    return func(
        crop_size=crop_size,
        shorter_side=shorter_side,
        low_scale=low_scale,
        high_scale=high_scale,
        img_mean=img_mean,
        img_std=img_std,
        img_scale=img_scale,
        ignore_label=ignore_label,
        num_stages=num_stages,
        dataset_type=dataset_type,
    )


def densetorch_dataset(
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    train_transforms,
    val_transforms,
    masks_names,
    stage_names,
    train_download,
    val_download,
):
    from densetorch.data import MMDataset as Dataset

    def line_to_paths_fn(x):
        rgb, segm = x.decode("utf-8").strip("\n").split("\t")[:2]
        return [rgb, segm]

    train_sets = [
        Dataset(
            data_file=train_list_path[i],
            data_dir=train_dir[i],
            line_to_paths_fn=line_to_paths_fn,
            masks_names=masks_names,
            transform=train_transforms[i],
        )
        for i in range(len(train_transforms))
    ]
    val_set = Dataset(
        data_file=val_list_path,
        data_dir=val_dir,
        line_to_paths_fn=line_to_paths_fn,
        masks_names=masks_names,
        transform=val_transforms,
    )
    return train_sets, val_set


def torchvision_dataset(
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    train_transforms,
    val_transforms,
    masks_names,
    stage_names,
    train_download,
    val_download,
):
    from torchvision.datasets.voc import VOCSegmentation
    from torchvision.datasets import SBDataset
    from functools import partial

    train_sets = []
    for i, stage in enumerate(stage_names):
        if stage.lower() == "voc":
            Dataset = partial(VOCSegmentation, image_set="train", year="2012",)
        elif stage.lower() == "sbd":
            Dataset = partial(SBDataset, mode="segmentation", image_set="train_noval")
        train_sets.append(
            Dataset(
                root=train_dir[i],
                transforms=train_transforms[i],
                download=train_download[i],
            )
        )

    val_set = VOCSegmentation(
        root=val_dir,
        image_set="val",
        year="2012",
        download=val_download,
        transforms=val_transforms,
    )

    return train_sets, val_set


def get_datasets(
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    train_transforms,
    val_transforms,
    masks_names,
    dataset_type,
    stage_names,
    train_download,
    val_download,
):
    if dataset_type == "densetorch":
        func = densetorch_dataset
    elif dataset_type == "torchvision":
        func = torchvision_dataset
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")
    return func(
        train_dir,
        val_dir,
        train_list_path,
        val_list_path,
        train_transforms,
        val_transforms,
        masks_names,
        stage_names,
        train_download,
        val_download,
    )
