from densetorch.misc import broadcast


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
      num_stages (int): broadcast training parameters to have this length.

    Returns:
      train_transforms, val_transforms

    """
    from torchvision.transforms import Compose
    from densetorch.data import (
        Pad,
        RandomCrop,
        RandomMirror,
        ResizeAndScale,
        ToTensor,
        Normalise,
    )

    common_transformations = [
        Normalise(scale=img_scale, mean=img_mean, std=img_std),
        ToTensor(),
    ]
    crop_size, shorter_side, low_scale, high_scale = [
        broadcast(param, num_stages)
        for param in (crop_size, shorter_side, low_scale, high_scale)
    ]
    train_transforms = []
    for stage in range(num_stages):
        train_transforms.append(
            Compose(
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
    return train_transforms, Compose(common_transformations)


def get_datasets(
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    train_transforms,
    val_transforms,
    masks_names,
):
    from densetorch.data import MMDataset as Dataset

    # Broadcast train dir to have the same length as train_transforms
    train_dir *= len(train_transforms)
    train_list_path *= len(train_transforms)

    def line_to_paths_fn(x):
        rgb, segm = x.decode("utf-8").strip("\n").split("\t")
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
