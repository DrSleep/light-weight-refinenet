import numpy as np
import pytest
import random
import torch

from densetorch.misc import broadcast

from data import get_transforms


def get_dummy_image_and_mask(size=(512, 512)):
    image = np.random.randint(low=0, high=255, size=size + (3,)).astype(np.float32)
    mask = np.random.randint(low=0, high=15, size=size, dtype=np.uint8)
    return image, mask


def pack_sample(image, mask, dataset_type):
    image = image.copy()
    mask = mask.copy()
    if dataset_type == "densetorch":
        sample = ({"image": image, "mask": mask, "names": ("mask",)},)
    elif dataset_type == "torchvision":
        sample = (image, mask)
    return sample


def unpack_sample(sample, dataset_type):
    if dataset_type == "densetorch":
        image = sample["image"]
        mask = sample["mask"]
    elif dataset_type == "torchvision":
        image, mask = sample
    return image, mask


@pytest.fixture()
def num_stages():
    return random.randint(1, 5)


@pytest.fixture()
def crop_size():
    crop_size = random.randint(160, 960)
    if crop_size % 2 == 1:
        # NOTE: In DenseTorch, the crop is always even.
        crop_size -= 1
    return crop_size


@pytest.fixture()
def shorter_side():
    return random.randint(160, 960)


@pytest.fixture()
def low_scale():
    return random.random()


@pytest.fixture()
def high_scale():
    return random.random()


@pytest.mark.parametrize("augmentations_type", ["densetorch", "albumentations"])
@pytest.mark.parametrize("dataset_type", ["densetorch", "torchvision"])
def test_transforms(
    augmentations_type,
    crop_size,
    dataset_type,
    num_stages,
    shorter_side,
    low_scale,
    high_scale,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    img_scale=1.0 / 255,
    ignore_label=255,
):
    train_transforms, val_transforms = get_transforms(
        crop_size=broadcast(crop_size, num_stages),
        shorter_side=broadcast(shorter_side, num_stages),
        low_scale=broadcast(low_scale, num_stages),
        high_scale=broadcast(high_scale, num_stages),
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
        img_scale=1.0 / 255,
        ignore_label=255,
        num_stages=num_stages,
        augmentations_type=augmentations_type,
        dataset_type=dataset_type,
    )
    assert len(train_transforms) == num_stages
    for is_val, transform in zip(
        [False] * num_stages + [True], train_transforms + [val_transforms]
    ):
        image, mask = get_dummy_image_and_mask()
        sample = pack_sample(image=image, mask=mask, dataset_type=dataset_type)
        output = transform(*sample)
        image_output, mask_output = unpack_sample(
            sample=output, dataset_type=dataset_type
        )
        # Test shape
        if not is_val:
            assert (
                image_output.shape[-2:]
                == mask_output.shape[-2:]
                == (crop_size, crop_size)
            )
        # Test that the outputs are torch tensors
        assert isinstance(image_output, torch.Tensor)
        assert isinstance(mask_output, torch.Tensor)
        # Test that there are no new segmentation classes, except for probably ignore_label
        uq_classes_before = np.unique(mask)
        uq_classes_after = np.unique(mask_output.numpy())
        assert (
            len(
                np.setdiff1d(
                    uq_classes_after, uq_classes_before.tolist() + [ignore_label]
                )
            )
            == 0
        )
        if is_val:
            # Test that for validation transformation the output shape has not changed
            assert (
                image_output.shape[-2:]
                == image.shape[:2]
                == mask_output.shape[-2:]
                == mask.shape[:2]
            )
            # Test that there were no changes to the classes at all
            assert all(uq_classes_before == uq_classes_after)
