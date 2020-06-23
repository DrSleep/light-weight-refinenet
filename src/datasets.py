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

from __future__ import print_function, division

import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class Pad(object):
    """Pad image and mask to the desired size

    Args:
      size (int) : minimum length/width
      img_val (array) : image padding value
      msk_val (int) : mask padding value

    """

    def __init__(self, size, img_val, msk_val):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        image = np.stack(
            [
                np.pad(
                    image[:, :, c],
                    pad,
                    mode="constant",
                    constant_values=self.img_val[c],
                )
                for c in range(3)
            ],
            axis=2,
        )
        mask = np.pad(mask, pad, mode="constant", constant_values=self.msk_val)
        return {"image": image, "mask": mask}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top : top + new_h, left : left + new_w]
        mask = mask[top : top + new_h, left : left + new_w]
        return {"image": image, "mask": mask}


class ResizeShorterScale(object):
    """Resize shorter side to a given value and randomly scale."""

    def __init__(self, shorter_side, low_scale, high_scale):
        assert isinstance(shorter_side, int)
        self.shorter_side = shorter_side
        self.low_scale = low_scale
        self.high_scale = high_scale

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        min_side = min(image.shape[:2])
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if min_side * scale < self.shorter_side:
            scale = self.shorter_side * 1.0 / min_side
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        mask = cv2.resize(
            mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return {"image": image, "mask": mask}


class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        do_mirror = np.random.randint(2)
        if do_mirror:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return {"image": image, "mask": mask}


class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, scale, mean, std):
        self.scale = scale
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample["image"]
        return {
            "image": (self.scale * image - self.mean) / self.std,
            "mask": sample["mask"],
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


class NYUDataset(Dataset):
    """NYUv2-40"""

    def __init__(self, data_file, data_dir, transform_trn=None, transform_val=None):
        """
        Args:
            data_file (string): Path to the data file with annotations.
            data_dir (string): Directory with all the images.
            transform_{trn, val} (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [
            (k, v)
            for k, v in map(
                lambda x: x.decode("utf-8").strip("\n").split("\t"), datalist
            )
        ]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = "train"

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.datalist[idx][0])
        msk_name = os.path.join(self.root_dir, self.datalist[idx][1])

        def read_image(x):
            img_arr = np.array(Image.open(x))
            if len(img_arr.shape) == 2:  # grayscale
                img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
            return img_arr

        image = read_image(img_name)
        mask = np.array(Image.open(msk_name))
        if img_name != msk_name:
            assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
        sample = {"image": image, "mask": mask}
        if self.stage == "train":
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == "val":
            if self.transform_val:
                sample = self.transform_val(sample)
        return sample
