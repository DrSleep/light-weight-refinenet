import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))  # noqa: E402
from models.mobilenet import mbv2
from models.resnet import rf_lw50, rf_lw101, rf_lw152
from utils.helpers import prepare_img

from class_info import CLASS_NAMES


DATASETS_CLASSES = {
    "PASCAL Person": 7,
    "PASCAL VOC": 21,
    "NYUDv2": 40,
    "PASCAL Context": 60,
}
SUPPORTED_DATASETS = {
    "50": ["PASCAL Person", "PASCAL VOC", "NYUDv2"],
    "101": ["PASCAL Person", "PASCAL VOC", "NYUDv2", "PASCAL Context"],
    "152": ["PASCAL Person", "PASCAL VOC", "NYUDv2", "PASCAL Context"],
    "mbv2": ["PASCAL VOC"],
}


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="Light-Weight-RefineNet segment and save demo."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument(
        "--enc-backbone", type=str, choices=["50", "101", "152", "mbv2"], default="50",
    )
    parser.add_argument(
        "--run-on-cpu",
        action="store_true",
        help="If provided, will run on CPU even if GPU is available.",
    )
    args = parser.parse_args()
    return args


def get_segmenter(
    enc_backbone, device,
):
    """Create Encoder-Decoder; ResNet [50,101,152] and MobileNet-v2 encoders are supported"""
    supported_datasets = SUPPORTED_DATASETS[enc_backbone]
    print(f"For the backbone '{enc_backbone:s}' the following datasets are supported:")
    for i, dataset in enumerate(supported_datasets):
        print(f"{i:d}. {dataset:s} with {DATASETS_CLASSES[dataset]:d} output classes.")
    while True:
        answer = input(
            f"Please select dataset. Type a digit in range [0, {len(supported_datasets) - 1:d}]."
        )
        if (
            not answer.isnumeric()
            or (int(answer) >= len(supported_datasets))
            or (int(answer) < 0)
        ):
            print("Incorrect choice. Try again.")
            continue
        break
    dataset = supported_datasets[int(answer)]
    num_classes = DATASETS_CLASSES[dataset]
    if enc_backbone == "50":
        net = rf_lw50
    elif enc_backbone == "101":
        net = rf_lw101
    elif enc_backbone == "152":
        net = rf_lw152
    elif enc_backbone == "mbv2":
        net = mbv2
    else:
        raise ValueError("{} is not supported".format(str(enc_backbone)))
    return net(num_classes, pretrained=True).eval().to(device), dataset


def prepare_image(image_path, device):
    image_pillow = Image.open(image_path).convert("RGB")
    image_numpy_hwc = np.array(image_pillow)
    normalised_image_hwc = prepare_img(image_numpy_hwc)
    normalised_image_1chw = normalised_image_hwc.transpose(2, 0, 1)[None]
    return image_pillow, torch.from_numpy(normalised_image_1chw).to(device)


def save_image_and_mask(image, mask, chosen_class, choices):
    if chosen_class == "":
        cmap = np.load("utils/cmap.npy")
        coloured_mask = cmap[mask]
        Image.fromarray(coloured_mask).save("coloured_mask.png")
    selected = mask == choices[int(chosen_class)]
    binary_mask = Image.fromarray(selected.astype(np.uint8) * 255).convert("L")
    image.putalpha(binary_mask)
    binary_mask.save("binary_mask.png")
    image.save("image.png")


def main():
    args = get_arguments()
    device = "cpu"
    if torch.cuda.is_available() and not args.force_run_on_cpu:
        device = "cuda"
    segmenter, dataset = get_segmenter(args.enc_backbone, device)
    input_image, input_tensor = prepare_image(args.image_path, device)
    with torch.no_grad():
        output = segmenter(input_tensor.float())
        output = F.interpolate(
            output, size=input_tensor.shape[-2:], mode="bilinear", align_corners=True
        )
        output_np = output.squeeze(0).cpu().numpy()
        output_mask = output_np.argmax(0)
        print("The following classes were recognised in the provided image:")
        uq_classes = np.unique(output_mask)
        for i, uq_class in enumerate(uq_classes):
            print(f"{i:d}. {CLASS_NAMES[dataset][uq_class]:s}.")
        while True:
            answer = input(
                f"Please select the class of which the image and the binary mask are to be saved. "
                f"Type a digit in range [0, {i:d}]. "
                f"Leave empty if you want to save "
                f"the multi-class segmentation mask of the whole image."
            )
            if answer == "":
                break
            elif not answer.isnumeric() or (int(answer) > i) or (int(answer) < 0):
                print("Incorrect choice. Try again.")
                continue
            break
        save_image_and_mask(input_image, output_mask, answer, uq_classes)


if __name__ == "__main__":
    main()
