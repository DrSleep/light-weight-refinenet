import numpy as np
import pytest
import random
import torch

import densetorch as dt

from network import get_segmenter, get_encoder_and_decoder_params


NUMBER_OF_PARAMETERS_WITH_21_CLASSES = {
    "152": 61993301,
    "101": 46349653,
    "50": 27357525,
    "mbv2": 3284565,
}

NUMBER_OF_ENCODER_DECODER_LAYERS = {
    "152": (465, 28),
    "101": (312, 28),
    "50": (159, 28),
    "mbv2": (156, 27),
}


def get_dummy_input_tensor(height, width, channels=3, batch=4):
    input_tensor = torch.FloatTensor(batch, channels, height, width).float()
    return input_tensor


def get_network_output_shape(h, w, output_stride=4):
    return np.ceil(h / output_stride), np.ceil(w / output_stride)


@pytest.fixture()
def num_classes():
    return random.randint(1, 40)


@pytest.fixture()
def input_height():
    return random.randint(33, 320)


@pytest.fixture()
def input_width():
    return random.randint(33, 320)


@pytest.mark.parametrize("enc_backbone", ["50", "101", "152", "mbv2"])
@pytest.mark.parametrize("enc_pretrained", [False, True])
def test_networks(enc_backbone, enc_pretrained, num_classes, input_height, input_width):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = (
        get_segmenter(
            enc_backbone=enc_backbone,
            enc_pretrained=enc_pretrained,
            num_classes=num_classes,
        )
        .eval()
        .to(device)
    )
    if num_classes == 21:
        assert (
            dt.misc.compute_params(network)
            == NUMBER_OF_PARAMETERS_WITH_21_CLASSES[enc_backbone]
        )

    enc_params, dec_params = get_encoder_and_decoder_params(network)
    n_enc_layers, n_dec_layers = NUMBER_OF_ENCODER_DECODER_LAYERS[enc_backbone]
    assert len(enc_params) == n_enc_layers
    assert len(dec_params) == n_dec_layers

    with torch.no_grad():
        input_tensor = get_dummy_input_tensor(
            height=input_height, width=input_width
        ).to(device)
        output_h, output_w = get_network_output_shape(*input_tensor.shape[-2:])
        output = network(input_tensor)
        assert output.size(1) == num_classes
        assert output.size(2) == output_h
        assert output.size(3) == output_w
