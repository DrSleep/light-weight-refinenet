import logging
import re

from models.mobilenet import mbv2
from models.resnet import rf_lw50, rf_lw101, rf_lw152


def get_segmenter(
    enc_backbone, enc_pretrained, num_classes,
):
    """Create Encoder-Decoder; for now only ResNet [50,101,152] Encoders are supported"""
    if enc_backbone == "50":
        return rf_lw50(num_classes, imagenet=enc_pretrained)
    elif enc_backbone == "101":
        return rf_lw101(num_classes, imagenet=enc_pretrained)
    elif enc_backbone == "152":
        return rf_lw152(num_classes, imagenet=enc_pretrained)
    elif enc_backbone == "mbv2":
        return mbv2(num_classes, imagenet=enc_pretrained)
    else:
        raise ValueError("{} is not supported".format(str(enc_backbone)))


def get_encoder_and_decoder_params(model):
    """Filter model parameters into two groups: encoder and decoder."""
    logger = logging.getLogger(__name__)
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            enc_params.append(v)
            logger.info(" Enc. parameter: {}".format(k))
        else:
            dec_params.append(v)
            logger.info(" Dec. parameter: {}".format(k))
    return enc_params, dec_params
