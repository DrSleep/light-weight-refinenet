from models.resnet import rf_lw50, rf_lw101, rf_lw152


def get_segmenter(
    enc_backbone, enc_pretrained, num_classes,
):
    """Create Encoder; for now only ResNet [50,101,152]"""
    if enc_backbone == "50":
        return rf_lw50(num_classes, imagenet=enc_pretrained)
    elif enc_backbone == "101":
        return rf_lw101(num_classes, imagenet=enc_pretrained)
    elif enc_backbone == "152":
        return rf_lw152(num_classes, imagenet=enc_pretrained)
    else:
        raise ValueError("{} is not supported".format(str(enc_backbone)))
