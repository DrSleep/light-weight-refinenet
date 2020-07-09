import argparse

from densetorch.misc import broadcast


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="Arguments for Light-Weight-RefineNet Training Pipeline"
    )

    # Common transformations
    parser.add_argument("--img-scale", type=float, default=1.0 / 255)
    parser.add_argument(
        "--img-mean", type=float, nargs=3, default=(0.485, 0.456, 0.406)
    )
    parser.add_argument("--img-std", type=float, nargs=3, default=(0.229, 0.224, 0.225))

    # Training augmentations
    parser.add_argument(
        "--augmentations-type",
        type=str,
        choices=["densetorch", "albumentations"],
        default="densetorch",
    )

    # Dataset
    parser.add_argument(
        "--val-list-path", type=str, default="./data/val.nyu",
    )
    parser.add_argument(
        "--val-dir", type=str, default="./datasets/nyud/",
    )
    parser.add_argument("--val-batch-size", type=int, default=1)

    # Optimisation
    parser.add_argument(
        "--enc-optim-type", type=str, default="sgd",
    )
    parser.add_argument(
        "--dec-optim-type", type=str, default="sgd",
    )
    parser.add_argument(
        "--enc-lr", type=float, default=5e-4,
    )
    parser.add_argument(
        "--dec-lr", type=float, default=5e-3,
    )
    parser.add_argument(
        "--enc-weight-decay", type=float, default=1e-5,
    )
    parser.add_argument(
        "--dec-weight-decay", type=float, default=1e-5,
    )
    parser.add_argument(
        "--enc-momentum", type=float, default=0.9,
    )
    parser.add_argument(
        "--dec-momentum", type=float, default=0.9,
    )
    parser.add_argument(
        "--enc-lr-gamma",
        type=float,
        default=0.5,
        help="Multilpy lr_enc by this value after each stage.",
    )
    parser.add_argument(
        "--dec-lr-gamma",
        type=float,
        default=0.5,
        help="Multilpy lr_dec by this value after each stage.",
    )
    parser.add_argument(
        "--enc-scheduler-type",
        type=str,
        choices=["poly", "multistep"],
        default="multistep",
    )
    parser.add_argument(
        "--dec-scheduler-type",
        type=str,
        choices=["poly", "multistep"],
        default="multistep",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="Ignore this label in the training loss.",
    )
    parser.add_argument("--random-seed", type=int, default=42)

    # Training / validation setup
    parser.add_argument(
        "--enc-backbone", type=str, choices=["50", "101", "152", "mbv2"], default="50"
    )
    parser.add_argument("--enc-pretrained", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--num-stages",
        type=int,
        default=3,
        help="Number of training stages. All other arguments with nargs='+' must "
        "have the number of arguments equal to this value. Otherwise, the given "
        "arguments will be broadcasted to have the required length.",
    )
    parser.add_argument("--num-classes", type=int, default=40)
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="densetorch",
        choices=["densetorch", "torchvision"],
    )
    parser.add_argument(
        "--val-download",
        type=int,
        choices=[0, 1],
        default=0,
        help="Only used if dataset_type == torchvision.",
    )

    # Checkpointing configuration
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints/")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./checkpoints/checkpoint.pth.tar",
        help="Path to the checkpoint file.",
    )

    # Arguments broadcastable across training stages
    stage_parser = parser.add_argument_group("stage-parser")
    stage_parser.add_argument(
        "--crop-size", type=int, nargs="+", default=(500, 500, 500,)
    )
    stage_parser.add_argument(
        "--shorter-side", type=int, nargs="+", default=(350, 350, 350,)
    )
    stage_parser.add_argument(
        "--low-scale", type=float, nargs="+", default=(0.5, 0.5, 0.5,)
    )
    stage_parser.add_argument(
        "--high-scale", type=float, nargs="+", default=(2.0, 2.0, 2.0,)
    )
    stage_parser.add_argument(
        "--train-list-path", type=str, nargs="+", default=("./data/train.nyu",)
    )
    stage_parser.add_argument(
        "--train-dir", type=str, nargs="+", default=("./datasets/nyud/",)
    )
    stage_parser.add_argument(
        "--train-batch-size", type=int, nargs="+", default=(6, 6, 6,)
    )
    stage_parser.add_argument(
        "--freeze-bn", type=int, choices=[0, 1], nargs="+", default=(1, 1, 1,)
    )
    stage_parser.add_argument(
        "--epochs-per-stage", type=int, nargs="+", default=(100, 100, 100),
    )
    stage_parser.add_argument("--val-every", type=int, nargs="+", default=(5, 5, 5,))
    stage_parser.add_argument(
        "--stage-names",
        type=str,
        nargs="+",
        choices=["SBD", "VOC"],
        default=("SBD", "VOC",),
        help="Only used if dataset_type == torchvision.",
    )
    stage_parser.add_argument(
        "--train-download",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=(0, 0,),
        help="Only used if dataset_type == torchvision.",
    )
    stage_parser.add_argument(
        "--grad-norm",
        type=float,
        nargs="+",
        default=(0.0,),
        help="If > 0.0, clip gradients' norm to this value.",
    )
    args = parser.parse_args()
    # Broadcast all arguments in stage-parser
    for group_action in stage_parser._group_actions:
        argument_name = group_action.dest
        setattr(
            args,
            argument_name,
            broadcast(getattr(args, argument_name), args.num_stages),
        )
    return args
