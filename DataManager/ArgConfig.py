"""Typed command-line configuration, following the uploaded ArgConfig design."""
import argparse
from dataclasses import dataclass, asdict, replace

@dataclass
class ArgConfig:
    mode: str = "train"
    width_mult: float = 1.0
    dropout: float = 0.2
    use_batchnorm: bool = True
    optimizer: str = "sgd"
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_scheduler: str = "cosine"
    step_size: int = 30
    gamma: float = 0.1
    seed: int = 42
    num_workers: int = 2
    device: str = "auto"
    data_dir: str = "./CIFAR10Dataset"
    output_dir: str = "./Results"
    calc_meanstd: bool = True
    normalize_mean: tuple = (0.4914, 0.4822, 0.4465)
    normalize_std: tuple = (0.2470, 0.2435, 0.2616)
    add_crop: bool = True
    crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
    add_jitter: bool = True
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
    add_random_erasing: bool = False
    random_erasing_prob: float = 0.25
    random_erasing_scale: tuple = (0.02, 0.1)
    random_erasing_ratio: tuple = (0.5, 2.0)
    wandb_project: str = "CS6886W_Assignment2_MobileNetV2"
    wandb_entity: str = None
    wandb_mode: str = "online"
    wandb_group: str = "Q1_Controlled"
    run_name: str = None
    no_wandb: bool = False
    no_download: bool = False
    run_dir: str = None

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser(description="CS6886W MobileNetV2: same OO design")
        parser.add_argument("--mode", choices=["train", "test"], default="train")
        parser.add_argument("--width_mult", "--width-mult", type=float, default=1.0)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--use_batchnorm", "--use-batchnorm", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--optimizer", choices=["sgd","nesterov_sgd","adam","adamw","rmsprop"], default="sgd")
        parser.add_argument("--batch_size", "--batch-size", type=int, default=128)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--learning_rate", "--learning-rate", type=float, default=0.05)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.0005)
        parser.add_argument("--lr_scheduler", "--lr-scheduler", choices=["cosine","step","none"], default="cosine")
        parser.add_argument("--step_size", type=int, default=30)
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
        parser.add_argument("--data_dir", default="./CIFAR10Dataset")
        parser.add_argument("--output_dir", default="./Results")
        parser.add_argument("--calc_meanstd", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--normalize_mean", type=float, nargs=3, default=(0.4914, 0.4822, 0.4465))
        parser.add_argument("--normalize_std", type=float, nargs=3, default=(0.2470, 0.2435, 0.2616))
        parser.add_argument("--add_crop", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--crop_padding", type=int, default=4)
        parser.add_argument("--horizontal_flip_prob", type=float, default=0.5)
        parser.add_argument("--add_jitter", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--jitter_brightness", type=float, default=0.2)
        parser.add_argument("--jitter_contrast", type=float, default=0.2)
        parser.add_argument("--jitter_saturation", type=float, default=0.2)
        parser.add_argument("--jitter_hue", type=float, default=0.05)
        parser.add_argument("--add_random_erasing", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--random_erasing_prob", type=float, default=0.25)
        parser.add_argument("--random_erasing_scale", nargs=2, type=float, default=(0.02,0.1))
        parser.add_argument("--random_erasing_ratio", nargs=2, type=float, default=(0.5,2.0))
        parser.add_argument("--wandb_project", default="CS6886W_Assignment2_MobileNetV2")
        parser.add_argument("--wandb_entity", default=None)
        parser.add_argument("--wandb_mode", choices=["online","offline","disabled"], default="online")
        parser.add_argument("--wandb_group", default="Q1_Controlled")
        parser.add_argument("--run_name", default=None)
        parser.add_argument("--no_wandb", action="store_true")
        parser.add_argument("--no_download", action="store_true")
        parser.add_argument("--run_dir", default=None, help="Existing experiment directory for --mode test")
        args = cls(**vars(parser.parse_args()))
        args.validate()
        return args

    def validate(self):
        if self.width_mult <= 0 or not 0 <= self.dropout < 1: raise ValueError("Invalid width/dropout")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0: raise ValueError("Invalid training settings")
        if self.num_workers < 0: raise ValueError("num_workers must be nonnegative")
        if not 0 <= self.horizontal_flip_prob <= 1 or not 0 <= self.jitter_hue <= 0.5: raise ValueError("Invalid augmentation settings")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3 or min(self.normalize_std) <= 0: raise ValueError("Invalid RGB normalization")

    def as_dict(self):
        return asdict(self)
