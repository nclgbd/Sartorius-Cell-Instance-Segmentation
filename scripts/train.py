import argparse

import pandas as pd
import segmentation_models_pytorch as smp
import time
import wandb

from datetime import datetime
from pprint import pprint
from torch import optim
from torch.utils.data import DataLoader

from config import Config
from Utilities import CellDataset
from Training import make_model, MixedLoss, train

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument(
    "--model_name",
    "-m",
    type=str,
    choices=["unet", "unetplusplus"],
    help="the name of the model",
)
parser.add_argument(
    "--backbone",
    "-b",
    type=str,
    default="resnet34",
    choices=["resnet34"],
    help="the name of the backbone. Defaults to `resnet34`",
)
parser.add_argument(
    "--params_path",
    "-p",
    type=str,
    default="config/params.yaml",
    help="The path to the parameters.yaml file. Defaults to `config/params.yaml`",
)
parser.add_argument(
    "--log",
    "-l",
    type=str,
    default="True",
    help="Boolean representing whether to log metrics to wandb or not. Defaults to `True`",
)
parser.add_argument(
    "--checkpoint",
    "-c",
    type=str,
    default="True",
    help="Boolean representing whether to save the model or not. Defaults to `True`",
)
parser.add_argument(
    "--sweep",
    "-s",
    type=str,
    default="False",
    help="Boolean representing whether to conduct a sweep. Currently unimplemented. Defaults to `False`",
)

args = parser.parse_args().__dict__

def main():
    MODEL_NAME = args["model_name"]
    CONFIG_PATH = args["params_path"]
    LOG = args["log"] == "True"
    BACKBONE = args["backbone"]
    CHECKPOINT = args["checkpoint"]
    SWEEP = args["sweep"] == "True"

    print(f"\nLoading configuration from `{CONFIG_PATH}`...")
    config = Config(
        model_name=MODEL_NAME,
        backbone=BACKBONE,
        log=LOG,
        checkpoint=CHECKPOINT,
        sweep=SWEEP,
        config_path=CONFIG_PATH,
    )
    print("Loading configuration complete.\n")
    pprint(config.cfg)
    train(config=config)


if __name__ == "__main__":
    main()
