import argparse

import pandas as pd
import segmentation_models_pytorch as smp

from datetime import datetime
from pprint import pprint
from torch import optim
from torch.utils.data import DataLoader

from config import Config
from utils import CellDataset
from Training import (make_model,
                       MixedLoss,
                       train)

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--model_name', "-m", type=str, choices=['unet'],
                    help='the name of the model')
parser.add_argument('--backbone', "-b", type=str, default="resnet34", choices=['resnet34'],
                    help='the name of the backbone. Defaults to `resnet34`')
parser.add_argument('--params_path', "-p", type=str, default="config/params.yaml",
                    help='The path to the parameters.yaml file. Defaults to `config/params.yaml`')
parser.add_argument('--log', "-l", type=str, default="True",
                    help='Boolean representing whether to log metrics to wandb or not. Defaults to `True`')
parser.add_argument('--checkpoint', "-c", type=str, default="True",
                    help='Boolean representing whether to save the model or not. Defaults to `True`')

args = parser.parse_args().__dict__


if __name__ == "__main__":
    
    MODEL_NAME = args["model_name"]
    CONFIG_PATH = args["params_path"]
    LOG = args["log"] == "True"
    BACKBONE = args["backbone"]
    CHECKPOINT = args["checkpoint"]
    
    print(f"\nLoading configuration from `{CONFIG_PATH}`...")
    config = Config(model_name=MODEL_NAME,
                    backbone=BACKBONE,
                    config_path=CONFIG_PATH)
    print("Loading configuration complete.\n")
    pprint(config.cfg)
    
    print("\nLoading training data...")
    df_train = pd.read_csv(config.TRAIN_CSV)
    print("Loading training data complete.\n")
    print(df_train.info())
    
    # Dataset/loader prep
    print("\nConfiguring data...")
    ds_train = CellDataset(df_train, config=config)
    dl_train = DataLoader(ds_train, 
                          batch_size=config.BATCH_SIZE, 
                          num_workers=4, 
                          pin_memory=True, 
                          shuffle=False)
    print("\nConfiguring data complete.")
    
    print("\nCreating model...")
    model = make_model(model_name=MODEL_NAME, config=config)
    print("\nCreating model complete.")
    
    print("\nConfiguring hyperparameters...")
    hyperparams = config.configure_hyperparameters(keys=set(["iou", "dice_loss", "adam"]), 
                                                   model=model)
    print("\nConfiguring hyperparameters complete.")
    
    start = datetime.now()
    
    print(f"\nConfiguration setup complete. Training began at {start} ...\n")
    train(model_name=MODEL_NAME,
          dataset=ds_train,
          config=config,
          log=LOG,
          checkpoint=CHECKPOINT,
          kwargs=hyperparams)
    
    end = datetime.now()
    train_time = end - start
    print(f"\nTraining complete. Total training time {train_time}.")
    
    