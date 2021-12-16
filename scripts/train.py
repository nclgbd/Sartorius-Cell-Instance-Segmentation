import argparse
import os
import time
import torch
import wandb
import pandas as pd

from datetime import datetime
from pprint import pprint
from torch import nn
from torch import optim
from torch.utils.data import (DataLoader, 
                              Dataset)
from tqdm import tqdm
from statistics import (mean, 
                        stdev)
from utils import (create_loader,
                   make_model,
                   CellDataset)

from config import Config

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--model_name', "-m", type=str, choices=['unet'],
                    help='the name of the model')
parser.add_argument('--backbone', "-b", type=str, default="resnet34", choices=['resnet34'],
                    help='the name of the backbone. Defaults to `resnet34`')
parser.add_argument('--params_path', "-p", type=str, default="config/params.yaml",
                    help='The path to the parameters.yaml file. Defaults to `config/params.yaml`')
parser.add_argument('--log', "-l", type=str, default="False",
                    help='Boolean representing whether to log metrics to wandb or not. Defaults to `False`')

args = parser.parse_args().__dict__


def _get_kwargs(**kwargs):
    criterion = kwargs["criterion"]
    optimizer = kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]
    
    return (criterion, 
            optimizer, 
            scheduler)


def _loop_fn(model: nn.Module, fold: int, loader: DataLoader, config=None, log=False, mode="train", **kwargs):
    criterion, optimizer, scheduler = _get_kwargs(kwargs)
    
    if mode == "train":
        model.train()
    
    elif mode == "test":
        model.eval()
    
    n_batches = len(loader)
    for epoch in range(1, config.EPOCHS + 1):
        print(f"Fold {fold} Epoch: {epoch}")
        running_loss = running_iou = 0.0
        optimizer.zero_grad()
        
        loss = 0
        for images, masks in tqdm(loader, desc=mode.title()):
            # Predict
            images, masks = images.cuda(),  masks.cuda()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            
            # Back prop
            if mode == "train":
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            running_iou += (outputs.argmax(1) == masks).sum().item()
            
            if log:
                wandb.log({"loss": loss.item()})
        
            if scheduler:
                scheduler.step(running_loss)

        epoch_loss = running_loss / n_batches
        epoch_iou = running_iou / n_batches
        
        if log:
            wandb.log({"epoch_loss": epoch_loss})
            wandb.log({"epoch_iou": epoch_iou})
            wandb.watch(model, criterion, log_graph=True)
            
        return epoch_loss, epoch_iou


def _train(model_name, dataset: Dataset, config=None, log=False, **kwargs):
    
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    model = make_model(model_name, config=config)
    model.cuda()
    
    total_train_ious = total_train_losses = []
    total_valid_ious = total_valid_losses = []
    
    for idx, (train_idx, valid_idx) in dataset.folds:
        # Create loaders
        print(f"Fold {idx+1}")
        dl_train = create_loader(dataset, train_idx)
        dl_valid = create_loader(dataset, valid_idx)
        
        # Train and get metrics back
        train_epoch_loss, train_epoch_iou = _loop_fn(model,
                                                     fold=idx+1, 
                                                     loader=dl_train, 
                                                     log=log,
                                                     config=config,
                                                     mode="train",
                                                     kwargs=kwargs)
        
        valid_epoch_loss, valid_epoch_iou = _loop_fn(model,
                                                     fold=idx+1,
                                                     loader=dl_valid,
                                                     log=log,
                                                     config=config,
                                                     mode="test",
                                                     kwargs=kwargs)
        
        # Print epoch results
        print(f"Epoch {idx+1} |\tTrain loss {train_epoch_loss:.4f} -- Train IoU (Intersection over Union) {train_epoch_iou:.4f}")
        print(f"Epoch {idx+1} |\tValidation loss {valid_epoch_loss:.4f} -- Validation IoU (Intersection over Union) {valid_epoch_iou:.4f}")
        
        total_train_losses.append(train_epoch_loss)
        total_train_ious.append(train_epoch_iou)
        
        total_valid_losses.append(valid_epoch_loss)
        total_valid_ious.append(valid_epoch_iou)
        
    # Print all training and validation metrics    
    train_avg_iou = mean(total_train_ious)
    train_avg_iou_std = stdev(total_train_ious)
    
    train_avg_loss = mean(total_train_losses)
    train_avg_loss_std = stdev(total_train_losses)
    
    valid_avg_iou = mean(total_valid_ious)
    valid_avg_iou_std = stdev(total_valid_ious)
    
    valid_avg_loss = mean(total_valid_losses)
    valid_avg_loss_std = stdev(total_valid_losses)
    
    print(f"Average training IoU (Intersection over Union) of all folds:\t{train_avg_iou:.4f} +/- {train_avg_iou_std:.4f}")
    print(f"Average validation IoU (Intersection over Union) of all folds:\t{valid_avg_iou:.4f} +/- {valid_avg_iou_std:.4f}\n")
    
    print(f"Average training loss of all folds:\t{train_avg_loss:.4f} +/- {train_avg_loss_std:.4f}")
    print(f"Average validation loss of all folds:\t{valid_avg_loss:.4f} +/- {valid_avg_loss_std:.4f}")
    
    # Log the metrics if using wanb
    if log:
        avg_metrics = {}
        
        avg_metrics["train_avg_iou"] = train_avg_iou
        avg_metrics["train_avg_iou_std"] = train_avg_iou_std
        avg_metrics["train_avg_loss"] = train_avg_loss
        avg_metrics["train_avg_loss_std"] = train_avg_loss_std
        
        avg_metrics["valid_avg_iou"] = valid_avg_iou
        avg_metrics["valid_avg_iou_std"] = valid_avg_iou_std
        avg_metrics["valid_avg_loss"] = valid_avg_loss
        avg_metrics["valid_avg_loss_std"] = valid_avg_loss_std
        
        wandb.log(avg_metrics)
    

def train(model_name, dataset: Dataset, config=None, log=False, **kwargs):
    # wandb implementation
    if log:
        os.environ['WANDB_API_KEY'] = config["wandb_api_key"]
        
        run = wandb.init(project=config["project"], 
                        entity=config["entity"], 
                        config=config,
                        reinit=True)
        
        with run:
            config = wandb.config
            _train(model_name=model_name, 
                   config=config, 
                   dataset=dataset, 
                   log=log, 
                   kwargs=kwargs)
         
    # local implementation
    else:
        _train(model_name=model_name, 
               config=config, 
               dataset=dataset, 
               log=log,
               kwargs=kwargs)
    

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()
    
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
    

if __name__ == "__main__":
    
    MODEL_NAME = args["model_name"]
    CONFIG_PATH = args["params_path"]
    LOG = args["log"] == "True"
    BACKBONE = args["backbone"]
    
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
    print("\nConfiguring training...")
    ds_train = CellDataset(df_train, config=config)
    dl_train = DataLoader(ds_train, 
                          batch_size=config.BATCH_SIZE, 
                          num_workers=4, 
                          pin_memory=True, 
                          shuffle=False)
    
    model = make_model(model_name=MODEL_NAME, config=config)
    criterion = MixedLoss(10.0, 2.0)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    torch_args = {"criterion": criterion,
                  "optimizer": optimizer}
    
    start = datetime.now()
    
    print(f"\nConfiguration setup complete. Training began at {start} ...\n")
    train(model_name=MODEL_NAME,
          dataset=ds_train,
          config=config,
          log=LOG,
          kwargs=torch_args)
    
    end = datetime.now()
    train_time = end - start
    print(f"\nTraining complete. Total training time: {train_time}")
    
    