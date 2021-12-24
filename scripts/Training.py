import os
import torch
import wandb
import segmentation_models_pytorch as smp
import time
import pandas as pd

from datetime import datetime
from dotenv import dotenv_values
from pprint import pprint
from torch import nn
from torch.utils.data import (DataLoader, 
                              Dataset)
from torch.nn import functional as F
from tqdm import tqdm
from statistics import (mean, 
                        stdev)
from Utilities import (EarlyStopping, create_loader,
                   make_model, CellDataset)


def _init_train(model_name, config=None, checkpoint=True, run=None):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    model = make_model(model_name, config=config)
    model.cuda()
    early_stopping = None
    
    if checkpoint:
        early_stopping = EarlyStopping(model_dir=config.MODEL_PATH,
                                       model_name=model_name,
                                       run=run,
                                       config=config)
    
    return model, early_stopping


def _train(model_name, dataset: Dataset, config=None, run=None, device="cuda", **kwargs):
    total_train_ious = []
    total_train_losses = []
    total_valid_ious = []
    total_valid_losses = []
    
    kwargs = kwargs["kwargs"]
    criterion = kwargs["loss"]
    optimizer = kwargs["optimizer"]
    scheduler = None if "scheduler" not in list(kwargs.keys()) else kwargs["scheduler"]
    pprint(kwargs)
        
    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        model, early_stopping = _init_train(model_name,
                                        config=config,
                                        checkpoint=config.CHECKPOINT,
                                        run=run)
        if config.LOG:
            wandb.watch(model, criterion, log_graph=True)
            
        # Create loaders
        print(f"\nFold: {idx+1}\n--------")
        dataset.dl_train = dl_train = create_loader(dataset, train_idx, config=config)
        dataset.dl_valid = dl_valid = create_loader(dataset, valid_idx, config=config)
        
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            device=device,
            verbose=True,
            **kwargs
        )
        
        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            device=device,
            verbose=True,
            metrics=kwargs["metrics"],
            loss=kwargs["loss"]
        )
        
        for epoch in range(1, config.EPOCHS + 1):
            print(f"Epoch {epoch}")
            print()
            
            train_logs = train_epoch.run(dl_train)
            pprint(train_logs)
            
            valid_logs = valid_epoch.run(dl_valid)
            pprint(valid_logs)
            
            keys = list(train_logs.keys())
            if "mixed_loss" in keys:
                train_epoch_loss = train_logs['mixed_loss']
                valid_epoch_loss = valid_logs['mixed_loss']
            elif "dice_loss" in keys:
                train_epoch_loss = train_logs['dice_loss']
                valid_epoch_loss = valid_logs['dice_loss']
                
            train_epoch_iou = train_logs['iou_score']  
            valid_epoch_iou = valid_logs['iou_score']
            
            if config.LOG:
                wandb.log({"train_logs": train_logs})
                wandb.log({"valid_logs": valid_logs})
                
            # Print epoch results
            print(f"\nTrain loss: {train_epoch_loss:.4f}\t Train iou: {train_epoch_iou:.4f}")
            print(f"Validation loss: {valid_epoch_loss:.4f}\t Validation iou: {valid_epoch_iou:.4f}")
            
            total_train_losses.append(train_epoch_loss)
            total_train_ious.append(train_epoch_iou)
            
            total_valid_losses.append(valid_epoch_loss)
            total_valid_ious.append(valid_epoch_iou)
            
            if early_stopping:
                breakpoint = early_stopping.checkpoint(model,
                                                       epoch=epoch,
                                                       loss=valid_epoch_loss,
                                                       iou=valid_epoch_iou,
                                                       optimizer=optimizer)
            
                if breakpoint:
                    break
        
    # Print all training and validation metrics    
    train_avg_iou = mean(total_train_ious)
    train_avg_iou_std = stdev(total_train_ious)
    
    train_avg_loss = mean(total_train_losses)
    train_avg_loss_std = stdev(total_train_losses)
    
    valid_avg_iou = mean(total_valid_ious)
    valid_avg_iou_std = stdev(total_valid_ious)
    
    valid_avg_loss = mean(total_valid_losses)
    valid_avg_loss_std = stdev(total_valid_losses)
    
    print(f"\nAverage training iou of all folds:\t{train_avg_iou:.4f} +/- {train_avg_iou_std:.4f}")    
    print(f"Average training loss of all folds:\t{train_avg_loss:.4f} +/- {train_avg_loss_std:.4f}")
    
    print(f"Average validation iou of all folds:\t{valid_avg_iou:.4f} +/- {valid_avg_iou_std:.4f}")
    print(f"Average validation loss of all folds:\t{valid_avg_loss:.4f} +/- {valid_avg_loss_std:.4f}")
    
    # Log the metrics if using wanb
    if config.LOG:
        avg_metrics = {}
        
        avg_metrics["train_avg_iou"] = train_avg_iou
        avg_metrics["train_avg_iou_std"] = train_avg_iou_std
        avg_metrics["train_avg_loss"] = train_avg_loss
        avg_metrics["train_avg_loss_std"] = train_avg_loss_std
        
        avg_metrics["valid_avg_iou"] = valid_avg_iou
        avg_metrics["valid_avg_iou_std"] = valid_avg_iou_std
        avg_metrics["valid_avg_loss"] = valid_avg_loss
        avg_metrics["valid_avg_loss_std"] = valid_avg_loss_std
        
        wandb.log({"avg_metrics": avg_metrics})
      
        
def setup(config=None):
    model_name = config.MODEL_NAME
    print("\nLoading training data...")
    df_train = pd.read_csv(config.TRAIN_CSV)
    print("Loading training data complete.\n")
    print(df_train.info())

    print("\nConfiguring data...")
    ds_train = CellDataset(df_train, config=config)
    print("Configuring data complete.\n")

    print(f"Creating model {model_name}...")
    model = make_model(model_name=model_name, config=config)
    print(f"Creating model {model_name} complete.\n")

    print("Configuring hyperparameters...")
    params = config.configure_parameters(model=model)
    print("Configuring hyperparameters complete.\n")

    return ds_train, params   
        
        
def train(config=None):
    ds_train, params = setup(config=config)
    start = datetime.now()

    print(f"\nConfiguration setup complete. Training began at {start} ...\n")
    time.sleep(2)
    if config.LOG:
        conf = dotenv_values("config/.env")
        os.environ['WANDB_API_KEY'] = conf["wandb_api_key"]
        github_sha = os.getenv('GITHUB_SHA')
        config.GITHUB_SHA = github_sha[:5] if github_sha else None
        
        run = wandb.init(project=conf["project"], 
                        entity=conf["entity"], 
                        config=config,
                        reinit=True)
        
        with run:
            _train(model_name=config.MODEL_NAME, 
                config=config, 
                dataset=ds_train, 
                log=config.LOG,
                run=run,
                checkpoint=config.CHECKPOINT,
                kwargs=params)
    # local implementation
    else:
        _train(model_name=config.MODEL_NAME, 
            config=config, 
            dataset=ds_train, 
            log=config.LOG,
            checkpoint=config.CHECKPOINT,
            kwargs=params)
      
    end = datetime.now()
    train_time = end - start
    print(f"\nTraining complete. Total training time {train_time}.")


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
        self.__name__ = "mixed_loss"

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()
    