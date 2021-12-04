import os
import torch
import wandb

from torch import nn
from torch.utils.data import (DataLoader, 
                              Dataset)
from tqdm import tqdm
from statistics import (mean, 
                        stdev)
from utils import (create_loader,
                   make_model)

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
        