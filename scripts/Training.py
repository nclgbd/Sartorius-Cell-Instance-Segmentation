import os
import torch
import wandb

from dotenv import dotenv_values
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import (DataLoader, 
                              Dataset)
from torch.nn import functional as F
from tqdm import tqdm
from statistics import (mean, 
                        stdev)
from utils import (EarlyStopping, create_loader,
                   make_model)


def _get_kwargs(**kwargs):
    criterion = None if "criterion" not in kwargs.keys() else kwargs["criterion"]
    optimizer = None if "optimizer" not in kwargs.keys() else kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]
    
    return (criterion, 
            optimizer, 
            scheduler)


def _loop_fn(model: nn.Module, loader: DataLoader, log=False, mode="train", **kwargs):
    kwargs = kwargs['kwargs']['kwargs']['kwargs']
    criterion = kwargs["criterion"]
    optimizer = kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]
    
    if mode == "train":
        model.train()
    
    elif mode == "test":
        model.eval()
    
    n_batches = len(loader)
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
            
        return epoch_loss, epoch_iou, optimizer


def _init_train(model_name, config=None, log=False, run=None):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    model = make_model(model_name, config=config)
    model.cuda()
    early_stopping: EarlyStopping
    
    if log:
        early_stopping = EarlyStopping(model_dir=config.MODEL_PATH,
                                       model_name=model_name,
                                       run=run)
    
    return model, early_stopping


def _train(model_name, dataset: Dataset, config=None, log=False, run=None, **kwargs):
    model, early_stopping = _init_train(model_name,
                                       config=config,
                                       log=log,
                                       run=run)
    
    total_train_ious = total_train_losses = []
    total_valid_ious = total_valid_losses = []
    # print(type(dataset.folds))
    # print(len(dataset.folds))
    # print(dataset.folds)
    criterion = None if "criterion" not in kwargs.keys() else kwargs["criterion"]
    optimizer = None if "optimizer" not in kwargs.keys() else kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]
    pprint(kwargs)
    
    if log:
        wandb.watch(model, criterion, log_graph=True)
        
    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        # Create loaders
        print(f"Fold {idx+1}\n")
        dataset.dl_train = create_loader(dataset, train_idx, config=config)
        dataset.dl_valid = create_loader(dataset, valid_idx, config=config)
        es_counter = 0
        for epoch in range(1, config.EPOCHS + 1):
            print(f"Epoch: {epoch}")
            # Train and get metrics back
            train_epoch_loss, train_epoch_iou, optimizer = _loop_fn(model,
                                                        loader=dataset.dl_train, 
                                                        log=log,
                                                        mode="train",
                                                        kwargs=kwargs)
            
            valid_epoch_loss, valid_epoch_iou, _ = _loop_fn(model,
                                                        loader=dataset.dl_valid,
                                                        log=log,
                                                        mode="test",
                                                        kwargs=kwargs)
            
            # Print epoch results
            print(f"Epoch {idx+1} |\tTrain loss {train_epoch_loss:.4f} -- Train IoU (Intersection over Union) {train_epoch_iou:.4f}")
            print(f"Epoch {idx+1} |\tValidation loss {valid_epoch_loss:.4f} -- Validation IoU (Intersection over Union) {valid_epoch_iou:.4f}")
            
            total_train_losses.append(train_epoch_loss)
            total_train_ious.append(train_epoch_iou)
            
            total_valid_losses.append(valid_epoch_loss)
            total_valid_ious.append(valid_epoch_iou)
            
            if log:
                es_counter = early_stopping.checkpoint(model,
                                                       epoch=epoch,
                                                       loss=valid_epoch_loss,
                                                       iou=valid_epoch_iou,
                                                       optimizer=optimizer)
                
                print(f"Patience currently: {es_counter}")
                if es_counter >= early_stopping.patience:
                    es_counter = 0
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


def _skf_train(model_name, dataset: Dataset, config=None, log=False, run=None, **kwargs):
    model, early_stopping = _init_train(model_name,
                                       config=config,
                                       log=log)
    
    total_train_ious = total_train_losses = []
    total_valid_ious = total_valid_losses = []

    for idx, (train_idx, valid_idx) in enumerate(dataset.folds):
        # Create loaders
        print(f"Fold {idx+1}\n")
        dl_train = create_loader(dataset, train_idx, config=config)
        dl_valid = create_loader(dataset, valid_idx, config=config)
        
        for epoch in range(1, config.EPOCHS + 1):
            print(f"Epoch: {epoch}")
            # Train and get metrics back
            train_epoch_loss, train_epoch_iou, optimizer = _loop_fn(model,
                                                        loader=dl_train, 
                                                        log=log,
                                                        mode="train",
                                                        kwargs=kwargs)
            
            valid_epoch_loss, valid_epoch_iou, _ = _loop_fn(model,
                                                        loader=dl_valid,
                                                        log=log,
                                                        mode="test",
                                                        kwargs=kwargs)
            
            # Print epoch results
            print(f"Epoch {idx+1} |\tTrain loss {train_epoch_loss:.4f} -- Train IoU (Intersection over Union) {train_epoch_iou:.4f}")
            print(f"Epoch {idx+1} |\tValidation loss {valid_epoch_loss:.4f} -- Validation IoU (Intersection over Union) {valid_epoch_iou:.4f}")
            
            total_train_losses.append(train_epoch_loss)
            total_train_ious.append(train_epoch_iou)
            
            total_valid_losses.append(valid_epoch_loss)
            total_valid_ious.append(valid_epoch_iou)
            
            if log:
                es_counter = early_stopping.checkpoint(model,
                                                       epoch=epoch,
                                                       loss=valid_epoch_loss,
                                                       iou=valid_epoch_iou,
                                                       optimizer=optimizer)
                
                print(f"Patience currently: {es_counter}")
                if es_counter >= early_stopping.patience:
                    es_counter = 0
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
    pass
    

def train(model_name, dataset: Dataset, config=None, log=False, **kwargs):
    # wandb implementation
    if log:
        conf = dotenv_values("config/.env")
        os.environ['WANDB_API_KEY'] = conf["wandb_api_key"]
        pprint(config)
        
        run = wandb.init(project=conf["project"], 
                        entity=conf["entity"], 
                        config=config,
                        reinit=True)
        
        with run:
            config = wandb.config
            _train(model_name=model_name, 
                config=config, 
                dataset=dataset, 
                log=log, 
                run=run,
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
    

