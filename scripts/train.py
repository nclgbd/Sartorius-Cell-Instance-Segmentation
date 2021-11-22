import torch
import wandb

from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def _get_kwargs(**kwargs):
    criterion = kwargs["criterion"]
    optimizer = kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]
    
    return (criterion, 
            optimizer, 
            scheduler)

def _train(model: nn.Module, config, dl_train: DataLoader, log=False, **kwargs):
    criterion, optimizer, scheduler = _get_kwargs(kwargs)
    
    n_batches = len(dl_train)
    for epoch in range(1, config.EPOCHS + 1):
        print(f"Epoch: {epoch}")
        running_loss = 0.0
        optimizer.zero_grad()
        
        loss = running_acc = 0
        for batch_idx, batch in enumerate(dl_train):
            
            # Predict
            images, masks = batch
            images, masks = images.cuda(),  masks.cuda()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            if log:
                wandb.log({"loss": loss})
                wandb.watch(model)
            
            # Back prop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == masks).sum().item()
        
            if scheduler:
                scheduler.step(running_loss)

        epoch_loss = running_loss / n_batches
        epoch_acc = running_acc / n_batches
        print(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f} - Train Accuracy {epoch_acc:.4f}")

def train(model: nn.Module, config, dl_train: DataLoader, log=False, **kwargs):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    model.cuda()
    model.train()

    criterion = kwargs["criterion"]
    optimizer = kwargs["optimizer"]
    scheduler = None if "scheduler" not in kwargs.keys() else kwargs["scheduler"]

    if log:
        with wandb.init(project=config["project"], 
                        entity=config["entity"], 
                        reinit=True):
            wandb.config = config
            _train(model, config, dl_train, criterion, optimizer, scheduler)
        
    else:
        _train(model, config, dl_train, criterion, optimizer, scheduler)
        
    # run = wandb.init(project='sartorius-unet', config=WANDB_CONFIG)



