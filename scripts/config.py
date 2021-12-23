import os
import random
import numpy as np
import torch
import yaml
import segmentation_models_pytorch as smp

from torch import optim
from torch import nn
from pprint import pprint
from Training import MixedLoss

class Config:
    def __init__(self, model_name, backbone="resnet34", config_path="config/params.yaml"):
        """
        Configuration  class for all models

        Parameters
        ----------
        `backbone` : `str`\n
            Name of the model
        """        
        
        with open(config_path, "r") as stream:
            self.base_cfg = yaml.safe_load(stream)
            self.model_cfg = self.base_cfg[model_name]
            self.cfg = self.model_cfg[backbone]
        
        self.BACKBONE = backbone
        self.DIRECTORY_PATH = self.base_cfg["project_path"]
        self.TRAIN_CSV = os.path.join(self.DIRECTORY_PATH, "train.csv")
        self.TRAIN_PATH = os.path.join(self.DIRECTORY_PATH, self.base_cfg["train"])
        self.TEST_PATH = os.path.join(self.DIRECTORY_PATH, self.base_cfg["test"])
        self.TRAIN_SEMI_SUPERVISED_PATH = os.path.join(self.DIRECTORY_PATH, self.base_cfg["train_semi_supervised"])
        
        self.SEED = self.base_cfg["seed"]
        self.MODEL_PATH = self.base_cfg["model_path"]
        self.KFOLD = self.base_cfg["kfold"]
        self.N_SPLITS = self.base_cfg["n_splits"]
        
        self.MEAN = self.cfg["mean"]
        self.STD = self.cfg["std"]
        self.IMAGE_RESIZE = self.cfg["input_size"]
        self.BATCH_SIZE = self.cfg["batch_size"]
        self.EPOCHS = self.cfg["epochs"]
        
        self.set_seed()
    
    
    def set_seed(self):
        """
        Sets the random seed for the environment
        """
        
        random.seed(self.SEED)
        os.environ["PYTHONHASHSEED"] = str(self.SEED)
        np.random.seed(self.SEED)
        
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def configure_hyperparameters(self, keys: set, model: nn.Module): 
        hyperparams = {"loss": None,
                       "optimizer": None,
                       "metrics": None}
        
        for k in self.model_cfg.keys():
            k_params = self.model_cfg[k]
            if "loss" == k:
                if "dice_loss" in keys:
                    hyperparams["loss"] = smp.utils.losses.DiceLoss(**k_params["dice_loss"])
                elif "mixed_loss" in keys:
                    hyperparams["loss"] = MixedLoss(**k_params["dice_loss"])
                    
            if "optimizer" == k:
                if "adam" in keys:
                    hyperparams["optimizer"] = optim.Adam(params=model.parameters(), **k_params["adam"])
                    
            if "metrics" == k:
                if "iou" in keys:
                    hyperparams["metrics"] = [smp.utils.metrics.IoU(**k_params["iou"])]
                    

        pprint(hyperparams)
        return hyperparams