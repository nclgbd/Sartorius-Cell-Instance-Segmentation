import os
import random
import numpy as np
import torch
import yaml

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
            base_cfg = yaml.safe_load(stream)
            self.cfg = base_cfg[model_name][backbone]
        
        self.BACKBONE = backbone
        self.DIRECTORY_PATH = base_cfg["project_path"]
        self.TRAIN_CSV = os.path.join(self.DIRECTORY_PATH, "train.csv")
        self.TRAIN_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["train"])
        self.TEST_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["test"])
        self.TRAIN_SEMI_SUPERVISED_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["train_semi_supervised"])
        self.SEED = base_cfg["seed"]
        
        self.MEAN = self.cfg["mean"]
        self.STD = self.cfg["std"]
        self.IMAGE_RESIZE = self.cfg["input_size"]
        self.LEARNING_RATE = self.cfg["eta"]
        self.BATCH_SIZE = self.cfg["batch_size"]
        self.EPOCHS = 1000
        
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
    
