import os
import random
import numpy as np
import torch
import yaml

class Config:
    def __init__(self, model_name="resnet34"):
        """
        Configuration  class for all models

        Parameters
        ----------
        `model_name` : `str`\n
            Name of the model
        """        
        
        with open("config/params.yaml", "r") as stream:
            base_cfg = yaml.safe_load(stream)
            cfg = base_cfg[model_name]
        
        self.DIRECTORY_PATH = base_cfg["project_path"]
        self.TRAIN_CSV = os.path.join(self.DIRECTORY_PATH, "train.csv")
        self.TRAIN_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["train"])
        self.TEST_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["test"])
        self.TRAIN_SEMI_SUPERVISED_PATH = os.path.join(self.DIRECTORY_PATH, base_cfg["train_semi_supervised"])
        self.SEED = base_cfg["seed"]
        
        self.MEAN = cfg["mean"]
        self.STD = cfg["std"]
        self.IMAGE_RESIZE = cfg["input_size"]
        self.LEARNING_RATE = cfg["eta"]
        self.BATCH_SIZE = cfg["batch_size"]
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
    
