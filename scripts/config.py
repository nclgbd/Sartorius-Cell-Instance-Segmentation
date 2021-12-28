import os
import random
import numpy as np
import torch
import yaml
import segmentation_models_pytorch as smp

from pprint import pprint
from torch import nn, optim
from tqdm import tqdm

from Losses import MixedLoss


def configure_params(config, model: nn.Module):

    model_name = config.model_name
    params = {"model_name": model_name}

    for k in tqdm(list(config.keys())):
        k_params = config[k]
        keys = list(k_params.keys())

        if "dice_loss" in keys:
            params["dice_loss"] = smp.utils.losses.DiceLoss(**k_params["dice_loss"])

        if "mixed_loss" in keys:
            params["mixed_loss"] = MixedLoss(**k_params["mixed_loss"])

        if "adam" in keys:
            params["adam"] = optim.Adam(params=model.parameters(), **k_params["adam"])

        if "iou" in keys:
            params["iou"].append(smp.utils.metrics.IoU(**k_params["iou"]))

    # self.params = params
    return params

class Config:
    def __init__(
        self,
        defaults_path="config/defaults.yaml",
    ):
        """
        Configuration  class for all models

        Parameters
        ----------
        `defaults_path` : `str`\n
            Path to the default configuration
        """
        with open(defaults_path, "r") as stream:
            self.defaults_cfg = yaml.safe_load(stream)
            self.defaults_path = defaults_path
            print("\nDefaults path:", {self.defaults_path})
            pprint(self.defaults_cfg)

            self.model_name = self.defaults_cfg["model_name"]
            self.config_path = self.defaults_cfg["config_path"]
            self.directory_path = self.defaults_cfg["directory_path"]

            self.train_path = os.path.join(
                self.directory_path, self.defaults_cfg["train_path"]
            )
            self.test_path = os.path.join(
                self.directory_path, self.defaults_cfg["test_path"]
            )

            self.train_semi_supervised_path = os.path.join(
                self.directory_path, self.defaults_cfg["train_semi_supervised_path"]
            )
            self.sweep_path = os.path.join(
                self.config_path, self.defaults_cfg["sweep_path"]
            )
            self.params_path = os.path.join(
                self.config_path, self.defaults_cfg["params_path"]
            )

            self.train_csv = os.path.join(
                self.directory_path, self.defaults_cfg["train_csv"]
            )

            self.sweep = self.defaults_cfg["sweep"]
            self.log = self.defaults_cfg["log"]
            self.checkpoint = self.defaults_cfg["checkpoint"]

            self.seed = self.defaults_cfg["seed"]
            self.model_path = self.defaults_cfg["model_path"]
            self.kfold = self.defaults_cfg["kfold"]
            self.n_splits = self.defaults_cfg["n_splits"]
            self.count = self.defaults_cfg["count"]
            self.epochs = self.defaults_cfg["epochs"]
            self.mean = self.defaults_cfg["mean"]
            self.std = self.defaults_cfg["std"]
            self.image_resize = self.defaults_cfg["image_resize"]

            self.github_sha = ""

        with open(self.params_path, "r") as stream:
            self.model_cfg = yaml.safe_load(stream)
            print("\nParameters path:", {self.params_path})
            pprint(self.model_cfg)

            self.model_name = self.model_cfg["model_name"]
            self.batch_size = self.model_cfg["batch_size"]

        with open(self.sweep_path, "r") as stream:
            self.sweep_cfg = yaml.safe_load(stream)
            print("\nSweep configuration path:", {self.sweep_path})
            pprint(self.sweep_cfg)

        self.set_seed()
        print("")

    def set_seed(self):
        """
        Sets the random seed for the environment
        """

        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
