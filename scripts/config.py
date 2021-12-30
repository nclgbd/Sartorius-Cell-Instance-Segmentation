import os
import random
from statistics import mode
import numpy as np
import torch
import yaml
import segmentation_models_pytorch as smp

from pprint import pprint
from torch import nn, optim
from tqdm import tqdm

from Losses import MixedLoss
from Utilities import make_model

AVAIL_PARAMS = {
    "iou": smp.utils.metrics.IoU,
    "dice_loss": smp.utils.losses.DiceLoss,
    "mixed_loss": MixedLoss,
    "adam": optim.Adam,
}


def configure_params(config, model_cfg):

    avail_params = AVAIL_PARAMS
    model = make_model(config)
    model_params = model.parameters()

    # config.model_params = model_params
    # config.model = model
    params = {
        "optimizer": None,
        "loss": None,
        "metrics": [],
    }

    for key, values in list(model_cfg.items()):
        if type(values) == dict:
            for n, kwargs in values.items():
                if key == "optimizer":
                    params[key] = avail_params[n](params=model_params, **kwargs)
                elif key == "metrics":
                    params[key].append(avail_params[n](**kwargs))
                else:
                    params[key] = avail_params[n](**kwargs)

    return model, params


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

            # Models
            self.unet = self.defaults_cfg["unet"]
            self.unetplusplus = self.defaults_cfg["unetplusplus"]
            self.model: nn.Module

            self.github_sha = ""

        with open(self.params_path, "r") as stream:
            self.model_cfg = yaml.safe_load(stream)
            print("\nParameters path:", {self.params_path})
            pprint(self.model_cfg)

            # self.model_name = self.model_cfg["model_name"]
            self.lr = self.model_cfg["lr"]
            self.batch_size = self.model_cfg["batch_size"]
            self.metrics = self.model_cfg["metrics"]
            self.loss = self.model_cfg["loss"]
            self.optimizer = self.model_cfg["optimizer"]
            self.model_params = None

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
