import os
import random
import numpy as np
import torch
import yaml

from pprint import pprint
from torch import nn

from Utilities import make_model, create_criterion, create_optimizer, create_metrics


def configure_params(config):
    # tunable_params = config.tunable_params
    model = make_model(config)
    model_params = model.parameters()

    # grab parameter names
    loss_type = list(config.loss.keys())[0]
    loss = config.loss[loss_type]

    opt_type = list(config.optimizer.keys())[0]
    optimizer = config.optimizer[list(config.optimizer.keys())[0]]

    metrics_type = list(config.metrics.keys())[0]
    metrics = config.metrics[list(config.metrics.keys())[0]]

    criterion = create_criterion(loss_type, **loss)
    optimizer = create_optimizer(opt_type, model_params, **optimizer)
    metrics = create_metrics(metrics_type, **metrics)

    return model, {
        "criterion": criterion,
        "optimizer": optimizer,
        "metrics": metrics,
    }


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
            self.mode = self.defaults_cfg["mode"]
            self.kfold = self.defaults_cfg["kfold"]

            self.n_splits = self.defaults_cfg["n_splits"]
            self.count = self.defaults_cfg["count"]
            self.epochs = self.defaults_cfg["epochs"]
            self.mean = self.defaults_cfg["mean"]
            self.std = self.defaults_cfg["std"]
            self.image_resize = self.defaults_cfg["image_resize"]
            self.batch_size = self.defaults_cfg["batch_size"]

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
            # self.lr = self.model_cfg["lr"]
            self.metrics = self.model_cfg["metrics"]
            self.loss = self.model_cfg["loss"]
            self.optimizer = self.model_cfg["optimizer"]
            self.defaults_cfg.update(self.model_cfg)
            self.model_params = None

            self.tunable_params = [
                self.batch_size,
                self.metrics,
                self.loss,
                self.optimizer,
            ]

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
