"""https://www.kaggle.com/ishandutta/sartorius-complete-unet-understanding/notebook"""

import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
import segmentation_models_pytorch as smp

from dotenv import dotenv_values
from pprint import pprint
from torch import nn, optim
from segmentation_models_pytorch.utils import losses
from segmentation_models_pytorch.utils.metrics import IoU

from Losses import MixedLoss

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    Normalize,
    Resize,
    Compose,
    GaussNoise,
)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split


CLASS_MAPPING = {0: "shsy5y", 1: "cort", 2: "astro"}
CLASS_MAPPING_ID = {v: k for k, v in CLASS_MAPPING.items()}


def setup_env(mode):
    conf = (
        dotenv_values("config/develop.env")
        if mode == "develop"
        else dotenv_values("config/train.env")
    )

    os.environ["WANDB_API_KEY"] = conf["wandb_api_key"]
    os.environ["WANDB_ENTITY"] = conf["wandb_entity"]
    os.environ["WANDB_RESUME"] = conf["wandb_resume"]
    os.environ["WANDB_MODE"] = conf["wandb_mode"]
    os.environ["WANDB_JOB_TYPE"] = conf["wandb_job_type"]
    os.environ["WANDB_TAGS"] = conf["wandb_tags"]


def wandb_setup(config=None):
    reset_wandb_env()

    run_id = wandb.util.generate_id()
    os.environ["WANDB_RUN_ID"] = run_id

    run_name = "".join(["unet", f"-{run_id}"])
    run = wandb.init(
        project="Sartorius-Kaggle-Competition",
        entity="nclgbd",
        config=config,
        reinit=True,
        name=run_name,
    )

    return wandb.config, run


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, _ in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def create_optimizer(optimizer, model_params, **kwargs):
    if optimizer == "adam":
        return optim.Adam(params=model_params, **kwargs)
    else:
        raise ValueError(f"No corresponding `{optimizer}` type")


def create_criterion(loss, **kwargs):
    if loss == "mixed_loss":
        return MixedLoss(**kwargs)
    elif loss == "dice_loss":
        return losses.DiceLoss(**kwargs)
    else:
        raise ValueError(f"No corresponding `{loss}` type")


def create_metrics(metrics, **kwargs):
    if metrics == "iou":
        return IoU(**kwargs)
    else:
        raise ValueError(f"No corresponding `{metrics}` type")


def make_model(config=None, cuda=True):
    model: nn.Module
    model_name = config.model_name
    if model_name == "unet":
        kwargs = config.unet
        model = smp.Unet(**kwargs)
    elif model_name == "unetplusplus":
        kwargs = config.unetplusplus
        model = smp.UnetPlusPlus(**kwargs)
    else:
        raise ValueError(f"`{model_name}` model not recognized")

    if cuda:
        model.cuda()
    return model


def create_loader(dataset: Dataset, idx, batch_size, shuffle=False):
    ds = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def create_dataset(config=None):
    print("\nLoading training data...")
    # df_train = pd.read_csv(config.train_csv)
    df_train = pd.read_csv("data/train.csv")
    print("Loading training data complete.\n")
    print(df_train.info())

    print("\nConfiguring data...")
    ds_train = CellDataset(df_train, config=config)
    print("Configuring data complete.\n")

    return ds_train


def display_dataset(img_paths, rows=2, cols=10):
    """
    Function to Display Images from Dataset.

    parameters: images_path(string) - Paths of Images to be displayed
                rows(int) - No. of Rows in Output
                cols(int) - No. of Columns in Output
    """
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 10))
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            ax.ravel()[idx].imshow(img)
            ax.ravel()[idx].set_axis_off()
        except:
            continue

    plt.tight_layout()
    plt.show()


def im_convert(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Private function for converting an image so it can be displayed using matplotlib functions properly.
    Parameters
    ----------
    `tensor`\n
        Tensor represention of image data.
    `mean` : `tuple` or `list`, `optional`\n
        Mean of the data; used for de-normalization of the image, by default (0.485, 0.456, 0.406).
    `std` : `tuple` or `list`, `optional`\n
        Standard deviation of the data; used for de-normalaztion of the image, by default (0.229, 0.224, 0.225).
    Returns
    -------
    `ndarray`\n
        Returns ndarry de-normalizazed representation of an image.
    """

    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array(std) + np.array(mean)  # [0, 1] -> [0, 255]
    image = image.clip(0, 1)
    return image


def _rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = color

    return img.reshape(shape)


def build_masks(df_train, image_id, input_shape):

    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))

    for label in labels:
        mask += _rle_decode(label, shape=(height, width))

    mask = mask.clip(0, 1)
    return mask


def get_img_paths(path):
    """
    Function to Combine Directory Path with individual Image Paths

    parameters: path(string) - Path of directory
    returns: image_names(string) - Full Image Path
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in tqdm(filenames):
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names


def plot_masks(image_id, df_train, config=None, colors=True):
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    cell_type = df_train[df_train["id"] == image_id]["cell_type"].tolist()
    cmap = {"shsy5y": (0, 0, 255), "astro": (0, 255, 0), "cort": (255, 0, 0)}

    if colors:
        mask = np.zeros((520, 704, 3))
        for label, cell_type in zip(labels, cell_type):
            c = cmap[cell_type]
            mask += _rle_decode(label, shape=(520, 704, 3), color=c)
    else:
        mask = np.zeros((520, 704, 1))
        for label in labels:
            mask += _rle_decode(label, shape=(520, 704, 1))
    mask = mask.clip(0, 1)

    img = cv2.imread(os.path.join(config.train_path, f"{image_id}.png"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 32))
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.imshow(mask)
    plt.axis("off")

    plt.show()


# Note the use of wandb.Image
def wandb_mask(bg_img, gt_mask):
    return wandb.Image(
        bg_img,
        masks={"ground_truth": {"mask_data": gt_mask, "class_labels": CLASS_MAPPING}},
    )


class CellDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config=None):
        """
        Custom dataset for kaggle competition

        Attributes
        ----------
        `df` : `pd.DataFrame`\n
        `config` : `Config`\n
        `base_path` : `str`\n
        `transforms` : `Compose`\n
        `image_ids` : `list`\n
        `folds` : `list`\n
        `dl_train` : `Dataloader`\n
        `dl_valid` : `Dataloader`\n

        Parameters
        ----------
        `df` : `pd.DataFrame`\n
            Dataframe representation of the dataset
        `config` : `Config`, `optional`\n
            The configuration instance, by default `None`
        """
        self.config = config
        self.df = df
        self.train_path = config.train_path

        self.transforms = Compose(
            [
                Resize(config.image_resize[0], config.image_resize[1]),
                Normalize(mean=config.mean, std=config.std, p=1),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                # GaussNoise(mean=config.mean),
                # ShiftScaleRotate(),
                ToTensorV2(),
            ]
        )

        self.gb = self.df.groupby("id")
        self.image_ids = list(df.id.unique())
        self.folds = list(self._split_data(n_splits=config.n_splits))

        self.dl_train: DataLoader
        self.dl_valid: DataLoader

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df["annotation"].tolist()
        img_path = os.path.join(self.train_path, image_id + ".png")
        # img_path = df["img_path"].loc[image_id]
        image = cv2.imread(img_path)
        mask = build_masks(self.df, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype("float32")
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask.reshape(
            (1, self.config.image_resize[0], self.config.image_resize[1])
        )

    def __len__(self):
        return len(self.image_ids)

    def _split_data(self, n_splits=5):
        # creates folds
        n_splits = self.config.n_splits if self.config else n_splits
        X = []
        y = []

        for image_id in tqdm(self.image_ids):
            X.append(os.path.join(self.train_path, image_id + ".png"))
            y.append(build_masks(self.df, image_id, input_shape=(520, 704)))

        X = np.array(X)
        y = np.array(y)
        yn, nh, nw = y.shape
        y = y.reshape((yn, nh * nw))
        # mask = (mask >= 1).astype('float32')

        if self.config.kfold:
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True).split(
                X, y.argmax(1)
            )
            return list(folds)

        else:
            test_size = 1.0 / n_splits
            return list(
                train_test_split(
                    X, y.argmax(1), test_size=test_size, random_state=self.config.seed
                )
            )


class EarlyStopping:
    def __init__(
        self,
        model_dir: str,
        model_name: str,
        min_delta=0,
        patience=5,
        config=None,
        run=None,
    ):
        """
        Class for early stopping, because only plebs rely on set amounts of epochs.

        Attributes
        ----------
        `min_loss` : `float`\n
            The current minimized loss
        `max_iou` : `float`\n
            The current maximized intersection over union (iou)
        `min_delta` : `float`\n
            The minimum change expected to trigger a checkpoint
        `model_name` : `str`\n
            The name of the model
        `patience` : `int`\n
            The highest successive training epoch with no improvement of the model
        `count` : `int`\n
            The current patience level
        `first_run` : `bool`\n
            Whether this is the first run or not
        `best_model` : `nn.Module`\n
            The current best model
        `artifact` : `wandb.Artifact`\n
            The artifact associated with the current run
        `run` : `wandb.Run`\n
            The current training session
        `config` : `Config`\n
            The configuration instance
        `id` : `str`\n
            The run id
        `fname` : `str`\n
            The weights filename
        `path` : `str`\n
            The string representation of the path

        Parameters
        ----------
        `model_name` : `str`\n
            Model name.
        `fold` : `int`\n
            Number representing the current fold.
        `min_delta` : `int`, `optional`\n
            Smallest number the given metric needs to change in order to count as progress, by default 0.
        """

        self.min_loss = float("inf")
        self.max_iou = -float("inf")
        self.min_delta = min_delta
        self.model_name = model_name
        self.patience = patience
        self.count = 0
        self.first_run = True
        self.best_model = None
        self.artifact: wandb.Artifact
        self.run = run
        self.config = config
        self.id = run.id if run else ""
        self.run_name = f"{self.model_name}-{self.id}" if run else self.model_name
        self.fname = (
            "".join([self.run_name, ".pth"]) if run else self.model_name + ".pth"
        )

        self.path = str(os.path.join(model_dir, self.fname))

    def checkpoint(
        self, model: nn.Module, epoch: int, loss: float, iou: float, optimizer
    ) -> (bool):
        """
        Creates the checkpoint and keeps track of when we should stop training.

        Parameters
        ----------
        `model` : `nn.Module`\n
            The model to be saved.
        `epoch` : `int`\n
            Number representing the current epoch.
        `loss` : `float`\n
            Current loss.
        `iou` : `float`\n
            Current IoU.
        Returns
        -------
        `bool`\n
            Returns a number representing the current level of patience reached.
        """

        print(f"Loss to beat: {(self.min_loss - self.min_delta):.4f}")
        if (self.min_loss - self.min_delta) > loss or self.first_run:
            self.count = 0
            self.first_run = False
            self.min_loss = loss
            self.max_iou = iou
            self.best_model = model
            self.state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": self.min_loss,
                "iou": self.max_iou,
            }

        else:
            self.count += 1

        return self._check_patience()

    def _check_patience(self) -> (bool):
        print(f"Patience: {self.count}/{self.patience}")
        return self.count >= self.patience

    def save_model(self):
        if self.config.checkpoint:
            torch.save(self.state_dict, self.path)
            if self.config.log:
                self.artifact = wandb.Artifact(self.run_name, type="model")
                self.artifact.add_file(self.path)
                self.run.log_artifact(self.artifact)
