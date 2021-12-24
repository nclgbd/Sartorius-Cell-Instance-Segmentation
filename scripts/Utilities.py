"""https://www.kaggle.com/ishandutta/sartorius-complete-unet-understanding/notebook"""

import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
import segmentation_models_pytorch as smp

from pprint import pprint
from torch import nn

from albumentations import (HorizontalFlip, VerticalFlip, 
                            ShiftScaleRotate, Normalize, Resize, 
                            Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import (Dataset, 
                              Subset, 
                              DataLoader)
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

CLASS_MAPPING = {
    0: 'shsy5y',
    1: 'cort', 
    2: 'astro'
}

CLASS_MAPPING_ID = {v:k for k, v in CLASS_MAPPING.items()}


def make_model(model_name="unet", config=None):
    model: nn.Module
    if model_name == "unet":
        model = smp.Unet(config.BACKBONE, encoder_weights="imagenet", activation=None)
    elif model_name == "unetplusplus":
        model = smp.UnetPlusPlus(config.BACKBONE, encoder_weights="imagenet", activation=None)
    
    return model
    

def create_loader(dataset: Dataset, idx, config=None, shuffle=False):
    loader = DataLoader(torch.utils.data.Subset(dataset, idx),
                        batch_size=config.BATCH_SIZE, 
                        num_workers=4, 
                        pin_memory=True, 
                        shuffle=shuffle)
    
    return loader
  
    
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
            continue;
        
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
    image = image * np.array(std) + np.array(mean) # [0, 1] -> [0, 255]
    image = image.clip(0, 1)
    return image


def _rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
        
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
    cmap = {"shsy5y": (0,0,255),
            "astro": (0,255,0),
            "cort": (255,0,0)}

    if colors:
        mask = np.zeros((520, 704, 3))
        for label,cell_type in zip(labels,cell_type):
            c = cmap[cell_type]
            mask += _rle_decode(label, shape=(520, 704, 3), color=c)
    else:
        mask = np.zeros((520, 704, 1))
        for label in labels:
            mask += _rle_decode(label, shape=(520, 704, 1))
    mask = mask.clip(0, 1)

    img = cv2.imread(os.path.join(config.TRAIN_PATH, f"{image_id}.png"))
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
    
    plt.show();


# Note the use of wandb.Image
def wandb_mask(bg_img, gt_mask):
  return wandb.Image(bg_img, masks={
      "ground_truth" : {
          "mask_data" : gt_mask,
          "class_labels": CLASS_MAPPING
      }
    }
  )


class CellDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, config=None):
        self.config = config
        self.df = df
        self.base_path = config.TRAIN_PATH
        
        self.transforms = Compose([Resize(config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]), 
                                   Normalize(mean=config.MEAN, std=config.STD, p=1), 
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5),
                                   ToTensorV2()])
        
        self.gb = self.df.groupby('id')
        self.image_ids = list(df.id.unique())
        self.folds = list(self._split_data(n_splits=config.N_SPLITS))

        self.dl_train: DataLoader
        self.dl_valid: DataLoader


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        img_path = os.path.join(self.base_path, image_id+".png")
        # img_path = df["img_path"].loc[image_id]
        image = cv2.imread(img_path)
        mask = build_masks(self.df, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask.reshape((1, self.config.IMAGE_RESIZE[0], self.config.IMAGE_RESIZE[1]))


    def __len__(self):
        return len(self.image_ids)
    
    
    def _split_data(self, n_splits=5):
        # creates folds
        n_splits = self.config.N_SPLITS if self.config else n_splits
        X = []
        y = []
        
        for image_id in tqdm(self.image_ids): 
            X.append(os.path.join(self.base_path, image_id+".png"))
            y.append(build_masks(self.df, image_id, input_shape=(520, 704)))
        
        X = np.array(X)   
        y = np.array(y)  
        yn, nh, nw = y.shape
        y = y.reshape((yn, nh * nw))
        # mask = (mask >= 1).astype('float32')
        
        if self.config.KFOLD:
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True).split(X, y.argmax(1))
            return list(folds)
        
        else:
            test_size = 1.0/n_splits
            return list(train_test_split(X, y.argmax(1), 
                                    test_size=test_size, 
                                    random_state=self.config.SEED))
            
       
class EarlyStopping():
    
    def __init__(self, 
                 model_dir:str, 
                 model_name:str, 
                 min_delta=0,
                 patience=5,
                 config=None,
                 run=None):
        """
        Class for early stopping, because only plebs rely on set amounts of epochs.
        
        Attributes
        ----------
        `TODO`
        Parameters
        ----------
        `model_name` : `str`\n
            Model name.
        `fold` : `int`\n
            Number representing the current fold.
        `min_delta` : `int`, `optional`\n
            Smallest number the given metric needs to change in order to count as progress, by default 0.
        """        
        
        self.min_loss = float('inf')
        self.max_iou = -float('inf')
        self.min_delta = min_delta
        self.model_name = model_name 
        self.patience = patience
        self.count = 0
        self.first_run = True
        self.best_model = None
        self.artifact: wandb.Artifact
        self.run = run
        self.id = run.id if not config else config.GITHUB_SHA[:5]
        self.fname = "".join([self.model_name, f"-{self.id}", '.pth']) if run else self.model_name+'.pth'
            
        self.path = str(os.path.join(model_dir, self.fname))
        
        
    def checkpoint(self, model:nn.Module, epoch:int, loss:float, iou:float, optimizer, log=False, checkpoint=True):
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
        `optimizer`
            The optimization function currently in use.
        `log` : `bool`, `optional`\n
            Boolean representing whether we're saving logging information to wandb, by default True.
        Returns
        -------
        `int`\n
            Returns a number representing the current level of patience reached.
        """        
        
        print(f'Loss to beat: {(self.min_loss - self.min_delta):.4f}')
        if (self.min_loss - self.min_delta) > loss or self.first_run:                
            self.count = 0
            self.first_run = False
            self.min_loss = loss
            self.max_iou = iou
            self.best_model = model
            self.state_dict = {'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'loss': self.min_loss,
                               'iou': self.max_iou}
            if checkpoint:
                torch.save(self.state_dict, self.path)
                
                if log:
                    self.artifact = wandb.Artifact('unet', type='model')
                    self.artifact.add_file(self.path)
                    self.run.log_artifact(self.artifact)

        else:
            self.count += 1
            
        return self.check_patience()


    def check_patience(self):
        print(f"Patience: {self.count}/{self.patience}" )
        return self.count >= self.patience