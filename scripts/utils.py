"""https://www.kaggle.com/ishandutta/sartorius-complete-unet-understanding/notebook"""

import cv2
import numpy as np
import os

from albumentations import (HorizontalFlip, VerticalFlip, 
                            ShiftScaleRotate, Normalize, Resize, 
                            Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.config import Config

config = Config()


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


class CellDataset(Dataset):
    
    def __init__(self, df):
        self.df = df
        self.base_path = config.TRAIN_PATH
        
        self.transforms = Compose([Resize(config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]), 
                                   Normalize(mean=config.MEAN, std=config.STD, p=1), 
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5),
                                   ToTensorV2()])
        
        self.gb = self.df.groupby('id')
        self.image_ids = list(df.id.unique())


    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        image_path = os.path.join(self.base_path, image_id+".png")
        image = cv2.imread(image_path)
        mask = build_masks(self.df, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask.reshape((1, config.IMAGE_RESIZE[0], config.IMAGE_RESIZE[1]))


    def __len__(self):
        return len(self.image_ids)
       