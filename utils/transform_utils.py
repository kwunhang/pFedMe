import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def ISIC_raw_train_transforms():
    transforms = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.RandomCrop(height=244, width=244),
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.1),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.1),
        # group 1
        A.OneOf([
            A.Compose([A.MotionBlur(blur_limit=9,p=0.1),],p=1.0),
            A.Compose([A.MedianBlur(blur_limit=3, p=0.005),],p=1.0),
            A.Compose([A.Blur(blur_limit=3, p =0.005),],p=1.0),
        ], p=0.05),
        # group 2
        A.OneOf([
            A.Compose([A.OpticalDistortion(distort_limit=0.5,shift_limit=0.5, p=0.03),],p=1.0),
            A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),],p=1.0),
            A.Compose([A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=0.03)],p=1.0),
        ], p=0.2),
        # group 3
        A.OneOf([
            A.Compose([A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),],p=1.0),
            A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),],p=1.0),
            A.Compose([A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5),],p=1.0),
            A.Compose([A.RandomBrightnessContrast(
            brightness_limit=0.1, 
            contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),],p=1.0),
        ], p=0.05),

        A.PadIfNeeded(min_height=224, min_width=224),
        
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(p=1.0),
    ], p=1.0)
    return transforms

def ISIC_raw_valid_transforms():
    transforms = A.Compose([
        A.LongestMaxSize(max_size=256),
        A.CenterCrop(height=244, width=244),

        A.PadIfNeeded(min_height=224, min_width=224),
        
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(p=1.0),
    ], p=1.0)
    return transforms