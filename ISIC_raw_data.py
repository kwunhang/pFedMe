import matplotlib.pyplot as plt
from PIL import Image
from utils.transform_utils import ISIC_raw_train_transforms, ISIC_raw_valid_transforms
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

# Path to your image
image_name = 'ISIC_0073247'
image_path = 'data/ISIC_2019_Training_Input/{}.jpg'.format(image_name)  # replace with your image path

# Open the image file
image = Image.open(image_path)

# Initialize your transforms
train_transforms = ISIC_raw_train_transforms()
valid_transforms = ISIC_raw_valid_transforms()

fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# Apply the transforms to the same image multiple times
for i in range(3):
    # Apply the transforms and convert to tensor
    train_image = train_transforms(image=np.array(image))['image']
    valid_image = valid_transforms(image=np.array(image))['image']
    
    # Convert tensor to image for plotting
    train_image = train_image.permute(1, 2, 0).numpy()
    valid_image = valid_image.permute(1, 2, 0).numpy()
    
    # Plot original and transformed images side by side
    axs[i, 0].imshow(image)
    axs[i, 0].set_title('Original Image')
    
    axs[i, 1].imshow(train_image)
    axs[i, 1].set_title('Transformed Image')

plt.tight_layout()

# Save the plot to the specified directory
save_dir = 'plot/data_augmentation'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
group = 'group1'
plt.savefig(f'{save_dir}/data_augmentation_{image_name}_{group}.png')

print(f"Plot saved to {save_dir}/data_augmentation_{image_name}.png")
