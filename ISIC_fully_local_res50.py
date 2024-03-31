#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import torchvision
import torchvision.transforms as transforms
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from utils.model_utils import read_data, read_user_data, read_test_byClient
torch.manual_seed(0)
import time
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import os
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from analysis_utils import plot_function


load_dotenv()

learning_rate = 0.001
batch_size = 128
save_path = '/ISIC19/fully_res50local_base/'
checkpoint_path = 'check_point.pth'
EPOCHS = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_transforms():
    transforms = A.Compose([
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

# only resize, scale [-1, 1] and converting to tensor array[h,w,c] -> tensor[c,h,w]
def valid_transforms():
    transforms = A.Compose([
        A.PadIfNeeded(min_height=224, min_width=224),        
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(p=1.0),
    ], p=1.0)
    return transforms


class ISIC19Dataset(Dataset):
    """ISIC19 raw dataset in tensor (convert from numpy)"""
    """Assume the input is just resize to 244 and without normalization"""

    def __init__(self, data, transform=None):
        """
        Arguments:
            data (array): Array of X,Y tuple with tensor data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x, y = self.data[idx]

        if self.transform:
            # x = np.transpose(x.numpy(), (1,2,0)) #ToTensorV2 change[h,w,c] -> [c,h,w], revert the change
            
            # x = self.transform(x)
            
            x = np.array(x)
            image = {"image": x}
            image = self.transform(**image)["image"]
            x=image
                

        return x,y
    
    def get_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x, y = self.data[idx]

        return x,y
    
    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        images, labels = [[tensor[None] for tensor in subset] for subset in (images, labels)]
        images, labels = [torch.cat(subset, dim=0) for subset in (images, labels)]
        return images, labels

resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
num_ftrs = resnet50.fc.in_features
# freeze the pretrained weight
for param in resnet50.parameters():
    param.requires_grad = False
resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(512, 64),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(64, 8),
    nn.LogSoftmax(dim=1))
model = resnet50.to(device)

dataset = "ISIC19_raw"
_ , _ , train_data_tmp, test_data_tmp = read_data(dataset)
train_data,test_data = [], []
for _, data in train_data_tmp.items():
    train_data = train_data + data
for _, data in test_data_tmp.items():
    test_data = test_data + data
X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
X_train = torch.Tensor(X_train).view(-1, 3, 224, 224).type(torch.float32)
y_train = torch.Tensor(y_train).type(torch.int64)
X_test = torch.Tensor(X_test).view(-1, 3, 224, 224).type(torch.float32)
y_test = torch.Tensor(y_test).type(torch.int64)
train_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_train, y_train)]
test_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_test, y_test)]
train_data = ISIC19Dataset(train_data, transform=train_transforms())
test_data = ISIC19Dataset(test_data, transform=valid_transforms())

train_samples = len(train_data)
test_samples = len(test_data)

trainloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
testloader =  DataLoader(test_data, test_samples,num_workers=2)


criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

valid_ac = []
best_valid_acc = 0
rs_train_accs, rs_train_loss, rs_valid_acc, rs_valid_loss = [],[],[],[]

for epoch in range(1, EPOCHS+1):
    time_1 = time.time()
    train_loss, train_accs = [], 0
    model.train()
    for step, batch in tqdm(enumerate(trainloader, 1), total=len(trainloader)):
        model.train()
        images, labels = batch
        
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out)
        
        loss.backward()
        optimizer.step()
        
        accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()
        
        train_loss.append(loss.item())
        train_accs += accuracy

    train_accs /= train_samples
    
    print('epoch:', epoch, 
                  '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
                  '\ttrain accuracy:','{:.4f}'.format(train_accs),
                  '\ttime:', '{:.4f}'.format((time.time()-time_1)), 's')
    
    valid_loss, valid_accs = [], 0
    model.eval()

    for step, batch in enumerate(tqdm(testloader)):
        model.eval()
        images, label = batch
        out = model(images)
        loss = criterion(out)
        
        accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()
        
        valid_loss.append(loss.item())
        valid_accs += (accuracy)
    valid_accs /= train_samples
        
    print('epoch:', epoch, '/', EPOCHS,
          '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)),
          '\tvalid accuracy', '{:.4f}'.format(valid_accs))
    if valid_accs > best_valid_acc:
        torch.save(model.state_dict(), save_path + checkpoint_path)
        print(f'Model weights saved to {save_path}{checkpoint_path}')
        
# save result of acc and loss
rs_train_accs, rs_train_loss, rs_valid_acc, rs_valid_loss = [],[],[],[]

with h5py.File(save_path+ 'result/' +'result.h5', 'w') as hf:
    hf.create_dataset('rs_train_accs', data=rs_train_accs)
    hf.create_dataset('rs_train_loss', data=rs_train_loss)
    hf.create_dataset('rs_valid_acc', data=rs_valid_acc)
    hf.create_dataset('rs_valid_loss', data=rs_valid_loss)
    hf.close()
    
# plot graph
data = read_test_byClient(dataset, "final_test")
_ , _ , _, test_data_tmp = read_data(dataset)
test_data = [], []
for _, data in test_data_tmp.items():
    test_data = test_data + data
X_test, y_test = test_data['x'], test_data['y']
X_test = torch.Tensor(X_test).view(-1, 3, 224, 224).type(torch.float32)
y_test = torch.Tensor(y_test).type(torch.int64)
test_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_test, y_test)]
test_data = ISIC19Dataset(test_data, transform=valid_transforms())

test_samples = len(test_data)

testloader =  DataLoader(test_data, test_samples,num_workers=2)
predict_output = []
true_label = []
graph_path = ""
for step, batch in enumerate(tqdm(testloader)):
    model.eval()
    images, label = batch
    out = model(images)
    loss = criterion(out)
    
    predict = (torch.argmax(out, dim=1) )
    
    accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()
    
    print("accuracy in validation:", accuracy/len(label))
    true_label += label
    predict_output += predict
    
os.environ["SAVE_PLOT_PATH"] = save_path + "graph/"
plot_function(true_label, predict_output, "full_local")




