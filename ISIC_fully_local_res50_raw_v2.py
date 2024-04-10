#!/usr/bin/env python
# 256 batch size, weight cross entropy loss, adam optimizer
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
from utils.transform_utils import ISIC_raw_train_transforms, ISIC_raw_valid_transforms
from utils.model_utils import ISIC19DatasetRawImage


load_dotenv()

EPOCHS = 300
learning_rate = 0.001
batch_size = 256
save_path = 'ISIC19/fully_res50local_base_raw_v2/'
checkpoint_path = 'check_point.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

base_path = os.getenv('STORAGE_PATH')
if not(base_path == None or base_path == ""):
    save_path = os.path.join(base_path, save_path)

print(save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)



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

dataset = "ISIC19_raw_img_splited"
_ , _ , train_data_tmp, test_data_tmp = read_data(dataset)
train_data, test_data = [np.array([])]*2
for _, data in train_data_tmp.items():
    train_data = train_data.extend(data)
    
for _, data in train_data_tmp.items():
    test_data = test_data.extend(data)

train_data = ISIC19DatasetRawImage(train_data, transform=ISIC_raw_train_transforms())
test_data = ISIC19DatasetRawImage(test_data, transform=ISIC_raw_valid_transforms())

train_samples = len(train_data)
test_samples = len(test_data)

trainloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=2)
testloader =  DataLoader(test_data, batch_size,num_workers=2)

weighting_loss = []
y_train = [y for _,y in train_data]
for i in range(8):
    weighting_loss.append(train_samples/(torch.sum(y_train==i).item()))

weighting_loss = torch.tensor(weighting_loss)
criterion = nn.NLLLoss(weight=weighting_loss.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

valid_ac = []
best_valid_acc = 0
rs_train_accs, rs_train_loss, rs_valid_acc, rs_valid_loss = [],[],[],[]

for epoch in range(1, EPOCHS+1):
    time_1 = time.time()
    train_loss, train_accs = [], 0
    model.train()
    for step, batch in enumerate(tqdm(trainloader)):
        model.train()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()
        
        accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()
        
        train_loss.append(loss.item())
        train_accs += accuracy

    train_accs /= train_samples
    cur_train_loss = np.mean(train_loss)
    print('epoch:', epoch, 
                  '\ttrain loss:', '{:.4f}'.format(np.mean(cur_train_loss)),
                  '\ttrain accuracy:','{:.4f}'.format(train_accs),
                  '\ttime:', '{:.4f}'.format((time.time()-time_1)), 's')
    rs_train_accs.append(train_accs)
    rs_train_loss.append(cur_train_loss)
    
    valid_loss, valid_accs = [], 0
    model.eval()
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(testloader)):
            model.eval()
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            out = model(images)
            loss = criterion(out, labels)
            
            accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()
            
            valid_loss.append(loss.item())
            valid_accs += (accuracy)
    valid_accs /= test_samples
    
    cur_valid_loss = np.mean(valid_loss)
        
    print('epoch:', epoch, '/', EPOCHS,
          '\tvalid loss:', '{:.4f}'.format(cur_valid_loss),
          '\tvalid accuracy', '{:.4f}'.format(valid_accs))
    best_valid_acc = max(rs_valid_acc) if len(rs_valid_acc) >0 else 0
    rs_valid_acc.append(valid_accs)
    rs_valid_loss.append(cur_valid_loss)
    
    
    if valid_accs > best_valid_acc:
        torch.save(model.state_dict(), save_path + checkpoint_path)
        print(f'Model weights saved to {save_path}{checkpoint_path}')
        
# save result of acc and loss
if not os.path.exists(save_path+ 'result/'):
    os.makedirs(save_path+ 'result/')

with h5py.File(save_path+ 'result/' +'result.h5', 'w') as hf:
    hf.create_dataset('rs_train_accs', data=rs_train_accs)
    hf.create_dataset('rs_train_loss', data=rs_train_loss)
    hf.create_dataset('rs_valid_acc', data=rs_valid_acc)
    hf.create_dataset('rs_valid_loss', data=rs_valid_loss)
    hf.close()
    
# plot graph
data = read_test_byClient(dataset, "final_test")
_ , _ , _, test_data_tmp = read_data(dataset)
X_test, y_test = [np.array([])]*2
for _, data in test_data_tmp.items():
    X_test = np.append(X_test,data['x'])
    y_test = np.append(y_test, data['y'])
X_test = torch.Tensor(X_test).view(-1, 3, 224, 224).type(torch.float32)
y_test = torch.Tensor(y_test).type(torch.int64)
test_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_test, y_test)]
test_data = ISIC_raw_valid_transforms(test_data, transform=ISIC_raw_valid_transforms())

test_samples = len(test_data)

testloader =  DataLoader(test_data, batch_size,num_workers=2)
predict_output = []
true_label = []
graph_path = ""
model.eval()
accs = []
with torch.no_grad():
    for step, batch in enumerate(tqdm(testloader)):
        model.eval()
        images, labels = batch
        true_label.extend(labels.numpy())
        images, labels = images.to(device), labels.to(device)
        
        out = model(images)
        
        predict = (torch.argmax(out, dim=1) )
        
        accuracy = (torch.sum(torch.argmax(out, dim=1) == labels)).item()

        predict_output.extend(predict.cpu().numpy())
        accs.append(accuracy*1.0)
    print("accuracy in validation:", sum(accs)/len(true_label))
    
os.environ["SAVE_PLOT_PATH"] = save_path + "graph/"
plot_function(true_label, predict_output, "full_local")
