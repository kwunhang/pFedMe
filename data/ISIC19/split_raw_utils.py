import warnings
warnings.filterwarnings('ignore')

import os
import time

import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

import shutil

pd.set_option('display.max_colwidth', None) 

base_dir = "/research/d2/fyp23/khlau1/"
source_dir = os.path.join(base_dir, "ISIC19_raw_img")
output_dir = os.path.join(base_dir, "ISIC19_raw_img_splited")
image_source_dir = os.path.join(source_dir, "ISIC_2019_Training_Input")

if not os.path.exists(base_dir):
    print("base_dir is not exist. Please double check")

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed) #LYZ:设置 Python 的哈希种子，以确保散列操作的一致性。
    np.random.seed(seed)#设置 NumPy 的随机种子
    torch.manual_seed(seed)#设置 NumPy 的随机种子
    torch.cuda.manual_seed(seed)# 设置 PyTorch 的 GPU 随机种子。
    torch.backends.cudnn.deterministic = True# 设置使用 CuDNN（GPU加速的深度学习库） 的时候使用确定性算法，以保证结果的可重复性。
    torch.backends.cudnn.benchmark = True#设置启用 CuDNN 的性能优化，以加快运行速度。

seed=8471
image_size = 224
seed_everything(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Currently using "{device.upper()}" device.')


meta_data = pd.read_csv(os.path.join(source_dir, "ISIC_2019_Training_Metadata_FL.csv"))
clients_set = set(meta_data['dataset'].unique())

data_truth = pd.read_csv(os.path.join(source_dir, "ISIC_2019_Training_GroundTruth.csv")).drop("UNK", axis=1)
data_truth["labels"] = data_truth.iloc[:,1:].idxmax(axis=1)

classes_to_int = {v: i for i, v in enumerate(data_truth.columns[1:-1])}
int_to_classes = {i: v for i, v in enumerate(data_truth.columns[1:-1])}

data_truth["labels"] = data_truth["labels"].map(classes_to_int)
print("data_truth:", data_truth.tail(20))
print("classes_to_int:\n", classes_to_int)

merged_df = data_truth.merge(meta_data, how='inner', on='image')
merged_df = merged_df.loc[:,['image','labels','dataset']]
print("merged df:\n", merged_df.tail())
merged_df.to_csv(os.path.join(source_dir, "data_source_truth.csv"))


data_split_result = {}
num_classes = len(classes_to_int)

for client in clients_set:
    print("current dataset:", client)
    # train_data = pd.DataFrame(columns=merged_df.columns)
    # valid_data = pd.DataFrame(columns=merged_df.columns)
    # test_data = pd.DataFrame(columns=merged_df.columns)

    client_data = merged_df.loc[merged_df['dataset']==client]
    # for class_idx in range(num_classes):
    # class_data = client_data[client_data["labels"] == class_idx]
    train_valid_split, test_data = train_test_split(client_data, test_size=0.15, stratify=client_data["labels"], random_state=seed)
    train_data, valid_data = train_test_split(train_valid_split, test_size=0.15/0.85, stratify=train_valid_split["labels"], random_state=seed)
    # 重置索引
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    data_split_result[client] = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }


# checking data count correct
the_sum = 0
for name, client in data_split_result.items():
    print("client:", name)
    for i,tmp_df in client.items():
        the_sum += tmp_df.shape[0]
        print(i,"set:",tmp_df.shape[0])
    
print("**total sum:" ,the_sum)

# split the data source and store in sperate dir


dir_make = ["train", "valid", "test"]

for dir in dir_make:
    cur_dir = os.path.join(output_dir, dir)
    for client in clients_set:
        cur_clients_dir = os.path.join(cur_dir, client)
        if not os.path.exists(cur_clients_dir):
            os.makedirs(cur_clients_dir)


for client, split_data_dict in data_split_result.items():
    for data_type, data_df in split_data_dict.items():
        data_df.to_csv(os.path.join(output_dir, data_type, client, "data_truth.csv"))
        for _, data in data_df.iterrows():
            shutil.copy(os.path.join(image_source_dir,data["image"] + ".jpg"),
                        os.path.join(output_dir, data_type, client))