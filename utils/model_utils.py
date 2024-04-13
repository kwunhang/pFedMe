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
from utils.transform_utils import ISIC_raw_train_transforms, ISIC_raw_valid_transforms

from dotenv import load_dotenv

load_dotenv()


IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def suffer_data(data):
    data_x = data['x']
    data_y = data['y']
        # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    return (data_x, data_y)
    
def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts +1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x,data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)

def read_cifa_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data),shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),shuffle=False)

    for _, train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _, train_data in enumerate(testloader,0):
        testset.data, testset.targets = train_data

    random.seed(1)
    np.random.seed(1)
    NUM_USERS = 20 # should be muitiple of 10
    NUM_LABELS = 3
    # Setup directory for train/test data
    train_path = './data/train/cifa_train_100.json'
    test_path = './data/test/cifa_test_100.json'
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())
    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    cifa_data = []
    for i in trange(10):
        idx = cifa_data_label==i
        cifa_data.append(cifa_data_image[idx])


    print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
    users_lables = []

    ###### CREATE USER DATA SPLIT #######
    # Assign 100 samples to each user
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for j in range(NUM_LABELS):  # 3 labels for each users
            #l = (2*user+j)%10
            l = (user + j) % 10
            print("L:", l)
            X[user] += cifa_data[l][idx[l]:idx[l]+10].tolist()
            y[user] += (l*np.ones(10)).tolist()
            idx[l] += 10

    print("IDX1:", idx)  # counting samples for each labels

    # Assign remaining sample by power law
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)
    # print("here:",props/np.sum(props,(1,2), keepdims=True))
    #props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    #    props/np.sum(props, (1, 2), keepdims=True)
    #idx = 1000*np.ones(10, dtype=np.int64)
    # print("here2:",props)
    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):  # 4 labels for each users
            # l = (2*user+j)%10
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples)  + numran1 #+ 200
            if(NUM_USERS <= 20): 
                num_samples = num_samples * 2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len os user:", user, j,
                    "len data", len(X[user]), num_samples)

    print("IDX2:", idx) # counting samples for each labels

    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    # Setup 5 users
    # for i in trange(5, ncols=120):
    for i in range(NUM_USERS):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)

        num_samples = len(X[i])
        train_len = int(0.75*num_samples)
        test_len = num_samples - train_len

        #X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\
        
        test_data['users'].append(uname)
        test_data["user_data"][uname] =  {'x': X[i][:test_len], 'y': y[i][:test_len]} 
        test_data['num_samples'].append(test_len)

        train_data["user_data"][uname] =  {'x': X[i][test_len:], 'y': y[i][test_len:]}
        train_data['users'].append(uname)
        train_data['num_samples'].append(train_len)

    return train_data['users'], _ , train_data['user_data'], test_data['user_data']

def read_data_byClient(dataset):
    data_path = os.getenv('DATA_PATH')
    if data_path == None or data_path == "":
        data_path = 'data'
    train_data_dir = os.path.join(data_path,dataset,'data', 'train')
    test_data_dir = os.path.join(data_path,dataset,'data', 'test')
    clients = []
    train_data = {}
    test_data = {}
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.npz')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'rb') as inf:
            client_train_data = np.load(inf,allow_pickle=True)['data'].tolist()
        user = str(f).split('.')[0]
        clients.append(user)
        train_data[user] = client_train_data

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.npz')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'rb') as inf:
            client_test_data = np.load(inf,allow_pickle=True)['data'].tolist()
        test_data[str(f).split('.')[0]] = client_test_data
    
    return clients, train_data, test_data

def read_ISIC_data_byClient(dataset, test="test"):
    data_path = os.getenv('DATA_PATH')
    train_data_dir = os.path.join(data_path,dataset, 'train')
    test_data_dir = os.path.join(data_path,dataset, test)
    
    clients = []
    train_data = {}
    test_data = {}
    
    train_clients_dir = os.listdir(train_data_dir)
    for client in train_clients_dir:
        data_dir = os.path.join(train_data_dir, client)
        if not os.path.isdir(data_dir):
            continue
        clients.append(client)
        data_file = os.path.join(data_dir, "data_truth.csv")
        if not os.path.exists(data_file):
            print("data_file is not exist!\n",data_file)
        data_df = pd.read_csv(data_file)
        use_train_data = []
        for _, row in data_df.iterrows():
            image = Image.open(os.path.join(data_dir, row["image"] + ".jpg"))
            image = np.array(image)
            label = torch.as_tensor(row["labels"], dtype=torch.int64)
            use_train_data.append((image,label))

        train_data[client] = use_train_data
        
    test_clients_dir = os.listdir(test_data_dir)
    for client in test_clients_dir:
        data_dir = os.path.join(test_data_dir, client)
        if not os.path.isdir(data_dir):
            continue
        data_file = os.path.join(data_dir, "data_truth.csv")
        if not os.path.exists(data_file):
            print("data_file is not exist!\n",data_file)
        data_df = pd.read_csv(data_file)
        use_test_data = []
        for _, row in data_df.iterrows():
            image = Image.open(os.path.join(data_dir, row["image"] + ".jpg"))
            image = np.array(image)
            label = torch.as_tensor(row["labels"], dtype=torch.int64)
            use_test_data.append((image,label))
        
        test_data[client] = use_test_data
    return clients ,train_data, test_data

def read_test_byClient(dataset, folder_name):
    if dataset == "ISIC19_raw_img_splited":
        clients ,train_data, test_data = read_ISIC_data_byClient(dataset, "final_test")
        return clients,[] ,train_data, test_data
    data_path = os.getenv('DATA_PATH')
    if data_path == None or data_path == "":
        data_path = 'data'
    train_data_dir = os.path.join(data_path,dataset,'data', 'train')
    test_data_dir = os.path.join(data_path,dataset,'data', folder_name)
    clients = []
    train_data = {}
    test_data = {}
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.npz')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'rb') as inf:
            client_train_data = np.load(inf,allow_pickle=True)['data'].tolist()
        user = str(f).split('.')[0]
        clients.append(user)
        train_data[user] = client_train_data

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.npz')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'rb') as inf:
            client_test_data = np.load(inf,allow_pickle=True)['data'].tolist()
        test_data[str(f).split('.')[0]] = client_test_data
    
    return clients,[] ,train_data, test_data

def read_data(dataset):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    # if(dataset == "Cifar10"):
    #     clients, groups, train_data, test_data = read_cifa_data()
    #     return clients, groups, train_data, test_data
    
    if(dataset == "Cifar10ByClient" or dataset == "ISIC19" or dataset == "ISIC19_raw"):
        clients, train_data, test_data = read_data_byClient(dataset)
        return clients, [], train_data, test_data
    
    if(dataset == "ISIC19_raw_img_splited"):
        clients, train_data, test_data = read_ISIC_data_byClient(dataset)
        return clients, [], train_data, test_data
        

    train_data_dir = os.path.join('data',dataset,'data', 'train')
    test_data_dir = os.path.join('data',dataset,'data', 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage(image)
    return image

def read_user_data_ISIC_img(id,train_data,test_data):
    train_data = ISIC19DatasetRawImage(train_data, transform=ISIC_raw_train_transforms())
    test_data = ISIC19DatasetRawImage(test_data, transform=ISIC_raw_valid_transforms())
    print("print sample")
    print(train_data.get_sample)
    return id, train_data, test_data
    

def read_user_data(index,data,dataset):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    
    # special case for data source is raw image
    if (dataset == "ISIC19_raw_img_splited"):
        id, train_data, test_data = read_user_data_ISIC_img(id,train_data,test_data)
        return id, train_data, test_data

    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if(dataset == "Mnist"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset == "Cifar10" or dataset =="Cifar10ByClient"):
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif(dataset.startswith("ISIC19")):
        X_train = torch.Tensor(X_train).view(-1, 3, 224, 224).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 3, 224, 224).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    
    # train_data = [(x, y) for x, y in zip(X_train, y_train)]
    # test_data = [(x, y) for x, y in zip(X_test, y_test)]
    
    if(dataset == "ISIC19_raw"):
        print("load dataset for ISIC19_raw case")
        train_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_train, y_train)]
        test_data = [(transforms.ToPILImage()(x), y) for x, y in zip(X_test, y_test)]
        train_data = ISIC19Dataset(train_data, transform=new_train_transforms())
        test_data = ISIC19Dataset(test_data, transform=new_valid_transforms())
        print("print sample")
        print(train_data.get_sample)
    else:
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
    
    return id, train_data, test_data

def train_transforms():
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=(-30, 30))]), p=0.2),
            # transforms.RandomApply(torch.nn.ModuleList([AddGaussianNoise(0., 1.)]), p=0.1),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(brightness=(0.9, 1.1)),
                transforms.ColorJitter(contrast=(0.8, 1.2)),
                ]), p = 0.05),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(degrees=0,translate=(0,0),shear=45)]), p=0.2),
            # transforms.RandomPerspective(distortion_scale=0.5, p=0.05),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(hue=(0.3))]), p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transform

def new_train_transforms():
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
def new_valid_transforms():
    transforms = A.Compose([
        A.PadIfNeeded(min_height=224, min_width=224),        
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(p=1.0),
    ], p=1.0)
    return transforms

def valid_transforms():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transform

class Metrics(object):
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read
        metrics_dir = os.path.join('out', self.params['dataset'], 'metrics_{}_{}_{}_{}_{}.json'.format(
            self.params['seed'], self.params['optimizer'], self.params['learning_rate'], self.params['num_epochs'], self.params['mu']))
        #os.mkdir(os.path.join('out', self.params['dataset']))
        if not os.path.exists('out'):
            os.mkdir('out')
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)

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

class ISIC19DatasetRawImage(Dataset):
    """ISIC19 raw dataset in tensor (convert from numpy)"""

    def __init__(self, data,transform=None):
        """
        Arguments:
            data ():
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
        image, label = self.data[idx]

        if self.transform:
            sample = {"image": image}
            image = self.transform(**sample)["image"]

        return image, label
    
    def get_sample(self, idx):
        row = self.data.loc[idx].squeeze()
        
        image = Image.open(os.path.join(self.data_dir, row["image"] + ".jpg"))
        
        image = np.array(image)

        label = torch.as_tensor(row["labels"], dtype=torch.int64)

        if self.transform:
            sample = {"image": image}
            image = self.transform(**sample)["image"]

        return image, label
    
    def collate_fn(self, batch):
        images, labels = list(zip(*batch))
        images, labels = [[tensor[None] for tensor in subset] for subset in (images, labels)]
        images, labels = [torch.cat(subset, dim=0) for subset in (images, labels)]
        return images, labels