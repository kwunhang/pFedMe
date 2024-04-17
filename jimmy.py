#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverself import FedSelf
from FLAlgorithms.servers.serverIncFL import IncFL
from FLAlgorithms.trainmodel.models import *
from utils.model_utils import read_test_byClient, read_user_data, read_ISIC_data_byClient
from utils.plot_utils import *
import torch
import torchvision
import pretrainedmodels
import ssl
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pickle 



torch.manual_seed(0)
random.seed(0)

from dotenv import load_dotenv

load_dotenv()


def predict(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_iters, optimizer, numusers, K, personal_learning_rate, times, gpu, restore, itered, epsilon):

    # Get device status: Check GPU or CPU
    # cpu = "cpu"
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    print("device:", device)

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset.startswith("Cifar10")):
                model = ResNet18().to(device), model
                # model = CifarNet().to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else: 
                model = DNN(60,20,10).to(device), model
        
        if(model == "cnn_nBN"):
            if(dataset.startswith("Cifar10")):
                model = CifarNetNoBN().to(device), model
        if(model == "cnn"):
            if(dataset.startswith("ISIC19")):
                model = ResNet18_isic19(8).to(device), model
        if(model == "resnet50"):
            # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            resnet50 = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            num_ftrs = resnet50.fc.in_features
            resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 8), nn.LogSoftmax(dim=1))
            model = resnet50.to(device), model
            # model = torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device), model
        if(model == "resnet50_v2"):
            # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
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
            model = resnet50.to(device), model
            # model = torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device), model
        # model = torch.hub.load("pytorch/vision", "efficientnet_b3", weights="IMAGENET1K_V1")
        if(model == "se_resnext50"):
            os.environ["TORCH_HOME"] = "/research/d2/fyp23/khlau1/pretrainedmodels/"
            ssl._create_default_https_context = ssl._create_unverified_context
            seResNext = pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained='imagenet')
            num_ftrs = seResNext.last_linear.in_features
            # freeze the pretrained weight
            for param in seResNext.parameters():
                param.requires_grad = False
            seResNext.last_linear = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 8),
                nn.LogSoftmax(dim=1))
            model = seResNext.to(device), model

        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, i)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, K, personal_learning_rate, i)

        if(algorithm == "PerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, i)
        
        if(algorithm == "LocalSelf"):
            server = FedSelf(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, i)

        if(algorithm == "incFL"):
            server = IncFL(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, i, epsilon)


            
        server.load_all_model()
        
        # load the real test set
        data = read_test_byClient(dataset, "final_test")
        total_users = len(data[0])
        for i in range(total_users):
            uid, train , test = read_user_data(i, data, dataset)
            # find user by id
            the_user = None
            for user in server.users:
                if user.id == uid:
                    the_user = user
                    break
            the_user.new_dataloader(train , test)
        
        ret = {}
        for c in server.users:
            true_label, predict_label = c.test_and_get_label()
            f1 = f1_score(true_label,predict_label,average='macro')
            acc = accuracy_score(true_label,predict_label)
            ret[server.users.id] = {"f1": f1, "acc": acc}
        return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10", "Cifar10ByClient", "ISIC19", "ISIC19_raw", "ISIC19_raw_img_splited"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn", "cnn_nBN", "resnet50", "resnet50_v2", "se_resnext50"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--epsilon", type=float, default=0.01, help="epcilon for IncFL adaptive lr ")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg", "LocalSelf", "incFL"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--restore", type=int, default=0, help="Restore the previous training, 0 mean no and 1 mean restore from restore folder")
    parser.add_argument("--itered", type=int, default=0, help="Number of iteration of previous training ")
    args = parser.parse_args()
    
    if(args.restore == 1 and args.times!= 1):
        print("restore currently just work for training 1 and need to adjust the time")
        exit()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_iters))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)
    
    
    env = {
        "FedAvg": "/research/d2/fyp23/khlau1/seResNext_fedAvg_model_save",
        "incFL": "/research/d2/fyp23/khlau1/seResNext_incFL_model_save",
        "PerAvg": "/research/d2/fyp23/khlau1/seResNext_peravg_model_save",
        "pFedMe": "/research/d2/fyp23/khlau1/pFedMe_raw_seResNext_model_save",
        "LocalSelf": "/research/d2/fyp23/khlau1/seResNext_fedSelf_model_save"
    }
    algo_client = {}
    for algo, path in env.items():
        os.environ['SAVE_MODEL_PATH'] = path
        
        ret = predict(
            dataset=args.dataset,
            algorithm = algo,
            model=args.model,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            beta = args.beta, 
            lamda = args.lamda,
            num_glob_iters=args.num_global_iters,
            local_iters=args.local_iters,
            optimizer= args.optimizer,
            numusers = args.numusers,
            K=args.K,
            personal_learning_rate=args.personal_learning_rate,
            times = args.times,
            gpu=args.gpu,
            restore=args.restore,
            itered=args.itered,
            epsilon=args.epsilon
            )
        algo_client[algo] = ret
    with open('algo.pkl', 'wb') as f:
        pickle.dump(algo_client, f)

