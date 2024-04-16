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
from utils.plot_utils import *
import torch
from analysis_utils import plot_cm, computePRF, plot_train_results, plot_function

torch.manual_seed(0)

from dotenv import load_dotenv

load_dotenv()


cpu = torch.device('cpu')
    
def analyse(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_iters, optimizer, numusers, K, personal_learning_rate, times, gpu, analysis_file, itered, epsilon):
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    print("device:", device)
    if(model == "mclr"):
        if(dataset == "Mnist"):
            model = Mclr_Logistic().to(device), model
        else:
            model = Mclr_Logistic(60,10).to(device), model
            
    if(model == "cnn"):
        if(dataset == "Mnist"):
            model = Net().to(device), model
        elif(dataset == "Cifar10" or dataset == "Cifar10ByClient"):
            if "res" in analysis_file:
                model = ResNet18().to(device), model
            else:
                model = CifarNet().to(device), model
        elif(dataset == "ISIC19" or dataset == "ISIC19_raw"):
            model = ResNet18_isic19(8).to(device), model
        
    if(model == "dnn"):
        if(dataset == "Mnist"):
            model = DNN().to(device), model
        else: 
            model = DNN(60,20,10).to(device), model
            
    if(model == "cnn_nBN"):
        if(dataset.startswith("Cifar10")):
            model = CifarNetNoBN().to(device), model
    
    if(model == "resnet50"):
        # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        resnet50 = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", weights="IMAGENET1K_V2")
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 8), nn.LogSoftmax(dim=1))
        model = resnet50.to(device), model

    
    # path = "models/Cifar10_dist_caifarnet/FedAvg_server.pt"
    # model = model[0].to(cpu)
    # assert (os.path.exists(path))
    # model.load_state_dict(torch.load(path))
    # model = model.to(device),model
    # server.send_parameters()
    # server.evaluate()
    
    path = analysis_file
    print("path:", path)
    
    if(algorithm == "FedInc"):
        server = IncFL(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1, epsilon)
    
    if(algorithm == "FedAvg"):
        server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)

    if(algorithm == "pFedMe"):
        server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, K, personal_learning_rate, 1)

    if(algorithm == "PerAvg"):
        server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)


    # global model 
    assert (os.path.exists(analysis_file))
    

    server.plot_graph(analysis_file, analysis_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10", "Cifar10ByClient", "ISIC19", "ISIC19_raw"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn", "cnn_nBN", "resnet50"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg", "FedSelf", "FedInc"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--analysis_file", type=str, default="")
    parser.add_argument("--restore", type=int, default=0, help="Restore the previous training, 0 mean no and 1 mean restore from restore folder")
    parser.add_argument("--itered", type=int, default=0, help="Number of iteration of previous training ")
    parser.add_argument("--epsilon", type=float, default=0.01, help="epcilon for IncFL adaptive lr ")
    args = parser.parse_args()

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
    print("analysis_file       : {}".format(args.analysis_file))

    analyse(
    dataset=args.dataset,
    algorithm = args.algorithm,
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
    analysis_file = args.analysis_file,
    itered=args.itered,
    epsilon=args.epsilon
    )


    