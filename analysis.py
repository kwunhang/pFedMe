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
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from analysis_utils import plot_cm, computePRF

from utils.model_utils import read_data, read_user_data

torch.manual_seed(0)


cpu = torch.device('cpu')

    
def analyse(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_iters, optimizer, numusers, K, personal_learning_rate, times, gpu, analysis_file):
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
            # model = ResNet18().to(device), model
            model = CifarNet().to(device), model
        
    if(model == "dnn"):
        if(dataset == "Mnist"):
            model = DNN().to(device), model
        else: 
            model = DNN(60,20,10).to(device), model
            
    if(model == "cnn_nBN"):
        if(dataset.startswith("Cifar10")):
            model = CifarNetNoBN().to(device), model

    if(model == "cnn"):
        if(dataset == "ISIC19"):
            model = ResNet18_isic19(8).to(device), model
    
    # path = "models/Cifar10_dist_caifarnet/FedAvg_server.pt"
    # model = model[0].to(cpu)
    # assert (os.path.exists(path))
    # model.load_state_dict(torch.load(path))
    # model = model.to(device),model
    # server.send_parameters()
    # server.evaluate()
    
    path = "models/{}/{}".format(dataset, analysis_file)
    print("path:", path)
    
    if(algorithm == "FedAvg"):
        server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)

    if(algorithm == "pFedMe"):
        server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, K, personal_learning_rate, 1)

    if(algorithm == "PerAvg"):
        server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)


    def plot_function(true_label, predict_label, graph_name, ):
        plot_cm(true_label,predict_label, graph_name)
        computePRF(true_label,predict_label, graph_name)
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("{} acc:".format(graph_name) ,accuracy)


    # global model 
    # assert (os.path.exists(path))
    # server.model.load_state_dict(torch.load(path))
    
    assert (os.path.exists(path))
    with h5py.File(path, 'r') as hf:
        # Assuming 'rs_glob_acc', 'rs_train_acc', and 'rs_train_loss' are the datasets in the h5 file
        print("Keys: %s" % hf.keys())
    # server.model = torch.load(path)
    server.model = server.model.to(device)
    server.send_parameters()
    server.update_server_BN()
    server.update_user_BN()
    server.aggregate_parameters()
    
    
    true_label, predict_label = server.test_and_get_label()
    plot_function(true_label, predict_label, algorithm)

    # personalize --> pFedMe and PerAvg only
    if(algorithm == "pFedMe" or algorithm == "PerAvg"):
        # make prediction with personal model with 1step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            if(algorithm == "pFedMe"):
                user.train(1)
            elif(algorithm == "PerAvg"):
                user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_function(true_label, predict_label, "{}(PM1)1step".format(algorithm))
        
        # make prediction to with personal model of user 0
        true_label = []
        predict_label = []
        model = server.users[0].model
        model.eval()
        with torch.no_grad():
            for user in server.users:
                # testloader = user.testloaderfull
                for x,y in user.testloaderfull:
                    true_label.extend(y.numpy())
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    predict = (torch.argmax(output, dim=1) )
                    predict_label.extend(predict.cpu().numpy())
                    
                    
        plot_function(true_label, predict_label, "{}(PM2)1step".format(algorithm))
        
        # 4 more steps 
        
        # make prediction with personal model with 1step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            if(algorithm == "pFedMe"):
                user.train(4)
            elif(algorithm == "PerAvg"):
                user.train_one_step()
                user.train_one_step()
                user.train_one_step()
                user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_function(true_label, predict_label, "{}(PM1)5step".format(algorithm))
        
        # make prediction to with personal model of user 0
        true_label = []
        predict_label = []
        model = server.users[0].model
        model.eval()
        with torch.no_grad():
            for user in server.users:
                # testloader = user.testloaderfull
                for x,y in user.testloaderfull:
                    true_label.extend(y.numpy())
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    predict = (torch.argmax(output, dim=1) )
                    predict_label.extend(predict.cpu().numpy())
        
        plot_function(true_label, predict_label, "{}(PM2)5step".format(algorithm))
        
        # make prediction with personal model with 5step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            if(algorithm == "pFedMe"):
                user.train(5)
            elif(algorithm == "PerAvg"):
                user.train_one_step()
                user.train_one_step()
                user.train_one_step()
                user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_function(true_label, predict_label, "{}(PM1)10step".format(algorithm))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10", "Cifar10ByClient", "ISIC19"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_iters", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--analysis_file", type=str, default="")
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
    analysis_file = args.analysis_file
    )


    