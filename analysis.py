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
         local_iters, optimizer, numusers, K, personal_learning_rate, times, gpu):
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

    if(algorithm == "FedAvg"):
        server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)
        path = "models/ISIC19/FedAvg_server_285.pt"
        assert (os.path.exists(path))
        server.model.load_state_dict(torch.load(path))
        # server.model = torch.load(path)
        server.model = server.model.to(device)
        server.send_parameters()
        server.update_user_BN()
        server.aggregate_parameters()
        
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "FedAvg")
        computePRF(true_label,predict_label, "FedAvg")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("fedavg acc:" ,accuracy)
        # server.train()
        
    if(algorithm == "pFedMe"):
        server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, K, personal_learning_rate, 1)
        path = "models/ISIC19/pFedMe_server_294.pt"
        server.model.load_state_dict(torch.load(path))
        # server.model = torch.load(path)
        server.model = server.model.to(device)
        server.send_parameters()
        server.update_user_BN()
        server.aggregate_parameters()
            # make prediction to global model
        true_label = []
        predict_label = []
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "pFedMe(GM)")
        computePRF(true_label,predict_label, "pFedMe(GM)")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme(gm) acc:" ,accuracy)
        
        # make prediction with personal model with 1step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            user.train(1)
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PFedMe(PM1)1step")
        computePRF(true_label,predict_label, "PFedMe(PM1)1step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme pm1-1 acc:" ,accuracy)
        
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
                    
        plot_cm(true_label,predict_label, "PFedMe(PM2)1step")
        computePRF(true_label,predict_label, "PFedMe(PM2)1step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme pm2-1 acc:" ,accuracy)
        
        # 4 more steps
        
        # make prediction with personal model with 1step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            user.train(4)
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PFedMe(PM1)5step")
        computePRF(true_label,predict_label, "PFedMe(PM1)5step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme pm1-5 acc:" ,accuracy)
        
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
                    
        plot_cm(true_label,predict_label, "PFedMe(PM2)5step")
        computePRF(true_label,predict_label, "PFedMe(PM2)5step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme pm2-5 acc:" ,accuracy)
        
        # 
        true_label = []
        predict_label = []
        for user in server.users:
            user.train(5)
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PFedMe(PM1)10step")
        computePRF(true_label,predict_label, "PFedMe(PM1)10step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("pfedme pm1-10 acc:" ,accuracy)


    if(algorithm == "PerAvg"):
        server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_iters, optimizer, numusers, 1)
        path = "models/ISIC19/PerAvg_server_259.pt"
        # path = "models/Cifar10_dist_caifarnet/PerAvg_server.pt"
        # server.model = torch.load(path)
        server.model.load_state_dict(torch.load(path))
        server.model = server.model.to(device)
        server.send_parameters()
        server.update_user_BN()
        server.aggregate_parameters()
        # make prediction to global model
        true_label = []
        predict_label = []
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PerFed(GM)")
        computePRF(true_label,predict_label, "PerFed(GM)")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg GM acc:" ,accuracy)
        
        # make prediction with personal model with 1step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PerFed(PM1)1step")
        computePRF(true_label,predict_label, "PerFed(PM1)1step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg pm1-1 acc:" ,accuracy)
        
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
                    
        plot_cm(true_label,predict_label, "PerFed(PM2)1step")
        computePRF(true_label,predict_label, "PerFed(PM2)1step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg pm2-1 acc:" ,accuracy)
        # 4 more step
        
        # make prediction with personal model with 5step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            user.train_one_step()
            user.train_one_step()
            user.train_one_step()
            user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PerFed(PM1)5step")
        computePRF(true_label,predict_label, "PerFed(PM1)5step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg pm1-5 acc:" ,accuracy)
        
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
                    
        plot_cm(true_label,predict_label, "PerFed(PM2)5step")
        computePRF(true_label,predict_label, "PerFed(PM2)5step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg pm2-5 acc:" ,accuracy)
        
        # make prediction with personal model with 5step gradient decent
        true_label = []
        predict_label = []
        for user in server.users:
            user.train_one_step()
            user.train_one_step()
            user.train_one_step()
            user.train_one_step()
            
        true_label, predict_label = server.test_and_get_label()
        plot_cm(true_label,predict_label, "PerFed(PM1)10step")
        computePRF(true_label,predict_label, "PerFed(PM1)10step")
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("perfedavg pm1-10 acc:" ,accuracy)


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
    gpu=args.gpu
    )


    