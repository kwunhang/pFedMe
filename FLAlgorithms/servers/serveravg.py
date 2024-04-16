import torch
import os

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
from analysis_utils import plot_function
from utils.model_utils import read_test_byClient, read_user_data, read_ISIC_data_byClient

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_iters, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_iters, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_test_byClient(dataset, "final_test")
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate,beta,lamda, local_iters, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self, start_iter=0):
        loss = []
        for glob_iter in range(start_iter, self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()
            self.save_best_model(glob_iter)

            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # print selected user to observe the train accuracy change
            print("selected user: ", end='')
            for user in self.selected_users:
                print(user.id, end=' ')
            print('')
                
            for user in self.selected_users:
                user.train(self.local_iters) #* user.train_samples
            self.aggregate_parameters()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
            
            # if(glob_iter % 100 == 99):
            #     self.save_model(glob_iter+1)
            #     self.save_all_client_model(glob_iter+1)
            #     self.save_results()

        self.send_parameters()
        # Evaluate model each interation
        self.evaluate()
        #print(loss)
        self.save_results()
        self.save_model()
        self.save_all_client_model()
    
    def plot_graph(self):
        acc_log = []
        # graph_name = self.dataset + self.algorithm
        graph_name = self.algorithm
        if (graph_name == "FedSelf"):
            graph_name = "LocalSelf"
        elif (graph_name == "FedInc"):
            graph_name = "IncFL"
        self.send_parameters()
        true_label, predict_label = self.test_and_get_label()
        log = plot_function(true_label, predict_label, graph_name)
        acc_log.append(log)
        
        plot_path = os.getenv('SAVE_PLOT_PATH')
        if plot_path == None or plot_path == "":
            plot_path = "plot"
        full_path = os.path.join(plot_path, "acc_log.txt")
        
        with open(full_path, 'w') as f:
            for i in acc_log:
                f.write(i+"\n")