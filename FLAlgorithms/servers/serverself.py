import torch
import os

from FLAlgorithms.users.userself import UserSelf
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
from analysis_utils import plot_function

# Implementation for FedAvg Server

class FedSelf(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_iters, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_iters, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_test_byClient(dataset, "final_test")
        total_users = len(data[0])
        self.num_users = total_users
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserSelf(device, id, train, test, model, batch_size, learning_rate,beta,lamda, local_iters, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def train(self, start_iter=0):
        loss = []
        for glob_iter in range(start_iter, self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            # self.send_parameters()

            # Evaluate model each interation
            self.evaluate()
            self.save_best_model(glob_iter)

            self.selected_users = self.select_users(glob_iter,self.num_users)
                
            for user in self.selected_users:
                user.train(self.local_iters) #* user.train_samples


        # self.send_parameters()
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
        # self.send_parameters()
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
        
    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_GM = self.testGM()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        glob_acc_GM = np.sum(stats_GM[2])*1.0/np.sum(stats_GM[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        # print("stats_train[1]",stats_train[3][0])
        if (max(self.rs_glob_acc) == glob_acc):
            self.save_best= True
        last_best = self.rs_glob_acc[::-1].index(max(self.rs_glob_acc))
        if(last_best >= 30):
            print(f"!!!! There is already {last_best} round no improvemnt!")
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Accurancy: (Global model) ", glob_acc_GM)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)
    
    # overwrite
    def user_testGM(self, user):
        self.model.eval()
        test_acc = 0
        with torch.no_grad():
            for x, y in user.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
                
        accuracy = test_acc / user.test_samples
        if (accuracy > user.best_accuracy):
            user.best_accuracy = accuracy 
            user.best_model = copy.deepcopy(user.model)
        return test_acc, user.test_samples
