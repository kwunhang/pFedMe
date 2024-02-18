import torch
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from analysis_utils import plot_function


# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_iters, optimizer, num_users,times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_iters, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_iters, optimizer ,total_users , num_users)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

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

    def train(self, start_iter = 0):
        loss = []
        for glob_iter in range(start_iter, self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model with one step update")
            print("")
            self.evaluate_one_step()
            self.save_best_model(glob_iter)

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            # print selected user to observe the train accuracy change
            print("selected user: ", end='')
            for user in self.selected_users:
                print(user.id, end=' ')
            print('')

            for user in self.selected_users:
                user.train(self.local_iters) #* user.train_samples
                
            self.aggregate_parameters()
            # if(glob_iter % 100 == 99):
            #     self.save_model(glob_iter+1)
            #     self.save_all_client_model(glob_iter+1)
            #     self.save_results()

        self.save_results()
        self.save_model()
        self.save_all_client_model()
        
    def trainAllClient(self, step):
        for user in self.users:
            for i in range(step):
                user.train_one_step()
                
    def plot_graph(self):
        acc_log = []
        graph_name = self.dataset + self.algorithm
        self.send_parameters()
        true_label, predict_label = self.test_and_get_label()
        acc = plot_function(true_label, predict_label, graph_name)
        acc_log.append(acc)
        
        self.trainAllClient(step=1)
        true_label, predict_label = self.test_and_get_label()
        acc = plot_function(true_label, predict_label, "{}(PM)1step".format(graph_name))
        acc_log.append(acc)
        
        self.trainAllClient(step=4)
        true_label, predict_label = self.test_and_get_label()
        acc = plot_function(true_label, predict_label, "{}(PM)5step".format(graph_name))
        acc_log.append(acc)
        
        self.trainAllClient(step=5)
        true_label, predict_label = self.test_and_get_label()
        acc = plot_function(true_label, predict_label, "{}(PM)10step".format(graph_name))
        acc_log.append(acc)
                
        plot_path = os.getenv('SAVE_PLOT_PATH')
        if plot_path == None or plot_path == "":
            plot_path = "plot"
        full_path = os.path.join(plot_path, "acc_log.txt")
        
        with open(full_path, 'w') as f:
            for i in acc_log:
                f.write(i+"\n")
