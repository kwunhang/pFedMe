import torch
import os

from FLAlgorithms.users.userIncFL import UserIncFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from analysis_utils import plot_function
from FLAlgorithms.optimizers.fedoptimizer import MySGD


# Implementation for per-FedAvg Server

class IncFL(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_iters, optimizer, num_users,times, epsilon):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_iters, optimizer, num_users, times)

        self.epsilon = epsilon
        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserIncFL(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_iters, optimizer ,total_users , num_users)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating IncFL server.")

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
        
        # load a global model instead of general init, and take it as client requirement
        model_path = os.getenv('INC_INIT_MODEL_PATH')
        if model_path == None or model_path == "":
            model_path = "IncFL/server_model.pt"
        self.model.load_state_dict(torch.load(model_path))
        self.send_parameters()
        for user in self.users:
            user.init_rho()
        
        for glob_iter in range(start_iter, self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()
            user_returns = []

            # Evaluate gloal model on user for each interation
            # print("Evaluate global model with one step update")
            # print("")
            self.evaluate()
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
            
                
            self.incfl_aggregate_parameters()
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
        log = plot_function(true_label, predict_label, graph_name)
        acc_log.append(log)
        
        plot_path = os.getenv('SAVE_PLOT_PATH')
        if plot_path == None or plot_path == "":
            plot_path = "plot"
        full_path = os.path.join(plot_path, "acc_log.txt")
        
        with open(full_path, 'w') as f:
            for i in acc_log:
                f.write(i+"\n")
    
    def update_lr(self, learning_rate):
        print("Update lr")
        for c in self.users:
            c.learning_rate = learning_rate
            c.optimizer = MySGD(c.model.parameters(), lr=c.learning_rate)
        print("Finish to update lr")