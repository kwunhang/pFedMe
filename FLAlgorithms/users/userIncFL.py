import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients

class UserIncFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_iters, optimizer, total_users , num_users):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_iters)
        self.total_users = total_users
        self.num_users = num_users
        self.rho = None
        self.delta = None
        self.qk = None
        
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    # def train(self, iters):
    #     LOSS = 0
    #     self.model.train()
    def train2(self, iters):
        LOSS = 0
        self.model.train()
        for iter in range(0, iters):  # local update 
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        return LOSS  
    
    def train(self, epochs):
        LOSS = 0
        # calculate aggregation weight
        self.model.eval()
        init_loss = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                init_loss += self.loss(output, y)
        self.qk = nn.Sigmoid(init_loss - self.rho)
        
        # local_model save the previous model
        self.clone_model_paramenter(self.model.parameters(), self.local_model)

        self.model.train()
        for iter in range(1, self.local_iters + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        self.delta = self.get_train_delta(self.model.parameters(), self.local_model)       
        
        return LOSS

    def init_rho(self):
        init_loss = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                init_loss += self.loss(output, y)
        self.rho = init_loss