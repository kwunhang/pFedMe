import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_iters, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_iters)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        elif(model[1] == "se_resnext50"):
            weighting_loss = []
            y_train = torch.Tensor([y for _,y in train_data]).type(torch.int64)
            for i in range(8):
                num_i = torch.sum(y_train==i).item()
                weight = len(train_data)/(torch.sum(y_train==i).item()) if num_i != 0 else 1
                weighting_loss.append(weight)
            weighting_loss = torch.tensor(weighting_loss)
            self.loss = nn.NLLLoss(weight=weighting_loss.to(device))
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads): # didnt use
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for iter in range(1, self.local_iters + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        return LOSS

