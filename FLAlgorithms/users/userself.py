import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import numpy as np

# Implementation for FedAvg clients

class UserSelf(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_iters, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_iters)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.best_accuracy = 0
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
    
    # overwrite to print the self accuraccy
    def test_and_get_label(self):
        self.model.eval()
        predict_label = []
        true_label = [] 
        # test_acc = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                true_label.extend(y.numpy())
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                predict = (torch.argmax(output, dim=1) )
                predict_label.extend(predict.cpu().numpy())
                # test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
        acc = np.sum(true_label)*1.0/np.sum(predict_label)
        print(self.id + ", Test Accuracy:", acc )
        # print(self.id + ", Test Loss:", loss)
        return true_label, predict_label

