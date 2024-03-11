import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
torch.manual_seed(0)

# FedBN version
class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_iters = 0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_iters = local_iters
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, num_workers=1)
        self.testloader =  DataLoader(test_data, self.batch_size,num_workers=1)
        self.testloaderfull = DataLoader(test_data, self.batch_size,num_workers=1)
        self.trainloaderfull = DataLoader(train_data, self.batch_size,num_workers=1)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.best_model = None
    
    # fedBN, use original batch layer param
    # Refer to torch src code, parameter is sub-function of named_parameters, the order shd be the same
    def set_parameters(self, model):
        for old_param, new_layer, local_param in zip(self.model.parameters(), model.named_parameters(), self.local_model):
            layer_name, new_param = new_layer
            if layer_name.startswith("batch"):
                # keep all BN layer params
                # print("debug: check for detect BN layer")
                local_param.data = old_param.data.clone()
            else:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
        return test_acc, self.test_samples
    
    def test_and_get_label(self):
        self.model.eval()
        predict_label = []
        true_label = [] 
        test_acc = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                true_label.extend(y.numpy())
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                predict = (torch.argmax(output, dim=1) )
                predict_label.extend(predict.cpu().numpy())
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
                # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
                
        # print accuracy of each client
        print(self.id + ", Test Accuracy:", test_acc )
        return true_label, predict_label

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y) * y.shape[0]
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
            
            # change for case not full loader
            loss /= self.train_samples
        return train_acc, loss , self.train_samples
    
    def test_persionalized_model(self):
        self.model.eval()
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc,self.test_samples

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # loss += self.loss(output, y)
                loss += self.loss(output, y) * y.shape[0]
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
            loss /= self.train_samples
        self.update_parameters(self.local_model)
        return train_acc, loss , self.train_samples
    
    def get_train_delta(self, param, old_model_param):
        ret_model = copy.deepcopy(param)
        for new_param, prev_param, ret_param in zip(param, old_model_param , ret_model):
            ret_param.data = prev_param.data - new_param.data
        return ret_model

    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self, model_path=None):
        if model_path == None:
            model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    def user_upload_BN(self):
        self.model.train()
        for X, y in self.trainloader:
            # self.optimizer.zero_grad()
            X = X.to(self.device)
            output = self.model(X)
            # loss = self.loss(output, y)
            # loss.backward()
    
    def new_dataloader(self, train_data, test_data):
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, num_workers=1)
        self.testloader =  DataLoader(test_data, self.batch_size,num_workers=1)
        self.testloaderfull = DataLoader(test_data, self.batch_size,num_workers=1)
        self.trainloaderfull = DataLoader(train_data, self.batch_size,num_workers=1)
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))