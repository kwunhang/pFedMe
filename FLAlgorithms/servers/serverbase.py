import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

cpu = torch.device('cpu')
class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_iters, optimizer,num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_iters = local_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times
        self.save_best= False
        self.best_model = None
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self): #didnt use
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    # def add_parameters(self, server_model, user, ratio):
    #     # server model is server.model.state_dict
    #     user_model = user.model.state_dict()
    #     for key in server_model:
    #         # newdata = server_param.data + user_param.data.clone() * ratio
    #         server_model[key] = server_model[key] + user_model[key] * ratio

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    # def aggregate_parameters(self):
    #     assert (self.users is not None and len(self.users) > 0)
    #     global_model = self.model.state_dict()
    #     for key, data in global_model.items():
    #         global_model[key] =  (torch.zeros_like(data))
    #     total_train = 0
    #     #if(self.num_users = self.to)
    #     for user in self.selected_users:
    #         total_train += user.train_samples
    #     for user in self.selected_users:
    #         self.add_parameters(global_model, user, user.train_samples / total_train)
    #     self.model.load_state_dict(global_model)

    def save_model(self, global_iter=None):
        model_path = os.getenv('SAVE_MODEL_PATH')
        if model_path == None or model_path == "":
            model_path = "models"
            model_path = os.path.join(model_path, self.dataset)
        if global_iter:
            model_path = os.path.join(model_path, "iter_" + global_iter)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, self.algorithm + "_" + "server" + ".pt"))

    def save_all_client_model(self, global_iter=None, model_path=None):
        if model_path==None:
            model_path = os.getenv('SAVE_MODEL_PATH')
            if model_path == None or model_path == "":
                model_path = "models"
                model_path = os.path.join(model_path, self.dataset)
            if global_iter:
                model_path = os.path.join(model_path, "iter_" + global_iter)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        for user in self.users:
            user.save_model(model_path)
            
        # if global_iter:
        #     # torch.save(self.model, os.path.join(model_path, self.algorithm + "_" + "server" + "_" + str(global_iter) + ".pt"))
        #     torch.save(saveModel.state_dict(), os.path.join(model_path, self.algorithm + "_" + "server" + "_" + str(global_iter) + ".pt"))
        # else:
        #     # torch.save(self.model, os.path.join(model_path, self.algorithm + "_" + "server" + ".pt"))
        #     torch.save(saveModel.state_dict(), os.path.join(model_path, self.algorithm + "_" + "server" + ".pt"))
        

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # not in use
    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            newdata = server_param.data + user_param.data.clone() * ratio
            server_param.copy_(newdata)


    # def persionalized_aggregate_parameters(self):
    #     assert (self.users is not None and len(self.users) > 0)

    #     # store previous parameters
    #     previous_model = self.model.state_dict()
    #     global_model = self.model.state_dict()
    #     for key, data in global_model.items():
    #         global_model[key]= (torch.zeros_like(data))
    #     total_train = 0
    #     #if(self.num_users = self.to)
    #     for user in self.selected_users:
    #         total_train += user.train_samples

    #     # update global model including BN mean & var 
    #     for user in self.selected_users:
    #         self.add_parameters(global_model, user, user.train_samples / total_train)
    #         #self.add_parameters(user, 1 / len(self.selected_users))
    #     # self.model.load_state_dict(global_model)

    #     # aaggregate avergage model with previous model using parameter beta 
    #     # for pre_param, param in zip(previous_param, self.model.parameters()):
    #         # param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
    #     for key in previous_model:
    #         global_model[key] = (1 - self.beta)*previous_model[key] + self.beta*global_model[key]
    #         # param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
        
    #     self.model.load_state_dict(global_model)
    
    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    def incfl_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        
        factor_base = self.epsilon
        for user in self.select_users:
            factor_base += user.qk
        # factor_base += e
        cur_lr = self.learning_rate / factor_base
        
        for user in self.users:
            self.add_weight(self.model.parameters(),cur_lr,user.qk, user.delta)        
        # may ratio the weight with training sample       

    def add_weight(self, model_parameters, lr, qk, weight):
        for model_param, weight_param in zip(model_parameters, weight):
            model_param.data = model_param.data - lr * qk * weight_param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self, t= None):
        alg = self.dataset + "_" + self.algorithm
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_iters)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if not os.path.exists("./results/"):
            os.makedirs("./results/")
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_iters), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_iters)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_iters), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct
    
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
        return test_acc, user.test_samples

    def testGM(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = self.user_testGM(c)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def test_and_get_label(self):
        # self.model.eval()
        predict_label = []
        true_label = [] 
        for c in self.users:
            tl, pl = c.test_and_get_label()
            true_label.extend(tl)
            predict_label.extend(pl)
        
            
        return true_label, predict_label

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

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
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

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        if (max(self.rs_glob_acc_per) == glob_acc):
            self.save_best= True
        last_best = self.rs_glob_acc_per[::-1].index(max(self.rs_glob_acc_per))
        if(last_best >= 30):
            print(f"!!!! There is already {last_best} round no improvemnt!")
        # print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        if (max(self.rs_glob_acc_per) == glob_acc):
            self.save_best= True
        last_best = self.rs_glob_acc_per[::-1].index(max(self.rs_glob_acc_per))
        if(last_best >= 30):
            print(f"!!!! There is already {last_best} round no improvemnt!")
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        
    def update_user_BN(self):
        for c in self.users:
            c.user_upload_BN()
        print("updated user BN")

    
    def update_server_BN(self):
        global_model = self.model.state_dict()
        for key, data in global_model.items():
            if(key.endswith("running_var") or key.endswith("running_mean")):
                global_model[key]= (torch.zeros_like(data))
    
    def save_best_model(self, step = None, final = False):
        if (self.save_best== True or final==True):
            if (self.save_best== True):
                self.best_model = copy.deepcopy(self.model)
            model_path = os.getenv('SAVE_MODEL_PATH')
            if model_path == None or model_path == "":
                model_path = "models"
                model_path = os.path.join(model_path, self.dataset)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, "bestModel")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.best_model.state_dict(), os.path.join(model_path, self.algorithm + "_" + "server" + ".pt"))
            self.save_all_client_model(model_path=model_path)
            self.save_best= False
            
    # only load model after the data is readed, for the id purpose
    def load_all_model(self, model_path=None):
        if model_path ==None:
            model_path = os.getenv('SAVE_MODEL_PATH')
            model_path = os.path.join(model_path, "bestModel")
        model_files = os.listdir(model_path)
        model_files = [f for f in model_files if f.endswith('.pt')]
        for f in model_files:
            if "server" in f:
                if (self.algorithm + "_" + "server" + ".pt" != f):
                    print("!!!!!! The loading model of server seem have problem\nThe expect model file: {self.algorithm}_server.pt\nCurent load model file: {f}")
                path = os.path.join(model_path, f)
                self.model.load_state_dict(torch.load(path))
        for user in self.users:
            path = os.path.join(model_path, f"user_{user.id}.pt")
            assert (os.path.exists(path))
            user.model.load_state_dict(torch.load(path))
