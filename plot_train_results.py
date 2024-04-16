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
from FLAlgorithms.servers.serverself import FedSelf
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
from analysis_utils import plot_cm, computePRF, plot_train_results, plot_function, compare_train_results

torch.manual_seed(0)

from dotenv import load_dotenv

load_dotenv()


path = "model/BNcompare/fed_perfedavg/Cifar10ByClient_PerAvg_p_0.01_0.001_15_5u_20b_5_0.h5"
path2 = "model/BNcompare/silo_perfedavg/Cifar10ByClient_PerAvg_p_0.01_0.001_15_5u_20b_5_0.h5"
path3 = "model/BNcompare/fed_pFedMe/Cifar10ByClient_pFedMe_p_0.01_1.0_15_5u_20b_5_5_0.01_avg.h5"
path4 = "model/BNcompare/silo_pFedMe/Cifar10ByClient_pFedMe_p_0.01_1.0_15_5u_20b_5_5_0.01_avg.h5"
# path = "model/BNcompare/fed_pFedMe/Cifar10ByClient_pFedMe_0.01_1.0_15_5u_20b_5_5_0.01_avg.h5"
# path2 = "model/BNcompare/silo_pFedMe/Cifar10ByClient_pFedMe_0.01_1.0_15_5u_20b_5_5_0.01_avg.h5"


print("path:", path)

# global model 
assert (os.path.exists(path))
assert (os.path.exists(path2))

# graph name implementation
graph_name = "FedBN_peravg"
graph_name2 = "SiloBN_peravg"
graph_name3 = "FedBN_pFedMe"
graph_name4 = "SiloBN_pFedMe"
    

compare_train_results([path, path2, path3, path4], [graph_name, graph_name2, graph_name3, graph_name4])


    