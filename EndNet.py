"""
@author: danfeng Hong
implemented by Binqian Huang
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch_utils as utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def initialize_parameters():
    utils.set_seed(1)


