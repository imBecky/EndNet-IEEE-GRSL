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


class EndNet(nn.Module):
    def __init__(self):
        super().__init__()
        # x1 encoder: 144 → 16 → 32 → 64 → 128
        self.x1_enc = nn.Sequential(
            nn.Linear(144, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32),  nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # x2 encoder: 21 → 16 → 32 → 64 → 128
        self.x2_enc = nn.Sequential(
            nn.Linear(21, 16),  nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32),  nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # joint: 256 → 128 → 64 → 15
        self.joint_enc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 15)   # followed with cross_entropy so no ReLU
        )
        # decoder: 128 → 64 → 32 → 16 → original dim
        self.x1_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),       # why there's not ReLU anymore, and no BN?
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 144), nn.Sigmoid()
        )
        self.x2_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32), nn.Sigmoid(),
            nn.Linear(32, 16), nn.Sigmoid(),
            nn.Linear(16, 21), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        h1 = self.x1_enc(x1)
        h2 = self.x2_enc(x2)
        joint = torch.cat([h1, h2], dim=1)
        logits = self.joint_enc(joint)
        x1_rec = self.x1_dec(h1)
        x2_rec = self.x2_dec(h2)
        return logits, x1_rec, x2_rec

def train(x1_train, x2_train, x1_train_full, x2_train_full,
          x1_test, x2_test, x1_test_full, x2_test_full,
          y_train, y_test,
          lr_base=1e-3, beta_reg=1e-3, num_epochs=150,
          batch_size=64, print_cost=True):
    """
    :param x1_train: HSI data(reduced dimension) for VAE
    :param x2_train: LiDAR data(reduced dimension) for VAE
    :param x1_train_full: full HSI data for reconstruction loss
    :param x2_train_full: full LiDAR data for reconstruction loss
    """
    x1_train = torch.tensor(x1_train).to(device)
    x2_train = torch.tensor(x2_train).to(device)
    x1_train_full = torch.tensor(x1_train_full).to(device)
    x2_train_full = torch.tensor(x2_train_full).to(device)
    x1_test = torch.tensor(x1_test).to(device)
    x2_test = torch.tensor(x2_test).to(device)
    x1_test_full = torch.tensor(x1_test_full).to(device)
    x2_test_full = torch.tensor(x2_test_full).to(device)
    y_train = torch.tensor(y_train).to(device)
    y_test = torch.tensor(y_test).to(device)

    model = EndNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    y_train_cls = torch.argmax(y_train, dim=1)      # one_hot → cls idx
    y_test_cls = torch.argmax(y_test, dim=1)

    # recording
    costs, costs_dev = [], []    # what's the meaning of costs_dev
    train_acc, val_acc = [], []

    seed = 1
    m = x1_train.size(0)    # what does this mean?
