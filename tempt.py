# ======================
# 1. 依赖
# ======================
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

# ======================
# 2. 网络定义
# ======================
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # ----- encoder -----
        # x1: 144 → 16 → 32 → 64 → 128
        self.x1_enc = nn.Sequential(
            nn.Linear(144, 16), nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32),  nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # x2: 21 → 16 → 32 → 64 → 128
        self.x2_enc = nn.Sequential(
            nn.Linear(21, 16),  nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Linear(16, 32),  nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Linear(32, 64),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True)
        )
        # joint: 256 → 128 → 64 → 15
        self.joint_enc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64),  nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Linear(64, 15)                       # 无激活，供 softmax_cross_entropy
        )
        # ----- decoder -----
        # 128 → 64 → 32 → 16 → 原维度
        self.x1_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32),  nn.Sigmoid(),
            nn.Linear(32, 16),  nn.Sigmoid(),
            nn.Linear(16, 144), nn.Sigmoid()
        )
        self.x2_dec = nn.Sequential(
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32),  nn.Sigmoid(),
            nn.Linear(32, 16),  nn.Sigmoid(),
            nn.Linear(16, 21),  nn.Sigmoid()
        )

    # forward: 返回 (logits, x1_recon, x2_recon)
    def forward(self, x1, x2):
        h1 = self.x1_enc(x1)
        h2 = self.x2_enc(x2)
        joint = torch.cat([h1, h2], dim=1)
        logits = self.joint_enc(joint)
        x1_re = self.x1_dec(h1)
        x2_re = self.x2_dec(h2)
        return logits, x1_re, x2_re

# ======================
# 3. 工具函数
# ======================
def l2_loss(model):
    """返回模型所有权重矩阵的 L2 范数平方和（不含 bias 和 BN）"""
    l2 = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 += torch.sum(param**2)
    return l2

def random_mini_batches(x1, x2, x1_full, x2_full, y, batch_size, seed):
    """模仿 TF 版随机 mini-batch 生成器"""
    np.random.seed(seed)
    m = x1.shape[0]
    permutation = np.random.permutation(m)
    num_batches = int(np.ceil(m / batch_size))
    batches = []
    for k in range(num_batches):
        idx = permutation[k*batch_size : (k+1)*batch_size]
        batches.append((x1[idx], x2[idx], x1_full[idx], x2_full[idx], y[idx]))
    return batches

# ======================
# 4. 训练函数
# ======================
def train_mynetwork(x1_train, x2_train, x1_train_full, x2_train_full,
                    x1_test,  x2_test,  x1_test_full,  x2_test_full,
                    y_train, y_test,
                    lr_base=1e-3, beta_reg=1e-3, num_epochs=150,
                    batch_size=64, print_cost=True):
    # ---------- 数据 ----------
    x1_train = torch.tensor(x1_train, dtype=torch.float32).to(device)
    x2_train = torch.tensor(x2_train, dtype=torch.float32).to(device)
    x1_train_full = torch.tensor(x1_train_full, dtype=torch.float32).to(device)
    x2_train_full = torch.tensor(x2_train_full, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    x1_test = torch.tensor(x1_test, dtype=torch.float32).to(device)
    x2_test = torch.tensor(x2_test, dtype=torch.float32).to(device)
    x1_test_full = torch.tensor(x1_test_full, dtype=torch.float32).to(device)
    x2_test_full = torch.tensor(x2_test_full, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # ---------- 网络 & 优化器 ----------
    net = MyNetwork().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_base)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)   # 指数阶梯衰减
    loss_fn = nn.CrossEntropyLoss()   # PyTorch 期望目标为 Long 类型索引
    # 把 one-hot 转类别索引
    y_train_cls = torch.argmax(y_train, dim=1)
    y_test_cls  = torch.argmax(y_test,  dim=1)

    # ---------- 记录 ----------
    costs, costs_dev = [], []
    train_acc, val_acc = [], []

    seed = 1
    m = x1_train.size(0)

    # ---------- 训练 ----------
    for epoch in range(num_epochs + 1):
        net.train()
        epoch_cost = 0.0
        epoch_acc  = 0.0
        seed += 1
        minibatches = random_mini_batches(x1_train.cpu().numpy(),
                                          x2_train.cpu().numpy(),
                                          x1_train_full.cpu().numpy(),
                                          x2_train_full.cpu().numpy(),
                                          y_train.cpu().numpy(),
                                          batch_size, seed)
        num_batches = len(minibatches)

        for (mb_x1, mb_x2, mb_x1f, mb_x2f, mb_y) in minibatches:
            mb_x1  = torch.tensor(mb_x1, dtype=torch.float32).to(device)
            mb_x2  = torch.tensor(mb_x2, dtype=torch.float32).to(device)
            mb_x1f = torch.tensor(mb_x1f, dtype=torch.float32).to(device)
            mb_x2f = torch.tensor(mb_x2f, dtype=torch.float32).to(device)
            mb_y   = torch.tensor(mb_y, dtype=torch.float32).to(device)
            mb_y_cls = torch.argmax(mb_y, dim=1)

            optimizer.zero_grad()
            logits, x1_re, x2_re = net(mb_x1, mb_x2)

            ce_loss = loss_fn(logits, mb_y_cls)
            mse1 = torch.mean((x1_re - mb_x1f)**2)
            mse2 = torch.mean((x2_re - mb_x2f)**2)
            l2 = l2_loss(net)
            cost = ce_loss + beta_reg*l2 + 1.0*mse1 + 1.0*mse2

            cost.backward()
            optimizer.step()

            epoch_cost += cost.item() / num_batches
            preds = torch.argmax(logits, dim=1)
            epoch_acc  += (preds == mb_y_cls).float().mean().item() / num_batches

        scheduler.step()

        # ---------- 验证 ----------
        net.eval()
        with torch.no_grad():
            logits_dev, x1d, x2d = net(x1_test, x2_test)
            ce_dev = loss_fn(logits_dev, y_test_cls)
            mse1_dev = torch.mean((x1d - x1_test_full)**2)
            mse2_dev = torch.mean((x2d - x2_test_full)**2)
            l2_dev   = l2_loss(net)
            cost_dev = ce_dev + beta_reg*l2_dev + 1.0*mse1_dev + 1.0*mse2_dev

            preds_dev = torch.argmax(logits_dev, dim=1)
            acc_dev   = (preds_dev == y_test_cls).float().mean().item()

        if print_cost and epoch % 50 == 0:
            print(f"epoch {epoch}: "
                  f"Train_loss: {epoch_cost:.4f}, Val_loss: {cost_dev.item():.4f}, "
                  f"Train_acc: {epoch_acc:.4f}, Val_acc: {acc_dev:.4f}")

        if epoch % 5 == 0:
            costs.append(epoch_cost)
            costs_dev.append(cost_dev.item())
            train_acc.append(epoch_acc)
            val_acc.append(acc_dev)

    # ---------- 画图 ----------
    plt.plot(costs, label='train')
    plt.plot(costs_dev, label='val')
    plt.ylabel('cost'); plt.xlabel('epoch (/5)'); plt.legend(); plt.show()

    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val')
    plt.ylabel('accuracy'); plt.xlabel('epoch (/5)'); plt.legend(); plt.show()

    # ---------- 返回 ----------
    # 提取参数到 dict（与 TF 版接口一致）
    state_dict = net.state_dict()
    parameters = {k: v.cpu().numpy() for k, v in state_dict.items()}

    # joint 特征 (转置后形状 [15, N]，与 TF 版相同)
    net.eval()
    with torch.no_grad():
        feature = net(x1_test, x2_test)[0].T.cpu().numpy()

    return parameters, val_acc, feature