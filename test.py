import numpy as np
import pandas as pda
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_num_threads(8)

# 打开数据文件并读取数据
dataFileName = "data.csv"
dataFile = open(dataFileName, "r")
fileData = pda.read_csv(dataFile)
dataFile.close()

# 数据预处理
numpyOriginalData = np.array(fileData.loc[:, :], dtype=np.float32)
xNumpyData = numpyOriginalData[:, :9]
yNumpyData = numpyOriginalData[:, 9]
x = torch.tensor(xNumpyData)
y = torch.tensor(yNumpyData).unsqueeze(1)  # 转置

# 区分训练集和验证集
n_samples = x.shape[0]  # 样本总数量
n_val = int(0.15 * n_samples)  # 验证集数量
shuffled_indices = torch.randperm(n_samples)  # 随机打乱序列

train_indices = shuffled_indices[:-n_val]  # 训练集序列
val_indices = shuffled_indices[-n_val:]  # 验证集序列

train_x = x[train_indices]
train_y = y[train_indices]

val_x = x[val_indices]
val_y = y[val_indices]

# 定义循环
def training_loop(n_epochs, optimizer, model, loss_fn, train_x, val_x, train_y, val_y):
    trainLossHistory = np.array([])
    valLossHistory = np.array([])
    for epoch in range(1, n_epochs + 1):
        train_y_predicted = model(train_x)
        train_loss = loss_fn(train_y_predicted, train_y)
        trainLossHistory = np.append(trainLossHistory, train_loss.item())

        with torch.no_grad():
            val_y_predicted = model(val_x)
            val_loss = loss_fn(val_y_predicted, val_y)
            assert val_loss.requires_grad == False
        valLossHistory = np.append(valLossHistory, val_loss.item())

        optimizer.zero_grad()  # 梯度清零
        train_loss.backward()  # 反向传播
        optimizer.step()       # 参数优化

        if epoch ==1 or epoch % 2000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.6f},"
                  f"Validation loss {val_loss.item():.6f}")
    
    return [trainLossHistory, valLossHistory]

# 神经网络模型
seq_model = nn.Sequential(
    nn.Linear(9, 20),
    nn.Tanh(),
    nn.Linear(20, 40),
    nn.Tanh(),
    nn.Linear(40, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)

optimizer = optim.SGD(seq_model.parameters(), lr=0.03)

[trainLossHistory, valLossHistory] = training_loop(
    n_epochs=100000,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),   # 均方误差
    train_x=train_x,
    val_x=val_x,
    train_y=train_y,
    val_y=val_y
)

# 可视化
DPI = 512
plt.figure("Loss function", dpi=DPI)
plt.title("Epoch=100000, lr=0.03, model=9-20-40-20-1")
plt.ylabel("loss")
plt.xlabel("iterations")
trainCounts = np.linspace(0, trainLossHistory.size-1, trainLossHistory.size)
plt.plot(trainCounts, trainLossHistory , label="train_loss")
plt.plot(trainCounts, valLossHistory , label="val_loss")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("test.png")