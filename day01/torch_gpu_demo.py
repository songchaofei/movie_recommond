# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.data import DataLoader

# 加载数据集
dataset = torchvision.datasets.CIFAR10(root="data1", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# loader加载
dataloader = DataLoader(dataset, batch_size=64)


# 网络
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 第一次卷积
            MaxPool2d(2),  # 第一次最大池化
            Conv2d(32, 32, 5, padding=2),  # 第二次卷积
            MaxPool2d(2),  # 第二次最大池化
            Conv2d(32, 64, 5, padding=2),  # 第三次卷积
            MaxPool2d(2),  # 第三次最大池化
            Flatten(),    # 展平层
            Linear(1024, 64),  # 第一个全连接层
            Linear(64, 10),  # 第二个全连接层
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型和数据转移到GPU上
mynn = NN().to(device)
print(mynn)
loss = nn.CrossEntropyLoss().to(device)

# 优化器
optim = torch.optim.SGD(mynn.parameters(), lr=0.01)

# 多轮学习  0 - 20  20轮
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        # 确保数据也转移到GPU上
        imgs, targets = data[0].to(device), data[1].to(device)

        optim.zero_grad()  # 清零梯度缓存
        outputs = mynn(imgs)  # 前向传播
        loss_value = loss(outputs, targets)  # 计算损失
        loss_value.backward()  # 反向传播，计算梯度
        optim.step()  # 根据梯度更新权重

        running_loss += loss_value.item()  # 累加损失值

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    print("------------------------------")
