# coding=utf-8
import math
import time

import pandas as pd
from collections import Counter
import random
import torch
import torch.nn as nn
from bert_chinese_encode import get_bert_encode_for_single
from RNN_MODEL import RNN


train_data = pd.read_csv('../resources/doctor_data/train_data.csv', header=None, encoding='utf-8', sep='\t')

# 打印正负标签比例
# print(dict(Counter(train_data[0].values)))


# 转换数据到列表形式
train_data = train_data.values.tolist()
# print(train_data[:10])

# 第一步 构建随机选取数据函数
def randomTrainExample(train_data):
    """
    功能: 随机选择数据 函数

    :param train_data:  训练集 列表形式的数据
    :return:
    """
    # 从train_data中随机选择一条数据
    category, line = random.choice(train_data)
    # 将里面的文字使用bert进行编码 获取编码后的tensor类型的数据
    line_tensor = get_bert_encode_for_single(line)
    # 将分类标签封装成tensor
    catetory_tensor = torch.tensor([int(category)])
    # 返回四个结果
    return category, line, catetory_tensor, line_tensor


# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainExample(train_data)
#     print('category:', category, '/ line:', line, '/ category_tensor:', category_tensor, '/ line_tensor:', line_tensor)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 构建模型训练函数
# 选取损失函数为NullLoss()
criterion = nn.NLLLoss().to(device)
# 学习率 设置为0.005
learning_rate = 0.005

# 定义RNN模型参数
input_size = 768
hidden_size = 128
output_categories = 2



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# rnn = RNN(input_size, hidden_size, output_categories).to(device)
rnn = RNN(input_size, hidden_size, output_categories).to(device)


def train(category_tensor, line_tensor):
    """
    模型训练函数
    :param category_tensor: 代表类别张量
    :param line_tensor: 代表编码后的文本张量
    :return:
    """
    # 初始化隐藏层
    hidden = rnn.initHidden().to(device)

    # 梯度归零
    rnn.zero_grad()
    # 遍历line_tensor中每一个字的张量表示
    for i in range(line_tensor.size()[0]):
        # 然后将其输入到RNN模型中，因为模型要求输入的必须是二维张量，因此需要拓展一个维度，循环调用RNN知道最后结束
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)

    # 根据损失函数计算损失 输入: rnn的输出 和 真正的类别标签
    loss = criterion(output, category_tensor)
    # 将误差进行反向传播
    loss.backward()

    # 更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量表示和参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(-learning_rate, p.grad.data)
    # 返回结果和损失函数的值
    return output, loss.item()


def valid(category_tensor, line_tensor):
    """
    功能: 模型验证函数
    :param category_tensor:
    :param line_tensor:
    :return:
    """
    # 初始化隐藏层
    hidden = rnn.initHidden()
    # 验证不需要计算梯度
    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
        # 获取损失
        loss = criterion(output, category_tensor)

    return output, loss.item()


def timeSince(since):
    """
    功能: 辅助函数
    :param since: 获得每次打印的训练耗时， since是训练开始的时间
    :return:
    """
    # 获得当前时间
    now = time.time()
    # 获取时间差 就是训练的耗时
    s = now - since
    # 将秒转换成分钟 并取整
    m = math.floor(s / 60)
    # 计算剩下不够1分钟的秒数
    s -= m * 60
    return '%dm %ds'%(m, s)


# 第4步： 调用训练和验证函数
# 设置迭代次数为50000步
n_iters = 50000
# n_iters = 1000
# 设置打印间隔为1000步
plot_every = 1000
# plot_every = 100
# 初始化打印间隔中训练和验证的损失和准确率
train_current_loss = 0
train_current_acc = 0
valid_current_loss = 0
valid_current_acc = 0

# 初始化每次打印间隔的平均损失和准确率
all_train_losses = []
all_train_acc = []
all_valid_losses = []
all_valid_acc = []

# 获取开始时间戳
start = time.time()

for iter in range(1, n_iters + 1):
    # 调用两次随机函数分别生成一条训练数据和验证数据
    category, line, category_tensor, line_tensor = randomTrainExample(train_data)
    category_, line_, category_tensor_, line_tensor_ = randomTrainExample(train_data)
    # 分别调用训练函数和验证函数将上述两条数据传入进去
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    category_tensor_ = category_tensor_.to(device)
    line_tensor_ = line_tensor_.to(device)

    train_output, train_loss = train(category_tensor, line_tensor)
    valid_output, valid_loss = train(category_tensor_, line_tensor_)
    # 训练损失，验证损失， 训练准确率，验证准确率分别进行累加
    train_current_loss += train_loss
    train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
    valid_current_loss += valid_loss
    valid_current_acc += (valid_output.argmax(1) == category_tensor_).sum().item()
    # 当迭代次数是指定打印次数的整数倍时
    if iter % plot_every == 0:
        # 计算当前这一轮（1000步）的平均损失和平均准确率
        train_average_loss = train_current_loss / plot_every
        train_average_acc = train_current_acc / plot_every
        valid_average_loss = valid_current_loss / plot_every
        valid_average_acc = valid_current_acc / plot_every
        # 打印迭代步。 耗时， 训练损失和准确率，验证损失和准确率
        print('Iter:', iter, '|', 'TimeSince:', timeSince(since=start))
        print('Train Loss:', train_average_loss, '|', 'Train Acc:', train_average_acc)
        print('Valid Loss:', valid_average_loss, '|', 'Valid Acc:', valid_average_acc)
        # 将计算得到的结果保存到平均数列表中 方便后续绘图使用
        all_train_losses.append(train_average_loss)
        all_train_acc.append(train_average_acc)
        all_valid_losses.append(valid_average_loss)
        all_valid_acc.append(valid_average_acc)
        # 将结果清0
        train_current_loss = 0
        train_current_acc = 0
        valid_current_loss = 0
        valid_current_acc = 0