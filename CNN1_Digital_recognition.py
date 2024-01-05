import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms 
import matplotlib.pyplot as plt
import numpy as np


# 构建数据集和训练集
# 定义超参数
input_size = 28  # 图像的总尺寸28*28*1，三维的
num_classes = 10  # 标签的种类数
num_epochs = 5  # 训练的总循环周期
batch_size = 64  # 一个撮（批次）的大小，64张图片


# 训练集
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# 通过设置 root 参数为 './data'，train 参数为 True 和 False 分别表示加载训练集和测试集
# 测试集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())
# 将图像数据转换为张量形式，并且 download=True 表示如果数据集不存在，则会下载数据集。

# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
# 利用 torch.utils.data.DataLoader 创建了训练集和测试集的数据加载器
# 这些数据加载器允许以批次的方式加载数据，并且可以选择是否对数据进行随机洗牌（shuffle），在训练神经网络时，通常会对数据进行随机洗牌以增加模型的泛化能力。


# nn.Module，这是PyTorch中所有神经网络模块的基类。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小“是一开始就定的大小” (1, 28, 28) conv1不是第一个卷积层，而是第一个卷积模块，包括卷积relu池化
            nn.Conv2d(
                in_channels=1,  # 灰度图，当前输入的特征图个数
                out_channels=16,  # 要得到几多少个特征图，应用16个不同的滤波器
                kernel_size=5,  # 卷积核大小5x5
                stride=1,  # 步长
                padding=2,  # 填充，如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),  # 卷积后relu层
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)，这将减小空间维度一半。
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),
            # 参数的简单写法，与conv1对应。# 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        # 创建一个全连接（线性）层（out），输入大小为32 * 7 * 7，输出大小为10，对应于MNIST数据集中的10个类别。
        # 输出大小为10是因为对应于MNIST数据集的10个数字类别，全连接层的输出向量将提供每个类别的预测得分。最终，对应得分最高的类别即为模型对输入图像所预测的数字类别。

    def forward(self, x):  # 前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        # 这一行的x是经过conv1到con2两次卷积后的结果
        x = x.view(x.size(0), -1)
        # flatten操作，结果为：(batch_size, 32 * 7 * 7)，将卷积层的输出展平为一维张量。使用-1可以自动推断该维度的大小。
        output = self.out(x)   # 张量通过全连接层
        return output


# 准确率作为评估标准
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
# 一、使用torch.max()函数来获取预测结果中每个样本的最大值及其对应的索引，predictions.data表示预测结果的数据，而1表示在每个样本中找到最大值
# 二、[0]，是最大值的张量，而[1]，是最大值对应的索引的张量。
#    使用eq()，方法来比较pred（预测结果的索引）和labels.data.view_as(pred)
#    eq()返回一个由0和1组成的张量，表示相应位置上的元素是否相等（相等为1，不相等为0）。  接着就是求和看数量呗
# 三、这个函数的返回值表示预测的准确率，可以通过将rights除以len(labels)来计算。


# 这三行都在实例化
net = CNN()
# 这行代码实例化了一个CNN
criterion = nn.CrossEntropyLoss()
# nn.CrossEntropyLoss()表示实例化一个交叉熵损失函数对象。
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 实例化Adam优化器，用于更新模型的参数，net.parameters()表示传入模型的可学习参数，lr就是学习率
# 普通的随机梯度下降算法


# 典型的深度学习模型的训练循环，通常用于训练神经网络
# 开始训练循环
for epoch in range(num_epochs):

    train_rights = []
    # 用于存储每个 epoch 中每个 batch 的训练集准确率

    for batch_idx, (data, target) in enumerate(train_loader):
        # train_loader 是一个数据加载器，这个循环会遍历每个批次的训练数据。batch_idx 是批次的索引，data 是输入数据，target 是对应的目标标签。
        # 这个循环的目的是通过多次迭代训练神经网络，使其逐渐调整权重以降低损失，提高对训练数据的预测准确率。
        # enumerate 函数同时返回索引和元素,(data, target) 是由 DataLoader 返回的元素
        # 在深度学习训练中，模型将使用 data 进行前向传播，然后与 target 进行比较以计算损失，并进行反向传播以更新模型的权重。

        net.train()  # 将神经网络设置为训练模式
        output = net(data)                # 通过神经网络进行前向传播，得到模型的输出
        loss = criterion(output, target)  # 计算模型输出与真实标签之间的损失，使用了预先定义的损失函数 criterion
        optimizer.zero_grad()             # 梯度归零，清除之前的梯度信息，以便进行新一轮的反向传播
        loss.backward()                   # 通过调用 backward 方法，计算损失相对于模型参数的梯度。
        optimizer.step()                  # 根据计算得到的梯度，通过调用优化器的 step 方法来更新模型的权重。
        right = accuracy(output, target)
        train_rights.append(right)        # 计算当前批次的预测准确率，并将其记录在列表 train_rights 中。

        if batch_idx % 100 == 0:          # 在每训练了100个批次之后执行的

            net.eval()                    # 将神经网络切换为评估模式。在评估模式下，一些层（如批量归一化层和dropout层）的行为可能会有所不同。
            val_rights = []               # 创建一个空列表

            # 测试集测试 对测试数据进行前向传播，得到输出 output，然后计算当前批次的准确率，并将其添加到 val_rights 列表中。
            # 这有助于监控模型在验证数据上的性能，以及检查是否存在过拟合等问题。
            for (data, target) in test_loader:
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            # 准确率计算
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

            print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                epoch,                                         # format 方法将这些值插入到字符串中的相应位置，形成最终的打印信息。
                       batch_idx * batch_size,
                       len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.data,
                       100. * train_r[0].numpy() / train_r[1],
                       100. * val_r[0].numpy() / val_r[1]))





# torch.nn（用于构建神经网络模型的模块）、torch.optim（用于优化器的模块）、以及torchvision（用于计算机视觉任务的工具库）等。