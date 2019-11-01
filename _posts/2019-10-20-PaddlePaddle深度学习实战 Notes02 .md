---
layout:     post
title:      PaddlePaddle深度学习实战 Notes02
subtitle:   读书笔记第二篇
date:       2019-10-20
author:     Tex
header-img: img/post-bg-paddle.png
catalog: true
tags:
    - 深度学习 (Deep Learning)
    - 机器学习 (Machine Learning)
---
## 前言&深度学习概论与PaddlePaddle入门
> 学会如何使用框架就像学会一套外功拳法，而基础知识和理论则是内功心法，了解神经网络的各个细节，深入算法性能优化的思路和技巧，这些是深度学习的精髓。

### 前/序言节选
- 以1956年的达特茅斯会议(Dartmouth Conference)为起点，人工智能经过了60余年的发展，经历了逻辑推理理论和专家系统的两次繁荣，也经历了随之而来的两次寒冬；近十年间，人工智能的技术再次出现了前所未有的爆发性增长和繁荣期；
- 本轮人工智能的快速发展离不开三大关键要素：一，大规模高性能的云端计算硬件集群是AI发展的强劲引擎；二，数据是推动AI发展的燃料，ML技术需要大量标注数据来进行模型训练，得到有用的知识从而做出决策；三，不断推陈出新的AI算法，新算法的提出/改进带来应用效果上的大幅度提升；


### 人工智能/机器学习/深度学习
- **三者的关系**：简单地描述为嵌套关系，机器学习是人工智能的子集，深度学习是机器学习的子集；人工智能是最早出现的概念，范围最广；随后出现的是机器学习，最内层是深度学习；

![](http://i2.tiimg.com/702266/7ae917f5ff2b8bb4.png)
![](https://i.imgur.com/uzqOw4b.png)

- **人工智能-定义**：计算机科学的一个分支，是一门研究机器智能的学科，即用人工的方法和技术来研制智能机器/系统，以此模仿/延申/扩展人的智能。人工智能的主要任务是建立智能信息处理理论，使计算机系统拥有近似于人类的智能行为；

- **机器学习-定义**：如果一个程序可以在任务T上，随着经验E的增加，效果P也可以随之增加，则称这个程序可以从经验中学习；

- **深度学习-定义**：基于多层神经网络的，以海量数据为输入的，规则自学习的方法；

![](http://i2.tiimg.com/702266/6ec316bee68a97e5.png)

- **深度学习-优势**：通过重复利用中间层计算单元，大大减少参数设定；通过学习深层非线性网络结构，只需简单的网络结构即可实现复杂函数的逼近，具备从大量*无标注样本集*中学习数据集*本质特征*的能力；

- **深度学习-劣势**：需要结合特定领域的先验知识，需要和其他方法才能得到最好的结果；可解释性不强，无法做出针对性的具体改进；

### 深度学习的发展历程
- **神经网络第一次高潮**：1957年感知机(Perceptron)的提出带来第一次高潮，成为日后发展神经网络和支持向量机(SVM)的基础；感知机是一种算法构造的**分类器**，是一种线性分类模型，原理是通过不断试错以期寻找一个合适的超平面把数据分开；

- **神经网络第一次寒冬**：1969年Minsky分析了单层神经网络的功能及局限，证明感知机不能解决线性不可分问题（简单如异或），直言大部分关于感知机的研究是没有价值的；自此人们试图通过增加隐层创造多层感知机（最简单的前馈神经网络），研究表明随着隐层数的增多，区域可以形成任意的形状，从而解决任何复杂的分类问题；但由于隐层的权值训练无从下手，发展走入第一次寒冬；

- **神经网络第二次高潮**：1982年**Hopfile网络**的提出解决了一些识别和约束优化的问题；1986年Hinton和Rumelhart提出了**反向传播算法(BP算法)**，随后基于BP算法的应用取得了良好的效果，走入第二次高潮；

- **神经网络第二次寒冬**：1989年LeCun基于BP算法提出了**卷积神经网络(CNN)**，但由于浅层的限制问题，神经网络中越远离输出层的参数越难训练，这就是“梯度爆炸”问题，另外受限于当时的算力，深层网络的训练需求得不到满足；此时SVM问世，这种分类算法可以通过一种“核机制”的非线性映射算法，将线性不可分的样本转化到高位特征空间中使样本可分，其应用效果远超同时期神经网络算法的表现，走入第二次寒冬；

- **深度学习崛起**：2006年Hinton提出深度信念网络(Deep Belif Nets)，立刻在效果上击败了SVM；Hinton表示：一，多层神经网络模型具备很强的特征学习能力，习得的特征数据对于原始数据有更本质的代表性，大大便于分类和可视化问题；二，深度神经网络很难训练到最优的问题可以通过逐层训练的方法解决，即将上层训练好的结果作为下层训练过程中的初始化参数。

### 常见的深度学习网络结构

- **全连接网络结构(Full Connected)**：所有的输出与输入都是相连的，参数的冗余需要相当的存储和计算空间，这使得很少会使用纯FC到复杂场景中；FC大多作为CNN的“防火墙”，当训练集和测试集有较大的差异时，保证模型良好的迁移能力；

![](http://i2.tiimg.com/702266/d2b3495a00880a71.png)

- **卷积神经网络(Convolutional Neural Network)**：CNN由一系列层构成，每层都通过一个**可微函数**将一个量转化为另一个量，上下层神经元不直接连接，通过“卷积核”作为中介，大大减少了隐层的参数；层的类型包括卷积层(Convolutional Layer)/池化层(Pooling Layer)/全连接层(FC Layer)等；CNN专用于处理类网格结构的数据，例如图像数据；

![](http://i2.tiimg.com/702266/d86a46085612a024.jpg)

- **循环（递归）神经网络(Recurrent Neural Network)**：隐层的输出再送回隐层的输入，反复地训练；用于处理序列数据；

![](http://i2.tiimg.com/702266/9b61bbaecb4551c7.png)

### 机器学习回顾

#### 线性回归模型
- 模型建立流程：获取数据>数据预处理>训练模型>应用模型；
- **假设函数(Hypothesis Function)**：描述自变量和因变量间关系的函数；
- **损失函数(Loss Function)**：衡量函数预测结果和真实值之间的误差的函数；通常会定义为平方损失函数(Quadratic Loss Function)，其他包括均方差、交叉熵等等。
- **优化算法(Optimization Algorithm)**：决定了模型的精度和运算速度；最常见的是梯度下降法，其中常用`alpha`来表示**学习率**，可以形象地理解为每次迭代时移动的步长，决定了梯度下降的速率和稳定性。

#### 数据处理
- 首先要处理的是数据类型不同的问题，如果有离散值和连续值的情况，就必须对离散值进行处理，可采用转换为二值属性或映射为多维向量；

- 然后是**归一化(Normalize)**，将各维属性的取值范围放缩到差不多的区间；必须进行归一化操作的理由如下：一，过大或过小的数值范围会导致计算时的浮点上溢或下溢；二，不同的数值范围导致不同属性对模型的重要性不同（至少在训练初期如此），而这个隐含假设通常是不合理的；如果不进行修正，会加长训练模型的时间，对优化过程造成困难；三，很多技巧/模型（eg.L1/L2正则项，向量空间模型）都基于这样的假设：所有的属性取值都几乎以0为均值且取值范围是相近的；

- 一般会将到手的数据划分为训练集和测试集；模型在两个集合上的误差分别被称为训练误差和测试误差；分割数据的比例要这样考虑：更多的训练数据会降低参数估计的方差，从而得到更可信的模型，而更多的测试数据会降低测试误差的方差，从而得到更可信的测试误差；常用比例为**8:2**；实际项目中还会划分出验证集，因为复杂模型中还有一些**超参数(Hyperparameter)**去调节，通过尝试多种超参数的组合分别训练多个模型，然后对比他们在验证集上的表现，选择相对最好的一个模型再在测试集上评估测试误差；

### 深度学习框架

- 框架可以简化计算图的搭建：**计算图(Computational Graph)**的本质是一个有向无环图，可以被用于大部分基础表达式的建模；框架中包含许多张量相关的运算，随着操作种类的增多，多个操作的中间执行关系十分复杂；通过计算图可以使网络中参数的传播过程描述得更精确；

- 框架可以简化偏导计算：模型搭建的过程中，不可避免地要计算损失函数，这就需要不停地做微分运算，而计算图正好可以通过模块化的方式表达模型的内部逻辑，进而通过遍历计算图来实现微分运算，这被称为**基于计算图的声明式求解**，2012年之后主流框架都采用了这种求解方式；

![](http://i2.tiimg.com/702266/86135d334b072af2.png)

- 框架可以高效运行：目前对于大规模的深度学习来说，巨大的数据量使得单机很难在有限时间内完成训练；这就需要集群分布式并行计算或使用多卡GPU计算，具有分布式性能的框架可以使模型训练更高效。

### PaddlePaddle实战(预测房价 From Andrew Ng)
```
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    使用PaddlePaddle来做线性回归，拟合房屋价格与房屋面积的线性关系，具体步骤如下：
    1.载入数据和预处理：load_data()
    2.定义train()和test()用于读取训练数据和测试数据，分别返回一个reader
    3.初始化
    4.配置网络结构和设置参数：
        - 定义成本函数cost
        - 创建parameters
        - 定义优化器optimizer
    5.定义event_handler
    6.定义trainer
    7.开始训练
    8.打印参数和结果print_parameters()
    9.展示学习曲线plot_costs()
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import paddle.v2 as paddle


TRAIN_DATA = None
X_RAW = None
TEST_DATA = None


def load_data(filename, feature_num=2, ratio=0.8):
    """
    载入数据并进行数据预处理
    Args:
        filename: 数据存储文件，从该文件读取数据
        feature_num: 数据特征数量
        ratio: 训练集占总数据集比例
    Return:
    """
    global TRAIN_DATA, TEST_DATA, X_RAW
    # data = np.loadtxt()表示将数据载入后以矩阵或向量的形式存储在data中
    # delimiter=',' 表示以','为分隔符
    data = np.loadtxt(filename, delimiter=',')
    X_RAW = data.T[0].copy()
    # axis=0 表示按列计算
    # data.shape[0]表示data中一共多少列
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]

    # 归一化，data[:, i] 表示第i列的元素
    for i in xrange(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # offset用于划分训练数据集和测试数据集，例如0.8表示训练集占80%
    offset = int(data.shape[0] * ratio)
    TRAIN_DATA = data[:offset].copy()
    TEST_DATA = data[offset:].copy()


def read_data(data_set):
    """
    读取训练数据或测试数据，服务于train()和test()
    Args:
        data_set: 要获取的数据集
    Return:
        reader: 用于获取训练数据集及其标签的生成器generator
    """
    def reader():
        """
        一个reader
        Args:
        Return:
            data[:-1], data[-1:] -- 使用yield返回生成器(generator)，
            data[:-1]表示前n-1个元素，也就是训练数据，data[-1:]表示最后一个元素，也就是对应的标签
        """
        for data in data_set:
            yield data[:-1], data[-1:]
    return reader


def train():
    """
    定义一个reader来获取训练数据集及其标签
    Args:
    Return:
        read_data: 用于获取训练数据集及其标签的reader
    """
    global TRAIN_DATA
    load_data('data.txt')
    return read_data(TRAIN_DATA)


def test():
    """
    定义一个reader来获取测试数据集及其标签
    Args:
    Return:
        read_data: 用于获取测试数据集及其标签的reader
    """
    global TEST_DATA
    load_data('data.txt')
    return read_data(TEST_DATA)


def network_config():
    """
    配置网络结构
    Args:
    Return:
        cost: 损失函数
        parameters: 模型参数
        optimizer: 优化器
        feeding: 数据映射，python字典
    """
    # 输入层，paddle.layer.data表示数据层,name=’x’：名称为x_input,
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    x_input = paddle.layer.data(name='x',
                                type=paddle.data_type.dense_vector(1))

    # 输出层，paddle.layer.fc表示全连接层，input=x: 该层输入数据为x
    # size=1：神经元个数，act=paddle.activation.Linear()：激活函数为Linear()
    y_predict = paddle.layer.fc(input=x_input, size=1,
                                act=paddle.activation.Linear())

    # 标签数据，paddle.layer.data表示数据层，name=’y’：名称为output_y
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y_label = paddle.layer.data(name='y',
                                type=paddle.data_type.dense_vector(1))

    # 定义成本函数为均方差损失函数square_error_cost
    cost = paddle.layer.square_error_cost(input=y_predict, label=y_label)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum
    optimizer = paddle.optimizer.Momentum(momentum=0)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {'x': 0, 'y': 1}

    result = [cost, parameters, optimizer, feeding]

    return result


def plot_costs(costs):
    """
    利用costs展示模型的训练曲线
    Args:
        costs: 记录了训练过程的cost变化的list，每一百次迭代记录一次
    Return:
    """
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("House Price Distributions")
    plt.show()
    plt.savefig('costs.png')


def print_parameters(parameters):
    """
    打印训练结果的参数以及测试结果
    Args:
        parameters: 训练结果的参数
    Return:
    """
    print "Result Parameters as below:"
    theta_a = parameters.get('___fc_layer_0__.w0')[0]
    theta_b = parameters.get('___fc_layer_0__.wbias')[0]

    x_0 = X_RAW[0]
    y_0 = theta_a * TRAIN_DATA[0][0] + theta_b

    x_1 = X_RAW[1]
    y_1 = theta_a * TRAIN_DATA[1][0] + theta_b

    param_a = (y_0 - y_1) / (x_0 - x_1)
    param_b = (y_1 - param_a * x_1)

    print 'a = ', param_a
    print 'b = ', param_b


def main():
    """
    程序入口，完成初始化，定义神经网络结构，训练，打印等逻辑
    Args:
    Return:
    """
    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=False, trainer_count=1)

    # 配置网络结构和设置参数
    cost, parameters, optimizer, feeding = network_config()

    # 记录成本cost
    costs = []

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式。
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # 处理事件
    def event_handler(event):
        """
        事件处理器，可以根据训练过程的信息作相应操作
        Args:
            event: 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
                costs.append(event.cost)

        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(test(), batch_size=2),
                feeding=feeding)
            print "Test %d, Cost %f" % (event.pass_id, result.cost)

    # 模型训练

    # paddle.reader.shuffle(train(), buf_size=500)：
    # 表示trainer从train()这个reader中读取了buf_size=500大小的数据并打乱顺序
    # paddle.batch(reader(), batch_size=256):
    # 表示从打乱的数据中再取出batch_size=256大小的数据进行一次迭代训练
    # feeding：用到了之前定义的feeding索引，将数据层x和y输入trainer
    # event_handler：事件管理机制，可以自定义event_handler，根据事件信息作相应的操作
    # num_passes：定义训练的迭代次数

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=500),
            batch_size=256),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=300)

    # 打印参数结果
    print_parameters(parameters)

    # 展示学习曲线
    plot_costs(costs)


if __name__ == '__main__':
    main()

```

![](http://i2.tiimg.com/702266/e913e21e54b1578f.png)
![](http://i2.tiimg.com/702266/3809e05b0ce2a36a.png)