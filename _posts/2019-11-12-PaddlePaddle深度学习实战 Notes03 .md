---
layout:     post
title:      PaddlePaddle深度学习实战 Notes03
subtitle:   读书笔记第三篇
date:       2019-11-12
author:     Tex
header-img: img/post-bg-paddle.png
catalog: true
tags:
    - 深度学习 (Deep Learning)
    - 机器学习 (Machine Learning)
---
## 深度学习的单层网络
> 

### Logistic回归模型

#### Logistics回归概述
- Logistic回归模型常用于处理**二分类问题**；用于分析各个影响因素`(x_1,x_2,...,x_n)`与分类结果`y`之间关系的监督学习方法。
    
    -其中影响因素可以是离散值或连续值；所谓预测就是，输入一组影响因素特征向量，输出结果概率；**预测就是计算**。

- 如图所示，Logistic回归模型可看作时仅含有一个神经元的单层神经网络

    ![](https://i.loli.net/2019/11/12/BihUlE1fKT6Qxou.png)

    - 用数学语言描述为：给出一组特征向量`x={x1, x2, … , xm}`，希望得到一个预测结果`y^`，即
    
    ![](https://i.loli.net/2019/11/12/szBl9kSCvrNhwi3.png)


- 典型的深度学习的计算过程包含3个过程，**前向传播(Forward Propagation)过程**、**后向传播（Backward Propagation）**和**梯度下降（Gradient Descent）**过程：
    - **FP过程**可以暂时把它理解为一个前向的计算过程即可；
    - **BP过程**可以简单把它理解为*层层求偏导数*的过程；
    - **GD过程**可以理解为参数沿着当前梯度相反的方向进行迭代搜索直到最小值的过程。

- **FP过程**可以想象为向量x从左向右的计算过程，先后分为线性变换和非线性变换两个部分；
    - **线性变换**：即完成线性回归工作，即将输入的特征向量进行线性组合：
    ![](https://latex.codecogs.com/gif.latex?z%3D%20%5Cvec%20w%5ET%20%5Cvec%20x&plus;b)
    - **非线性变换**：最终的期望输出是概率`y^`,取值范围应为`[0,1]`，但第一部分的输出为`z`，是一个范围未知的实数值，就需要将该实数值转换为概率值，我们需要一个非线性函数`g(z)`来做到
    
     ![](https://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D%20%3D%20g%28z%29%20%3D%20g%28%5Cvec%20w%5ET%20%5Cvec%20x%20&plus;b%29)
    
    在深度学习范围内非线性函数`g(z)`被称作**激活函数（Activation Function）**。在本Logistic回归中激活函数具体使用`Sigmoid()`函数。

    - **Sigmoid()函数**：主要作用就是把某实数映射到区间`[0,1]`内

    ![](https://i.loli.net/2019/11/12/3Z4fWwH8r5GAEdD.png)

    ![](https://i.loli.net/2019/11/12/EjzgyFwJ4h78Qfd.png)

- Logistic回归模型的工作重点与所有深度学习模型的工作重点一样，就在于训练一组最优参数值w和b。这组最合适的w和b使得预测结果更加精确。那么怎么才能找到这样的参数呢？这就需要定义一个**损失函数**，通过对这个损失函数的不断优化来最终训练出最优的w和b。 


#### 损失函数

- **损失函数(Loss Function / Error Function)**用于对参数w和b进行优化，而损失函数的选择需要具体问题具体分析，在不同问题场景下采用不同的函数。
    - 通常情况下，会将损失函数定义为平方损失函数：
    ![](https://i.loli.net/2019/11/12/yGZUI3fQzcjHRKq.png)
    
    - 在Logistic回归模型中通常使用**对数损失函数（Logarithmic Loss Function）**作为损失函数。对数损失函数又被称作**对数似然损失函数（Log-likelihood Loss Function）**：
    ![](https://i.loli.net/2019/11/12/YjNAZPl2IRE1DVp.png)
    其中最小化损失函数的过程，也是让预测结果更精确的过程：
    ![](https://latex.codecogs.com/gif.latex?L%28%5Cwidehat%7By%7D%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29%5Crightarrow%200%20%5CRightarrow%20%5Cwidehat%7By%7D%5E%7B%28i%29%7D%20%5Crightarrow%20y%5E%7B%28i%29%7D)
    再加上其凸优化性质，使其成为适合Logistics模型的损失函数。

    - 如上选择的原因是，平方损失函数会使参数优化问题变成**非凸的**，进而可能会导致陷入局部最优；凸优化问题是指求取最小值的目标函数为凸函数，也即局部最优解就是全局最优解。

- **成本函数（Cost Function）**用于针对全部训练样本的模型训练过程中，它的定义如下：

    ![](https://i.loli.net/2019/11/12/UqAxL3oZvbwsKVt.png)

    相比较下，损失函数是用于衡量模型在单个训练样本上的表现情况的。


#### Logistics回归的梯度下降

- 将单层的Logistic回归的基本示意图中的更多细节展示出来绘制成新图就可以得到如下计算图
    ![](https://i.loli.net/2019/11/12/xtTwZQyHpSX7eBs.png)
观察到系统的输入值由两部分组成，样本的特征向量和算法参数。
样本的特征向量为`x`，参数包含权重向量`w`和偏置`b`。将这些数据进行两步运算，线性变换和非线性变换。首先是线性变换生成中间值`z`，然后经过的非线性变换得到预测值。最后将预测值`a`和真实值传给函数损失函数`L`，求得二者的差值。

    **为了和`y`作区分，因此用`a`代替表示预测值。*
    
- 梯度下降中`w`的更新公式可表示如下：

    ![](https://latex.codecogs.com/gif.latex?w%20%3D%20w%20-%20%5Calpha%20%5Cfrac%7BdL%28w%29%7D%7Bdw%7D%20%5Crightarrow%20w%20%3D%20w%20-%20%5Calpha%20dw)
    
    其中
    
    ![](https://latex.codecogs.com/gif.latex?dw%20%3D%20%5Cfrac%7BdL%28w%29%7D%7Bdw%7D)

    求解过程可拆分为顺序计算`da`、`dz`、`dw`，其中记住`dz`的计算结果可以简化许多梯度计算的步骤：

    ![](https://latex.codecogs.com/gif.latex?dz%20%3D%20a-y)
    
    ![](https://latex.codecogs.com/gif.latex?dw%20%3D%20wdz)
    
    ![](https://latex.codecogs.com/gif.latex?db%20%3D%20dz)
    
    按照如上步骤即可完成单个训练样本的梯度下降更新。

- 多个训练样本中，梯度下降的计算变成了**各样本的参数梯度值做求和平均**，多次迭代更新后，各参数将逼近全局最优解。

    ![](https://latex.codecogs.com/gif.latex?dw%20%3D%20%5Cfrac%7BdJ%28w%2Cb%29%7D%7Bdw%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%20%5Cfrac%7BdL%28a%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29%7D%7Bdw%7D)

    ![](https://latex.codecogs.com/gif.latex?db%20%3D%20%5Cfrac%7BdJ%28w%2Cb%29%7D%7Bdb%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%20%5Cfrac%7BdL%28a%5E%7B%28i%29%7D%2Cy%5E%7B%28i%29%7D%29%7D%7Bdb%7D)

    如此这般仍有较大的工程问题：嵌套的两个循环：外层循环用于遍历所有训练样本，内层循环用于遍历所有训练参数；需要**通过向量化来替换循环**，从而提升效率。

- Logistics的向量化
    - 首先，消除遍历所有参数的循环，以如下矩阵计算完成
    
        ![](https://latex.codecogs.com/gif.latex?dw%20&plus;%3D%20x%5E%7B%28i%29%7Ddz%5E%7B%28i%29%7D)    

        充分利用了*GPU的并行计算能力*来提升效率。

    - 其次，消除遍历所有训练样本的循环，分别对线性变换过程、激活过程、偏导运算、权值`w`、偏置`b`完成向量化。
  
    ![](https://latex.codecogs.com/gif.latex?Z%3Dw%5ETX&plus;b)

    ![](https://latex.codecogs.com/gif.latex?A%3Dsigmoid%28Z%29)

    ![](https://latex.codecogs.com/gif.latex?dZ%3DA-Y)    

    ![](https://i.loli.net/2019/11/12/grU4872OwdSJiyB.png)
    
    ![](https://i.loli.net/2019/11/12/ESxJ3bIOkNq62Fs.png)

    - 对梯度`dw`和`db`做平均，python的广播机制可以完美应对。  
 


### Logistics回归模型实战

#### Train with numpy
```
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

    使用python及numpy库来实现逻辑回归识别猫案例，关键步骤如下：
    1.载入数据和预处理：load_data()
    2.初始化模型参数（Parameters）
    3.循环：
        a)	计算成本（Cost）
        b)	计算梯度（Gradient）
        c)	更新参数（Gradient Descent）
    4.计算准确度
    5.展示学习曲线plot_costs()
    6.利用模型进行预测
"""

import matplotlib.pyplot as plt
import numpy as np

import utils


def load_data():
    """
    载入数据,包括训练和测试数据

    Args:
    Return:
        X_train：原始训练数据集
        Y_train：原始训练数据标签
        X_test：原始测试数据集
        Y_test：原始测试数据标签
        classes(cat/non-cat)：分类list
        px_num:数据的像素长度
    """
    X_train, Y_train, X_test, Y_test, classes = utils.load_data_sets()

    train_num = X_train.shape[0]
    test_num = X_test.shape[0]
    px_num = X_train.shape[1]

    data_dim = px_num * px_num * 3
    X_train = X_train.reshape(train_num, data_dim).T
    X_test = X_test.reshape(test_num, data_dim).T

    X_train = X_train / 255.
    X_test = X_test / 255.

    data = [X_train, Y_train, X_test, Y_test, classes, px_num]

    return data


def sigmoid(x):
    """
    sigmoid 激活函数
    """
    return 1 / (1 + np.exp(-x))


def initialize_parameters(data_dim):
    """
    参数W和b初始化为0

    Args:
        data_dim: W向量的纬度
    Returns:
        W: (dim, 1)维向量
        b: 标量，代表偏置bias
    """
    W = np.zeros((data_dim, 1), dtype=np.float)
    b = 0.0

    return W, b


def forward_and_backward_propagate(X, Y, W, b):
    """
    计算成本Cost和梯度grads

    Args:
        W: 权重， (num_px * num_px * 3, 1)维的numpy数组
        b: 偏置bias
        X: 数据，shape为(num_px * num_px * 3, number of examples)
        Y: 数据的标签( 0 if non-cat, 1 if cat) ，shape (1, number of examples)
    Return:
        cost: 逻辑回归的损失函数
        dW: cost对参数W的梯度，形状与参数W一致
        db: cost对参数b的梯度，形状与参数b一致
    """
    m = X.shape[1]

    # 前向传播，计算成本函数
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    dZ = A - Y

    cost = np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    # 后向传播，计算梯度
    dW = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    cost = np.squeeze(cost)

    grads = {
        "dW": dW,
        "db": db
    }

    return grads, cost


def update_parameters(X, Y, W, b, learning_rate):
    """
    更新参数
    Args:
        X: 整理后的输入数据
        Y: 标签
        W: 参数W
        b: bias
        learning_rate: 学习步长
    Return：
        W：更新后的参数W
        b：更新后的bias
        cost：成本
    """
    grads, cost = forward_and_backward_propagate(X, Y, W, b)

    W = W - learning_rate * grads['dW']
    b = b - learning_rate * grads['db']

    return W, b, cost


def train(X, Y, W, b, iteration_nums, learning_rate):
    """
    训练的主过程，使用梯度下降算法优化参数W和b

    Args:
        X: 数据，shape为(num_px * num_px * 3, number of examples)
        Y: 数据的标签(0 if non-cat, 1 if cat) ，shape为 (1, number of examples)
        W: 权重， (num_px * num_px * 3, 1)维的numpy数组
        b: 偏置bias，标量
        iteration_nums: 训练的迭代次数
        learning_rate: 梯度下降的学习率，可控制收敛速度和效果

    Returns:
        params: 包含参数W和b的python字典
        costs: 保存了优化过程cost的list，可以用于输出cost变化曲线
    """
    costs = []
    for i in range(iteration_nums):
        W, b, cost = update_parameters(X, Y, W, b, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print "Iteration %d, cost %f" % (i, cost)

    params = {
        "W": W,
        "b": b
    }

    return params, costs


def predict_image(X, W, b):
    """
    用学习到的逻辑回归模型来预测图片是否为猫（1 cat or 0 non-cat）

    Args:
        X: 数据，形状为(num_px * num_px * 3, number of examples)
        W: 权重， (num_px * num_px * 3, 1)维的numpy数组
        b: 偏置bias

    Returns:
        predictions: 包含了对X数据集的所有预测结果，是一个numpy数组或向量

    """
    data_dim = X.shape[0]

    m = X.shape[1]

    predictions = []

    W = W.reshape(data_dim, 1)

    # 预测概率结果为A
    A = sigmoid(np.dot(W.T, X) + b)

    # 将连续值A转化为二分类结果0或1
    # 阈值设定为0.5即预测概率大于0.5则预测结果为1
    for i in range(m):
        if A[0, i] >= 0.5:
            predictions.append(1)
        elif A[0, i] < 0.5:
            predictions.append(0)

    return predictions


def calc_accuracy(predictions, Y):
    """
    计算train准确度
    """
    Y = np.squeeze(Y)
    right = 0
    for i in range(len(predictions)):
        if predictions[i] == Y[i]:
            right += 1
    accuracy = (right / float(len(predictions))) * 100
    return accuracy


def plot_costs(costs, learning_rate):
    """
    利用costs展示模型的学习曲线
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("learning rate =" + str(learning_rate))
    # plt.show()
    plt.savefig('costs.png')


def main():
    """
    main entry
    """
    X_train, Y_train, X_test, Y_test, classes, px_num = load_data()

    iteration_nums = 2000

    learning_rate = 0.005

    data_dim = X_train.shape[0]

    W, b = initialize_parameters(data_dim)

    params, costs = train(X_train, Y_train, W, b, iteration_nums,
                          learning_rate)

    predictions_train = predict_image(X_train, params['W'], params['b'])
    predictions_test = predict_image(X_test, params['W'], params['b'])

    print "Accuracy on train set: {} %".format(calc_accuracy(predictions_train,
                                                             Y_train))
    print "Accuracy on test set: {} %".format(calc_accuracy(predictions_test,
                                                            Y_test))

    index = 15  # index(1) is a cat, index(14) not a cat
    cat_img = X_test[:, index].reshape((px_num, px_num, 3))
    plt.imshow(cat_img)
    plt.axis('off')
    plt.show()
    print "The label of this picture is " + str(Y_test[0, index]) \
          + ", 1 means it's a cat picture, 0 means not " \
          + "\nYou predict that it's a "\
          + classes[int(predictions_test[index])].decode("utf-8") \
          + " picture. \nCongrats!"

    plot_costs(costs, learning_rate)


if __name__ == "__main__":
    main()
```
#### Train with PdPd
```
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    使用paddle框架实现逻辑回归识别猫案例，关键步骤如下：
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
    8.预测infer()并输出准确率train_accuracy和test_accuracy
    9.展示学习曲线plot_costs()
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import paddle.v2 as paddle

import utils


TRAINING_SET = None
TEST_SET = None
DATA_DIM = None


def load_data():
    """
    载入数据，数据项包括：
        train_set_x_orig：原始训练数据集
        train_set_y：原始训练数据标签
        test_set_x_orig：原始测试数据集
        test_set_y：原始测试数据标签
        classes(cat/non-cat)：分类list
    Args:
    Return:
    """
    global TRAINING_SET, TEST_SET, DATA_DIM

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes \
        = utils.load_data_sets()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 定义纬度
    DATA_DIM = num_px * num_px * 3

    # 数据展开,注意此处为了方便处理，没有加上.T的转置操作
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

    # 归一化
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    TRAINING_SET = np.hstack((train_set_x, train_set_y.T))
    TEST_SET = np.hstack((test_set_x, test_set_y.T))


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
    global TRAINING_SET

    return read_data(TRAINING_SET)


def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data: 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


def network_config():
    """
    配置网络结构和设置参数
    Args:
    Return:
        y_predict: 输出层，Sigmoid作为激活函数
        cost: 损失函数
        parameters: 模型参数
        optimizer: 优化器
        feeding: 数据映射，python字典
    """
    # 输入层，paddle.layer.data表示数据层,name=’image’：名称为image,
    # type=paddle.data_type.dense_vector(DATA_DIM)：数据类型为DATADIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATA_DIM))

    # 输出层，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=image, size=1, act=paddle.activation.Sigmoid())

    # 标签数据，paddle.layer.data表示数据层，name=’label’：名称为label
    # type=paddle.data_type.dense_vector(1)：数据类型为1维稠密向量
    y_label = paddle.layer.data(
        name='label', type=paddle.data_type.dense_vector(1))

    # 定义成本函数为交叉熵损失函数multi_binary_label_cross_entropy_cost
    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=y_predict,
                                                              label=y_label)

    # 利用cost创建parameters
    parameters = paddle.parameters.create(cost)

    # 创建optimizer，并初始化momentum和learning_rate
    optimizer = paddle.optimizer.Momentum(momentum=0, learning_rate=0.00002)

    # 数据层和数组索引映射，用于trainer训练时喂数据
    feeding = {
        'image': 0,
        'label': 1}

    result = [y_predict, cost, parameters, optimizer, feeding]

    return result


def get_data(data_creator):
    """
    使用参数data_creator来获取测试数据

    Args:
        data_creator: 数据来源,可以是train()或者test()
    Return:
        result: 包含测试数据(image)和标签(label)的python字典
    """
    data_creator = data_creator
    data_image = []
    data_label = []

    for item in data_creator():
        data_image.append((item[0],))
        data_label.append(item[1])

    result = {
        "image": data_image,
        "label": data_label
    }

    return result


def calc_accuracy(probs, data):
    """
    根据数据集来计算准确度accuracy

    Args:
        probs: 数据集的预测结果，调用paddle.infer()来获取
        data: 数据集

    Return:
        calc_accuracy: 训练准确度
    """
    right = 0
    total = len(data['label'])
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5 and data['label'][i] == 1:
            right += 1
        elif float(probs[i][0]) < 0.5 and data['label'][i] == 0:
            right += 1
    accuracy = (float(right) / float(total)) * 100
    return accuracy


def infer(y_predict, parameters):
    """
    预测并输出模型准确率

    Args:
        y_predict: 输出层，DATADIM维稠密向量
        parameters: 训练完成的模型参数

    Return:
    """
    # 获取测试数据和训练数据，用来验证模型准确度
    train_data = get_data(train())
    test_data = get_data(test())

    # 根据train_data和test_data预测结果，output_layer表示输出层，parameters表示模型参数，input表示输入的测试数据
    probs_train = paddle.infer(
        output_layer=y_predict, parameters=parameters,
        input=train_data['image']
    )
    probs_test = paddle.infer(
        output_layer=y_predict, parameters=parameters,
        input=test_data['image']
    )

    # 计算train_accuracy和test_accuracy
    print "train_accuracy: {} %".format(calc_accuracy(probs_train, train_data))
    print "test_accuracy: {} %".format(calc_accuracy(probs_test, test_data))


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
    plt.title("Learning rate = 0.00002")
    # plt.show()
    plt.savefig('costs.png')


def main():
    """
    main entry, 定义神经网络结构，训练、预测、检验准确率并打印学习曲线
    Args:
    Return:
    """
    global DATA_DIM

    # 载入数据
    load_data()

    # 初始化，设置是否使用gpu，trainer数量
    paddle.init(use_gpu=False, trainer_count=1)

    # 配置网络结构和设置参数
    y_predict, cost, parameters, optimizer, feeding = \
        network_config()

    # 记录成本cost
    costs = []

    # 处理事件
    def event_handler(event):
        """
        事件处理器，可以根据训练过程的信息作相应操作

        Args:
            event -- 事件对象，包含event.pass_id, event.batch_id, event.cost等信息
        Return:
        """
        if isinstance(event, paddle.event.EndIteration):
            if event.pass_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
                costs.append(event.cost)
                with open('params_pass_%d.tar' % event.pass_id, 'w') as para_f:
                    parameters.to_tar(para_f)

    # 构造trainer,配置三个参数cost、parameters、update_equation，它们分别表示成本函数、参数和更新公式。
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    # 模型训练
    # paddle.reader.shuffle(train(), buf_size=5000)：
    # 表示trainer从train()这个reader中读取了buf_size=5000大小的数据并打乱顺序
    # paddle.batch(reader(), batch_size=256)：
    # 表示从打乱的数据中再取出batch_size=256大小的数据进行一次迭代训练
    # feeding：用到了之前定义的feeding索引，将数据层image和label输入trainer
    # event_handler：事件管理机制，可以自定义event_handler，根据事件信息作相应的操作
    # num_passes：定义训练的迭代次数
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(train(), buf_size=5000),
            batch_size=256),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=5000)

    # 预测
    infer(y_predict, parameters)

    # 展示学习曲线
    plot_costs(costs)


if __name__ == '__main__':
    main()

```

#### Predict with PdPd
```
#!/usr/bin/env python
#  -*- coding:utf-8 -*-

"""

    用学习到的模型进行预测
    与train-with-paddle.py不同，这里不需要重新训练模型，只需要加载训练生成的parameters.tar
    文件来获取模型参数，对这组参数也就是训练完的模型进行检测。
    1.载入数据和预处理：load_data()
    2.从parameters.tar文件直接获取模型参数
    3.初始化
    4.配置网络结构
    5.获取测试数据
    6.根据测试数据获得预测结果
    7.将预测结果转化为二分类结果
    8.预测图片是否为猫
"""

import numpy as np
import paddle.v2 as paddle

from utils import load_data_sets

TEST_SET = None
PARAMETERS = None
DATA_DIM = None
CLASSES = None


def load_data():
    """
    载入数据，数据项包括：
        train_set_x_orig：原始训练数据集
        train_set_y：原始训练数据标签
        test_set_x_orig：原始测试数据集
        test_set_y：原始测试数据标签
        classes(cat/non-cat)：分类list

    Args:
    Return:
    """
    global TEST_SET, DATA_DIM, CLASSES

    train_x_ori, train_y, test_x_ori, test_y, classes = \
        load_data_sets()
    m_test = test_x_ori.shape[0]
    num_px = train_x_ori.shape[1]

    # 定义纬度
    DATA_DIM = num_px * num_px * 3

    # 展开数据
    test_x_flatten = test_x_ori.reshape(m_test, -1)

    # 归一化数据
    test_x = test_x_flatten / 255.

    TEST_SET = np.hstack((test_x, test_y.T))

    CLASSES = classes


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


def test():
    """
    定义一个reader来获取测试数据集及其标签

    Args:
    Return:
        read_data: 用于获取测试数据集及其标签的reader
    """
    global TEST_SET

    return read_data(TEST_SET)


def get_data(data_creator):
    """
    获取data，服务于get_train_data()和get_test_data()

    Args:
        data_creator: 数据来源,可以是train()或者test()
    Return:
        result: 包含测试数据(image)和标签(label)的python字典
    """
    data_creator = data_creator
    data_image = []
    data_label = []

    for item in data_creator():
        data_image.append((item[0],))
        data_label.append(item[1])

    result = {
        "image": data_image,
        "label": data_label
    }

    return result


def get_binary_result(probs):
    """
    将预测结果转化为二分类结果
    Args:
        probs: 预测结果
    Return:
        binary_result: 二分类结果
    """
    binary_result = []
    for i in range(len(probs)):
        if float(probs[i][0]) > 0.5:
            binary_result.append(1)
        elif float(probs[i][0]) < 0.5:
            binary_result.append(0)
    return binary_result


def network_config():
    """
    配置网络结构和设置参数
    Args:
    Return:
        y_predict: 输出层，Sigmoid作为激活函数
    """
    # 输入层，paddle.layer.data表示数据层
    # name=’image’：名称为image
    # type=paddle.data_type.dense_vector(DATA_DIM)：数据类型为DATA_DIM维稠密向量
    image = paddle.layer.data(
        name='image', type=paddle.data_type.dense_vector(DATA_DIM))

    # 输出层，paddle.layer.fc表示全连接层，input=image: 该层输入数据为image
    # size=1：神经元个数，act=paddle.activation.Sigmoid()：激活函数为Sigmoid()
    y_predict = paddle.layer.fc(
        input=image, size=1, act=paddle.activation.Sigmoid())

    return y_predict


def main():
    """
    main entry 预测结果并检验模型准确率
    Args:
    Return:
    """
    global PARAMETERS

    # 载入数据
    load_data()

    # 载入参数
    with open('params_pass_1900.tar', 'r') as param_f:
        PARAMETERS = paddle.parameters.Parameters.from_tar(param_f)

    # 初始化
    paddle.init(use_gpu=False, trainer_count=1)

    # 配置网络结构
    y_predict = network_config()

    # 获取测试数据
    test_data = get_data(test())

    # 根据test_data预测结果
    probs = paddle.infer(
        output_layer=y_predict, parameters=PARAMETERS, input=test_data['image']
    )

    # 将结果转化为二分类结果
    binary_result = get_binary_result(probs)

    # 预测图片是否为猫
    index = 14
    print ("y = " + str(binary_result[index]) +
           ", you predicted that it is a \"" +
           CLASSES[binary_result[index]].decode("utf-8") + "\" picture.")


if __name__ == "__main__":
    main()

```
      
#### 成本变化曲线

![](https://i.loli.net/2019/11/14/ApFa3Kgd5CQz8u6.png)

如图所示，成本再刚开始时收敛较快，随着迭代次数变多，收敛速度变慢，最终收敛到一个较小值；之后会通过调整学习率/迭代次数来改变模型的学习效果。