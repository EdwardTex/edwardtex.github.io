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


### 损失函数

- 



 