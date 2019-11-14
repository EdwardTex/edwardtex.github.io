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

 
      