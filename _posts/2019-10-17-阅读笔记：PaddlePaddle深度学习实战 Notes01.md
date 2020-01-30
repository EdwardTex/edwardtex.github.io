---
layout:     post
title:      阅读笔记：PaddlePaddle深度学习实战 Notes01
subtitle:   读书笔记第一篇
date:       2019-10-17
author:     Tex
header-img: img/post-bg-paddle.png
catalog: true
tags:
    - 深度学习 (Deep Learning)
    - 机器学习 (Machine Learning)
---
## 数学基础与Python库


### The Zen of Python

 ```
import this
 ```

### Learnt Basis
The basis of linear algebra in undergraduate, including the calculating of vectors/matrices; and the basis of calculus including derivative, chain rule, etc.

### Derivatives of vectors/matrices

#### 标量对向量求导
函数对向量求导数，其结果为函数对向量的各个分量求偏导；

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx_1%7D%5C%5C%20%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx_p%7D%20%5Cend%7Bbmatrix%7D%5Cin%20%5Cmathbf%7BR%5Ep%7D)

#### 向量对向量求导
由标量的函数构成的向量![](https://latex.codecogs.com/gif.latex?f)对于向量![](https://latex.codecogs.com/gif.latex?X)求导，其结果为一个矩阵，矩阵的第n行为函数向量![](https://latex.codecogs.com/gif.latex?f)中的每个函数对![](https://latex.codecogs.com/gif.latex?x)求偏导；

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7B%5Cpartial%20x%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cldots%20%26%20%5Cfrac%7B%5Cpartial%20f_q%7D%7B%5Cpartial%20x_1%7D%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_p%7D%20%26%20%5Cldots%20%26%20%5Cfrac%7B%5Cpartial%20f_q%7D%7B%5Cpartial%20x_p%7D%5C%5C%20%5Cend%7Bbmatrix%7D%5Cin%20R%5E%7Bp%5Ctimes%20q%7D)

#### 向量的链式法则
标量的链式法则

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20x_1%7D%3D%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20f_1%7D%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D&plus;%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20f_2%7D%5Cfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_1%7D)

而当![](https://latex.codecogs.com/gif.latex?f_1)和![](https://latex.codecogs.com/gif.latex?f_2)均为向量函数时，适应性链式法则如下

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cvec%7Bg%7D%7D%7B%5Cpartial%20x_1%7D%3DJ%5Cvec%7Bf_1%7D%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_1%7D&plus;J%5Cvec%7Bf_2%7D%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_2%7D%7D%7B%5Cpartial%20x_1%7D)

其中![](https://latex.codecogs.com/gif.latex?J%20%5Cvec%7Bf_1%7D)为Jacobi矩阵，定义如下

![](https://latex.codecogs.com/gif.latex?J%5Cvec%7Bf_1%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_n%7D%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_m%7D%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_m%7D%7D%7B%5Cpartial%20x_n%7D%20%5Cend%7Bbmatrix%7D%28i%3D1%2C2%2C%5Cldots%2Cm%29)

*链式法则中的乘法均为Hadamard Product (elementwise)
    
**链式法则是深度学习中最为常用的求导规则，常用于逆向传播算法训练神经网络的工作**

#### 常见矩阵导数计算与神经网络

- 向量对于其本身的导数为单位向量，即一个神经元如果受到自身变化的影响，那么其自身变化多少，影响的大小就有多少：

![](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20%5Cvec%20x%7D%7B%5Cpartial%20%5Cvec%20x%7D%3DI)

- 设向量![](https://latex.codecogs.com/gif.latex?%5Cvec%20w)和![](https://latex.codecogs.com/gif.latex?%5Cvec%20x)的乘积为![](https://latex.codecogs.com/gif.latex?%5Cvec%20z)，那么![](https://latex.codecogs.com/gif.latex?%5Cvec%20z)对于![](https://latex.codecogs.com/gif.latex?%5Cvec%20w)求偏导的结果为![](https://latex.codecogs.com/gif.latex?%5Cvec%20x)的转置：

![](https://latex.codecogs.com/gif.latex?%5Cvec%20z%20%3D%20%5Cvec%20w%20%5Cvec%20x)

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cvec%20z%7D%7B%5Cpartial%20%5Cvec%20w%7D%3D%7B%5Cvec%20x%7D%5ET)

扩展到矩阵同理

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Z%7D%7B%5Cpartial%20W%7D%3DX%5ET)

即 如果在神经网络中我们知道了神经元的输出结果和系数矩阵，就能反推得到输入；

- 矩阵![](https://latex.codecogs.com/gif.latex?A)和向量![](https://latex.codecogs.com/gif.latex?%5Cvec%20x)的乘积![](https://latex.codecogs.com/gif.latex?A%5Cvec%20x)对![](https://latex.codecogs.com/gif.latex?%5Cvec%20x)求偏导，其结果为![](https://latex.codecogs.com/gif.latex?A%5ET)

![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20A%5Cvec%20x%7D%7B%5Cpartial%20%5Cvec%20x%7D%3DA%5ET)

即 如果后一个神经元收到前一个神经元的影响是![](https://latex.codecogs.com/gif.latex?A%5Cvec%20x)，那么直接相连的前一个神经元增减一个单位时，后一个神经元相应地增减![](https://latex.codecogs.com/gif.latex?A%5ET)个单位；

- 向量![](https://latex.codecogs.com/gif.latex?%7B%5Cvec%20x%7D%5ET)与矩阵![](https://latex.codecogs.com/gif.latex?A)的乘积![](https://latex.codecogs.com/gif.latex?%7B%5Cvec%20x%7D%5ETA)对求偏导，其结果为![](https://latex.codecogs.com/gif.latex?A)本身

即 如果后一个神经元收到前一个神经元的影响是![](https://latex.codecogs.com/gif.latex?%7B%5Cvec%20x%7D%5ETA)，那么直接相连的前一个神经元增减一个单位时，后一个神经元相应地增减![](https://latex.codecogs.com/gif.latex?A)个单位； 

#### 梯度
梯度的本质是一个向量，用于描述某个函数在某一点处的方向导数沿该向量方向取得最大值，即函数在该点处沿该方向变化最快（变化率最大，值为梯度的模）；

在机器学习和深度学习中使用梯度下降法用以求解损失函数的最小值。

### Numpy (Numerical Pythono Extension)
#### 广播机制 (Broadcasting)
在array计算中如果一个array的维度和另一个array的子维度一致，则在没有对齐的维度上分别执行对位运算；广播机制使计算表达式保持简洁。

#### 向量化的重要性
在numpy中的array运算通过向量化实现，这对计算速度的提升是非常明显的（约为非向量化实现的500倍），这对于长时间的深度学习训练，可以节省大量时间。

更多内容可参考[官方文档](http://numpy.org "http://numpy.org")

### Matplotlib
主要用于可视化和图像处理，更多内容可参考[官方文档](http://matplotlib.org "http://matplotlib.org")