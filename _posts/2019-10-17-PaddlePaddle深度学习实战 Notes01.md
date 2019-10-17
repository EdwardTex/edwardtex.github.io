---
layout:     post
title:      PaddlePaddle深度学习实战 Notes01
subtitle:   读书笔记第一篇
date:       2019-10-17
author:     Tex
header-img: img/post-bg-paddle.png
catalog: true
tags:
    - 深度学习 (Deep Learning)
    - 机器学习 (Machine Learning)
---
## Deep Learning By PaddlePaddle
### 数学基础与Python库

1. The Zen of Python

    ```
import this
    ```

2. The basis of linear algebra in undergraduate, including the calculating of vectors/matrices; and the basis of calculus including derivative, chain rule, etc.
3. **Not Learned Before:**derivatives of vectors/matrices

    3-1 标量对向量求导

    函数对向量求导数，其结果为函数对向量的各个分量求偏导；

    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx_1%7D%5C%5C%20%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7Bx_p%7D%20%5Cend%7Bbmatrix%7D%5Cin%20%5Cmathbf%7BR%5Ep%7D)

    3-2 向量对向量求导
    
    由标量的函数构成的向量`f`对于向量`X`求导，其结果为一个矩阵，矩阵的第n行为函数向量`f`中的每个函数对`x`求偏导；
   ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20f%28x%29%7D%7B%5Cpartial%20x%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Cldots%20%26%20%5Cfrac%7B%5Cpartial%20f_q%7D%7B%5Cpartial%20x_1%7D%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_p%7D%20%26%20%5Cldots%20%26%20%5Cfrac%7B%5Cpartial%20f_q%7D%7B%5Cpartial%20x_p%7D%5C%5C%20%5Cend%7Bbmatrix%7D%5Cin%20R%5E%7Bp%5Ctimes%20q%7D)

    3-3 向量的链式法则

    标量的链式法则

    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20x_1%7D%3D%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20f_1%7D%5Cfrac%7B%5Cpartial%20f_1%7D%7B%5Cpartial%20x_1%7D&plus;%5Cfrac%7B%5Cpartial%20g%7D%7B%5Cpartial%20f_2%7D%5Cfrac%7B%5Cpartial%20f_2%7D%7B%5Cpartial%20x_1%7D)

    而当`f_1`和`f_2`均为向量函数时，适应性链式法则如下

    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20%5Cvec%7Bg%7D%7D%7B%5Cpartial%20x_1%7D%3DJ%5Cvec%7Bf_1%7D%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_1%7D&plus;J%5Cvec%7Bf_2%7D%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_2%7D%7D%7B%5Cpartial%20x_1%7D)

    其中`J\vec{f_1}`为Jacobi矩阵，定义如下

    ![](https://latex.codecogs.com/gif.latex?J%5Cvec%7Bf_1%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_1%7D%7D%7B%5Cpartial%20x_n%7D%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%20%5Cvdots%20%5C%5C%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_m%7D%7D%7B%5Cpartial%20x_1%7D%20%26%20%5Ccdots%20%26%20%5Cfrac%7B%5Cpartial%20%5Cvec%7Bf_m%7D%7D%7B%5Cpartial%20x_n%7D%20%5Cend%7Bbmatrix%7D%28i%3D1%2C2%2C%5Cldots%2Cm%29)

    *链式法则中的乘法均为Hadamard Product (elementwise)
    
    **链式法则时深度学习中最为常用的求导规则，常用于逆向传播算法训练神经网络的工作**

4. 