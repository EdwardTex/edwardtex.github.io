---
layout:     post
title:      华为HiQ 与 IBM Q Experience 实战比较
subtitle:   量子计算仿真平台
date:       2019-10-24
author:     Tex
header-img: img/post-bg-quantum.jpg
catalog: true
tags:
    - 量子计算 (Quantum Computing)
    - 量子机器学习 (Quantum Machine Learning)
---

> 量子计算目前被认为是一种对未来具有颠覆性影响的新型计算模式，其思想最早由费曼在20世纪80年代提出。与传统计算机不同，量子计算机遵循量子力学规律、通过调控量子比特进行信息处理；基于微观量子比特的相干叠加和纠缠等特性，以及量子电路的可逆性，在计算速度和能耗方面大大优于传统计算机。随着研究的不断发展，量子计算未来在人工智能、数据搜索、化学模拟、生物制药等方面具有极大的潜在应用价值。量子计算虽然有很多诱人的革命性优势，但是量子计算机商用从硬件、软件、算法到系统集成均有非常多的技术挑战，是一个复杂的系统工程，既需要硬件、操控系统的突破，也需要软件、算法方面的突破。在当前量子操控技术不完善、硬件开发成本高规模小的情况下，利用目前传统计算机建立一套模拟量子计算机的计算平台和测控编程系统已经成为业界推动量子计算在软件、算法和硬件开发等发展的有效路径。


## Part1 华为HiQ


### 模拟量子生成随机数 (Quantum Random Number Generator)
#### 单比特生成单个随机数

![](http://i2.tiimg.com/702266/59ff190dbd3887dd.png)

首先申请一个Qbit q1，作为输入，默认申请为dirac(0)态，经过H门，变为
![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B%5Csqrt2%7D%28%7C0%3E&plus;%7C1%3E%29)
即测量时各有50%的概率得到dirac(0)态或dirac(1)态；代码如下；

```python
from projectq.ops import H, Measure 
from hiq.projectq.backends import SimulatorMPI  
from hiq.projectq.cengines import GreedyScheduler, HiQMainEngine  
	  
# 创建主引擎。所有的量子程序都必须创建一个主引擎，用于解析后续指令(required)  

eng = HiQMainEngine(SimulatorMPI(gate_fusion=True))  
	  
# 使用主引擎提供的方法创建一个量子比特  

q1 = eng.allocate_qubit()  
	  
# 将Hadamard门作用到该量子比特上，创建叠加态  

H | q1  
	  
# 测量该量子比特。测量基为 {|0>, |1>}  

Measure | q1  
	  
# 调用主引擎的刷新操作，使得所有指令被执行  

eng.flush()  
	  
# 输出测量结果。注意到测量结果依然存储在比特中，只是该比特已经塌缩成经典比特  

print("Measured: {}".format(int(q1)))  

```
-----------------
#### 多比特生成多个随机数

同上容易得到结果如下，其中生成多少次随机数，每次生成多少随机数可自行调整；

![](http://i1.fuimg.com/702266/b742f09819137abe.png)

```python

from projectq.ops import H, Measure  
from hiq.projectq.backends import SimulatorMPI  
from hiq.projectq.cengines import GreedyScheduler, HiQMainEngine  

eng = HiQMainEngine(SimulatorMPI(gate_fusion=True))  

for i in range(0,10):  
    q1 = eng.allocate_qubit()  
    H | q1  
    Measure | q1
    q2 = eng.allocate_qubit()  
    H | q2  
    Measure | q2  
    q3 = eng.allocate_qubit()  
    H | q3  
    Measure | q3    
    eng.flush()  
    print(format(int(q1)), format(int(q2)), format(int(q3))) 
```

## Part2 IBM Q Experience
### 量子隐形传态 (Quantum Teleportation)

![](http://i1.fuimg.com/702266/ec7c8d963a532827.png)

![](http://i1.fuimg.com/702266/7bc5d0b022aba68b.png)


```python
# IBM Q代码

OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

ry(pi/2) q[0];
h q[1];
rz(pi/2) q[0];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
cz q[0],q[2];
cx q[1],q[2];
measure q[2] -> c[0];
measure q[3] -> c[1];
```

q[0]为Alice需要传输的量子态，q[1]与q[2]为bell态，q[1]分发给Alice，q[2]分发给Bob，要求Alice不能和Bob进行量子通信，把q[0]的量子态传输到q[2]，测量结果展示在c[0]上。

block1为构造要传输的量子态，经过block1之后，Alice生成了需要传输的  态；block2为创建q[1]与q[2]的bell态，分别分发给Alice和Bob；block3为量子隐形传输的主体部分，block4中Alice测量了q[0]和q[1]，同时这两个比特塌陷到相应的测量状态；block5为Bob收到Alice的经典信息(q[0],q[1])之后做出的相应操作，然后进行测量；最后一个测量只是为了将c[1]置0，有利于我们观察测量结果。

数学推导过程如下

![](http://i1.fuimg.com/702266/574a83c3044ec91e.png)

 *block5的CZ和CNOT实际是根据前一步的测量得到的经典结果来确定是否需要X和Z门,即q[0] = 1则q[2]执行Z，否则不执行Z，q[1] = 1则q[2]执行X，否则不执行X。本实例中，用CZ和CNOT来代替上述功能，实现同样的效果。*

运行结果如下

![](http://i1.fuimg.com/702266/7fb72333e425c42e.png)

*可以看到，测得c[0]的结果为dirac(0)态和dirac(1)态的结果均在50%左右，我们在block1制备的要传输的量子态为![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B%5Csqrt2%7D%28%7C0%3E&plus;%7C1%3E%29)，说明我们隐形传输的结果正确；可自行调整block1部分以调整要传输的量子态，得到传输后q[2]的测量结果*

-----------------
### 制备量子纠缠 (Quantum Entanglement)
#### 制备Bell态

![](http://i1.fuimg.com/702266/8c34bbb77486765a.png)
![](http://i1.fuimg.com/702266/0007fc76cb2a39d5.png)

-----------------
#### 制备3比特纠缠

![](http://i1.fuimg.com/702266/7ac5c24b0582f19c.png)
![](http://i1.fuimg.com/702266/3e2fe46b9ed10d74.png)

-----------------
#### 制备4比特纠缠

![](http://i1.fuimg.com/702266/987c3b661180bb07.png)
![](http://i1.fuimg.com/702266/c94a1904ce7bb92a.png)


## Part3 Problems & Lessons
### 量子纠缠中制备量子状态Ry Rz

### 4Qbit量子纠缠
