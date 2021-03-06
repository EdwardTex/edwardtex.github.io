---
layout:     post
title:      阅读笔记：统计学基础（上）
subtitle:   国行经典教材 贾俊平版《统计学》 茆诗松版《概率论与数理统计》
date:       2020-01-30
author:     Tex
header-img: img/post-bg-jia_statistics.jpg
catalog: true
tags:
    - 统计学 (Statistics)

---

> 笔者从通信(Telecommunication)这个计算机科学(Computer Science)，电子工程(Elctronical Engneering)和控制科学(Control Science)大交融杂烩的专业即将跳往大数据技术(Big Data Technology)专业，其项目设置相当于数据科学(Data Science)，正式踏入另一个“深渊”，在朋友的推荐下从国行经典教材开始填补基础，尽管现在流行的知识和技术与传统的统计学已经相去略远。

### 基础概念

- 定义：当接触到大数据技术/数据科学这些名词的时候，与更传统的统计联系起来是自然而然的
    - 统计学(Statistics)：是收集、处理、分析、解释数据并从数据中得出结论的科学；
    - 相对来讲数据科学(Data Science)结合了统计学、机器学习及其相关方法，旨在利用数据对实际现象进行理解和分析；
    - 总体上来说，很多统计学家认为，数据科学并不是一个独立的学科，而是统计的一部分；现实的需求与其说产生了数据科学这个独立的学科，还不如说是推动统计学朝着一个可以被称为“数据科学”的方向连续地发展。


- 数据分析使用的方法可分为如下两类：
    - 描述统计(descriptive statistics)：研究数据收集/处理/汇总/图标描述/概括与分析等统计方法；
    - 推断统计(inferential statistics)：研究如何利用样本数据来推断总体特征的统计方法；

- 数据分析的目的是从数据中找到规律，从数据中寻找启发，而不是寻找支持。

- 统计可以帮助分析数据，并从分析中得出结论，但对统计结论的进一步解释就需要专业知识；


- 定义：数据是对现象进行测量的结果

    - 依据计量尺度分类
        - 分类数据(categorical data)
        - 顺序数据(rank data)
        - 数值型数据(metric data)
        - 计量尺度分四种：分类尺度(nominal scale)、顺序尺度(ordinal scale)、间隔尺度(interval scale)、比率尺度(ratio scale)；其中间隔尺度和比率尺度的数据表现均为数字，间隔尺度下称为定距数据，比率尺度下称为定比数据，定距数据是无法进行乘除等比率性质的运算的，区别体现在是否存在绝对零点，统计学中的绝对零点是指所测零的属性为无。典型的例子为摄氏度和开氏度的关系，摄氏度下为定距数据，开氏度下为定比数据。

    - 依据数据收集方法分类
        - 观测数据(observational data)
        - 实验数据(experimental data) 

    - 依据时间关系分类
        - 截面数据(cross-sectional data)
        - 时序数据(time series data)


- 总体(population)分为有限总体和无限总体主要是为了判别在抽样中每次抽取是否独立；

- 定义：参数和统计量

    - 参数(parameter)：用于描述总体特征的概括性数字度量
    - 统计量(statistic)：用于描述样本特征的概括性数字度量
    - 抽样的目的就是根据样本的统计量去估计总体参数；


- 如何选取一个好的样本是很关键的问题，好的样本是相对而言的，相对包括两方面的含义：一是针对研究的问题而言；二是针对调查费用与估计精度的关系而言。

- 数据的误差

    - 抽样误差(sampling error)：由抽样的随机性引起的样本结果与总体真值之间的误差；
    - 非抽样误差(non-sampling error)：由其他原因引起的样本观察结果与总体真值之间的差异。


- 数据预处理：包括审核、筛选、排序等

    - 审核：主要针对完整性和准确性：完整性包括单位或个体是否有遗漏，准确性指审核异常值，若异常值不是因为记录错误导致的，则应该保留；
    - 筛选：根据需要找出符合特定条件的数据；
    - 排序：目的是为了便于发现一些明显的特征/趋势；


### 可视化

- 区别：比例、百分比和比率

    - 比例(proportion)：指某部分与全体之比，用于反应样本在总体中的组成或结构；
    - 百分比(percentage)：特指%结尾的数据；
    - 比率(ratio)：指各部分之间的比值，可能会大于1；


- 区别：条形图和直方图

    - 条形图(bar/column chart)：宽度仅表示属性，无数值意义；
    - 直方图(histogram)：宽度不仅表示组别，还表示组距，有数值意义，因此其面积也具有了意义；


- 分类数据的图示：条形图、帕累托图(Pareto chart)、饼图(pie chart)、环形图(doughnut chart)；

- 顺序数据的图示：除了上述，还可以计算累积频数(cumulative frequencies)和累积频率(cumulative percentages)；

- 数值型数据的图示：

    - 分组数据：直方图；
    - 未分组数据：茎叶图(stem-and-leaf display)、箱线图(box plot)；
    - 时序数据：线图(line plot)；
    - 多变量数据：散点图(scatter diagram)、气泡图(bubble chart)、雷达/蜘蛛图(radar/spider chart)；

- 衡量图形优劣：
    - “初学者往往会在图形的修饰上花费太多的时间和精力，这样做得不偿失也未必合理，或许会画蛇添足”；
    - 图优性(graphical excellency)：
        - 显示数据；
        - 让读者把注意力集中在图形的内容而非制作程序上；
        - 避免歪曲；
        - 强调数据之间的比较；
        - 服务于一个明确的目的；
        - 有统计描述和文字说明；
   - 在绘制图形时应**避免一切不必要的修饰**：过于花哨的修饰往往会使人注重图形本身，而掩盖了图形所要表达的信息；图形的视觉效果应与数据所体现的事务特征一致，否则有可能歪曲数据，给人留下错误的印象；

    - 关于表格：
        - 首先要合理安排统计表的结构，行列标题的位置应合理，横竖长度比例应适当，尽量避免过高或过宽的表格形式；
        - 表格标题中应当简明确切地概括出统计表的内容；关于计量单位的标注也需要注意；
        - 表格一般采用上下粗中间细的三线表制式；表格中尽量少使用横竖线；表格中数据一般是右对齐，有小数点时应以小数点对齐，且位数应当统一；没有数据的单元用`-`表示，一张填好的表格不应出现空白单元格；
        - 必要时在表格末尾注明数据出处；


### 数据的概括性度量

- 数据的分布特征：
    - 分布的集中趋势：各数据向中心值聚拢程度，反映一组**数据中心点位置**所在；
    - 分布的离散程度：各数据远离其中心值的趋势；
    - 分布的形状：数据的**偏态**&**峰态**；

- 集中趋势(central tendency)的度量：
    - 众数(mode)：一组数据中出现次数最多的变量值，用`M_o`表示；通常只有在数据量较大的情况下，众数才有意义；只适用于**分类数据**；
    - 中位数(median)：排序后处于中间位置上的变量值，用`M_e`表示；顾名思义中位数将整组数据二等分，中位数不适用于分类数据，主要用于**顺序数据**和数值数据；
    - 四分位数(quartile)：通过三个点将排序后的数据四等分，类似的还有十分位数(decile)、百分位数(percentile)；同样不适用于分类数据；
    - 平均数(mean)：
        - 简单平均数(simple mean)：
        
            ![](https://latex.codecogs.com/gif.latex?%5Cbar%7Bx%7D%3D%5Cfrac%7Bx_1&plus;x_2&plus;%5Ccdots&plus;x_n%7D%7Bn%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20x_i)

        - 加权平均数(weighted mean)：

          ![](https://latex.codecogs.com/gif.latex?%5Cbar%7Bx%7D%3D%5Cfrac%7BM_1f_1&plus;M_2f_2&plus;%5Ccdots&plus;M_kf_k%7D%7Bf_1&plus;f_2&plus;%5Ccdots&plus;f_k%7D%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20M_if_i%20%7D%7B%5Csum%7Bf_i%7D%7D)

        - 几何平均数(geometric mean)：
          
          ![](https://latex.codecogs.com/gif.latex?G%3D%5Csqrt%5Bn%5D%7Bx_1*x_2*%5Ccdots*x_n%7D%3D%5Csqrt%5Bn%5D%7B%5Cprod_%7Bi%3D1%7D%5E%7Bn%7Dx_i%7D)

          主要用于计算平均比率，当变量值为比率形式时，使用几何平均数更合理；

       - 平均数的主要缺点在于易受极端数据值的影响，对于偏态分布的数据，平均数的代表性会差很多；


- 离散程度的度量：
    - 异众比率(variation ratio)：非众数组的频数占总频数的比例，主要用于测度分类数据的离散程度，也可用于顺序数据；

      ![](https://latex.codecogs.com/gif.latex?V_r%3D%5Cfrac%7B%5Csum%20f_i-f_m%7D%7B%5Csum%20f_i%7D%3D1-%5Cfrac%7Bf_m%7D%7B%5Csum%20f_i%7D)
      
    -  四分位差(quartile deviation)：也称为内距或四分间距(inter-quartile range)，主要用于顺序数据，也可用于数值数据，但不适用于分类数据；

    - 极差(range)：一组数据的最大值与最小值之差，也称全距，用`R`表示；
    - 平均差(mean deciation)：也称平均绝对离差，各变量值与其平均数离差绝对值的平均数；
    
      ![](https://latex.codecogs.com/gif.latex?M_d%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cleft%20%7C%20x_i-%5Coverline%7Bx%7D%20%5Cright%20%7C%7D%7Bn%7D)

    - 方差(variance)和标准差(standard deviation)：方差是各变量值与其平均数离差平方的平均数，标准差是方差的平方根；

      ![](https://latex.codecogs.com/gif.latex?s%5E2%20%3D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i-%5Coverline%7Bx%7D%29%5E2%7D%7Bn-1%7D)

      标准差和方差的区别在于，标准差是有量纲的，其与变量值的计量单位相同，实际意义比方差更清楚；其中分母`n-1`称为自由度(degree of freedom)，这里是指，为了使样本方差与总体方差（我们无法得知）相同，n个样本值中，只有n-1个是独立的不受约束的；从实际应用角度看，在抽样估计中，用如上定义的`s^2`去估计总体方差`\sigma^2`时，实现了无偏估计，只需简单证明一下：

      ![](https://latex.codecogs.com/gif.latex?E%5B%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i-%5Coverline%7Bx%7D%29%5E2%5D%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7DE%5B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i-%5Coverline%7Bx%7D%29%5E2%5D%20%5C%5C%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7DE%5B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_i%5E2-2x_i%5Coverline%7Bx%7D&plus;%5Coverline%7Bx%7D%5E2%29%5D%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7DE%5B%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%5E2-n%5Coverline%7Bx%7D%5E2%5D%20%5C%5C%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7DE%5Bx_i%5E2%5D%20-%20%5Cfrac%7Bn%7D%7Bn-1%7DE%5B%5Coverline%7Bx%7D%5E2%5D%20%5C%5C%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5BD%28x_i%29&plus;E%5E2%28x_i%29%5D%20-%20%5Cfrac%7Bn%7D%7Bn-1%7D%5BD%28%5Coverline%7Bx%7D%29&plus;E%5E2%28%5Coverline%7Bx%7D%29%5D%20%5C%5C%20%3D%20%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5B%5Csigma%5E2&plus;%5Cmu%5E2%5D%20-%20%5Cfrac%7Bn%7D%7Bn-1%7D%5B%5Cfrac%7B%5Csigma%5E2%7D%7Bn%7D&plus;%5Cmu%5E2%5D%20%3D%20%5Cfrac%7Bn%7D%7Bn-1%7D%5B%5Csigma%5E2&plus;%5Cmu%5E2%5D%20-%20%5Cfrac%7Bn%7D%7Bn-1%7D%5B%5Cfrac%7B%5Csigma%5E2%7D%7Bn%7D&plus;%5Cmu%5E2%5D%20%3D%20%5Csigma%5E2)

    - 标准分数(standard score)：变量值与其平均数的离差除以标准差，也称标准化值或z分数；

      ![](https://latex.codecogs.com/gif.latex?z_i%3D%5Cfrac%7Bx_i-%5Coverline%7Bx%7D%7D%7Bs%7D)

    - 对于对称分布的数据，有**经验法则**表明：
        - 约有68%的数据在平均数`±1`个标准差的范围内；
        - 约有95%的数据在平均数`±2`个标准差的范围内；
        - 约有99%的数据在平均数`±3`个标准差的范围内；
        - 在`±3`个标准差之外的数据，称为离群点(outlier)。

    - 对于任意形状分布的数据，我们使用**切比雪夫不等式(Chebyshev's inequality)**；

    - 离散系数(coefficient of variation)：也称变异系数，指标准差与其相应的平均数之比，主要是为了消除变量值水平高低和计量单位不同对不同数据组测度值的影响；

      ![](https://latex.codecogs.com/gif.latex?v_s%3D%5Cfrac%7Bs%7D%7B%5Coverline%7Bx%7D%7D) 

      **当平均数接近零的时候，离散系数的值趋于增大，此时必须慎重解释。*

- 分布形状的度量：
    - 偏态(skewness)：  
      对于对称分布的数据，偏态系数为0；若偏态系数的绝对值大于1，称为高度偏态分布：

      ![](https://latex.codecogs.com/gif.latex?SK%3D%5Cfrac%7Bn%5Csum%28x_i-%5Coverline%7Bx%7D%29%5E3%7D%7B%28n-1%29%28n-2%29s%5E3%7D)

    - 峰态(kurtosis)：
      峰态是与标准正态分布相比较而言的，对于标准正态分布，峰态系数为0；若峰态系数明显不等于0，称为平峰分布或尖峰分布：

      ![](https://latex.codecogs.com/gif.latex?K%3D%5Cfrac%7Bn%28n&plus;1%29%5Csum%28x_i-%5Coverline%7Bx%7D%29%5E4-3%5B%5Csum%28x_i-%5Coverline%7Bx%7D%29%5E2%5D%5E2%28n-1%29%7D%7B%28n-1%29%28n-2%29%28n-3%29s%5E4%7D)


### 概率与概率分布

- 在同一组条件下，对某事物或现象所进行的观察或实验叫做**试验**，把观察或实验的结果叫做**事件**；

- 随机事件用`A`等大写字母表示，必然事件用`\Omega`表示，不可能事件用`\Phi`表示；不可再分解的事件称为基本事件；

- 古典概型：
    - 结果有限；
    - 各个结果等可能出现；

- 运算法则
    - 加法法则：
        - 对于互斥事件，有
        
          ![](https://latex.codecogs.com/gif.latex?P%28A%20%5Ccup%20B%29%3DP%28A%29&plus;P%28B%29)
        
        - 对于任意随机事件，有
          
          ![](https://latex.codecogs.com/gif.latex?P%28A%20%5Ccup%20B%29%3DP%28A%29&plus;P%28B%29-P%28A%20%5Ccap%20B%29)

    - 乘法法则：
        - 对于条件概率，有

          ![](https://latex.codecogs.com/gif.latex?P%28A%20%7C%20B%29%20%3D%20%5Cfrac%7BP%28AB%29%7D%7BP%28B%29%7D%2C%20P%28B%29%3E%200)

          当事件相互独立时，有

          ![](https://latex.codecogs.com/gif.latex?P%28A%20%7C%20B%29%20%3D%20P%28A%29)

- 离散型随机分布
    - 包括0-1分布、均匀分布(uniform distribution)等；
    - 期望(exception value)，记作`\mu`：
        
      ![](https://latex.codecogs.com/gif.latex?E%28X%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%3Dx_ip_i)

    - 方差和标准差，记作`\sigma`：

      ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2%3DD%28X%29%3DE%5BX-E%28X%29%5E2%5D%5E2%5Cxrightarrow%5Bas%5D%7Bsimplified%7D%3DE%28X%5E2%29-%5BE%28X%29%5D%5E2)

    - 离散系数：

      ![](https://latex.codecogs.com/gif.latex?V%3D%5Cfrac%7B%5Csigma%7D%7B%5Cmu%7D%3D%5Cfrac%7B%5Csqrt%7BD%28X%29%7D%7D%7BE%28X%29%7D)

    - 二项分布(binomial distribution)：
        - 包含n个相同的实验；
        - 每个实验只有两种可能的结果；
        - 两种结果的概率对每次实验都是相同的；
        - 实验是相互独立的；
        - 符合以上性质的实验被称为伯努利实验(Bernoulli trials)；
        - 服从二项分布，记作`X~B(n,p)`，公式如下：
          
          ![](https://latex.codecogs.com/gif.latex?P%5C%7BX%3Dx%5C%7D%3DC_n%5Exp%5Exq%5E%7Bn-x%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20x%3D0%2C1%2C2%2C%5Ccdots%2Cn%20%5C%5C)

          其中：

          ![](https://latex.codecogs.com/gif.latex?P%5C%7BX%3Dx%5C%7D%5Cgeq%200%2C%5C%3A%20%5C%3A%20%5C%3A%20x%3D0%2C1%2C2%2C%5Ccdots%2Cn%20%5C%3A%20%5C%3A%20%5C%3A%5C%3A%20%5C%3A%20%5C%3A%20%5Csum_%7Bx%3D0%7D%5E%7Bn%7DC_n%5Exp%5Exq%5E%7Bn-x%7D%3D%28p&plus;q%29%5En%3D1)      

          ![](https://latex.codecogs.com/gif.latex?E%28X%29%3Dnp%5C%3A%20%5C%3A%3B%20%5C%3A%20D%28X%29%3Dnpq)    

    - 泊松分布(Poisson distribution)：
        - 用于描述在一指定时间范围内或在指定的面积或体积内某一事件出现的**次数**的分布；
        - 公式如下，其中`\lamda`为给定的时间间隔内事件的平均数：
          
          ![](https://latex.codecogs.com/gif.latex?P%28X%29%3D%5Cfrac%7B%5Clambda%5Exe%5E%7B-%5Clambda%7D%7D%7Bx%21%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20x%3D0%2C1%2C2%2C%5Ccdots)

          ![](https://latex.codecogs.com/gif.latex?E%28X%29%3D%5Clambda%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20D%28X%29%3D%5Clambda)

        - 在伯努利试验中，当概率`p`很小，试验次数`n`很大时，二项分布可近似等于泊松分布，即

          ![](https://latex.codecogs.com/gif.latex?C_n%5Exp%5Exq%5E%7Bn-x%7D%5Capprox%20%5Cfrac%7B%5Clambda%5Exe%5E%7B-%5Clambda%7D%7D%7Bx%21%7D)

          在实际应用中，`p`低于0.25，`n`高于20，`n*p`不高于5时，用泊松分布近似二项分布效果良好。


- 连续型随机分布
    - 概率密度函数(probability density function)，记作`f(x)`，满足如下条件：
    
      ![](https://latex.codecogs.com/gif.latex?f%28x%29%5Cgeq%200%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%26%20%5C%3A%20%5C%3A%20%5C%3A%5Cint_%7B-%20%5Cinfty%7D%5E%7B&plus;%20%5Cinfty%7Df%28x%29dx%3D1)

    - 分布函数记作`F(x)`，定义为：

      ![](https://latex.codecogs.com/gif.latex?F%28x%29%3DP%28X%5Cleq%20x%29%3D%5Cint_%7B-%20%5Cinfty%7D%5E%7Bx%7Df%28t%29dt%2C%5C%3A%20%5C%3A%20%5C%3A%20-%5Cinfty%3C%20x%3C&plus;%5Cinfty)

    - 显然有
      
      ![](https://latex.codecogs.com/gif.latex?P%28a%20%3C%20X%20%3C%20b%29%3D%5Cint_%7Ba%7D%5E%7Bb%7Df%28x%29dx%3DF%28b%29-F%28a%29%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20f%28x%29%3DF%27%28x%29)

      ![](https://latex.codecogs.com/gif.latex?E%28X%29%3D%5Cint_%7B-%20%5Cinfty%7D%5E%7B&plus;%20%5Cinfty%7Dxf%28x%29dx%3D%5Cmu%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20D%28X%29%3D%5Cint_%7B-%20%5Cinfty%7D%5E%7B&plus;%20%5Cinfty%7D%5Bx-E%28x%29%5D%5E2f%28x%29dx%3D%5Csigma%5E2)

    - 正态分布(normal distribution)
        - 记作`X~N(\mu,\sigma^2)`，公式如下：

          ![](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B1%7D%7B%5Csigma%5Csqrt%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%28x-%5Cmu%29%5E2%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20-%20%5Cinfty%3Cx%20%3C&plus;%20%5Cinfty)

        - 在`x=\mu`处函数取到最大值；
        - 当`\mu=0`、`\sigma=1`时，称为标准正态分布(standard normal distribution)，用`\phi(x)`表示概率密度函数，用`\Phi(x)`表示分布函数；
        
    - 二项分布的正态近似
        - 德莫夫-拉普拉斯定理(Demoivre-Laplace)给出：设随机变量`X~B(n,p)`，对任意`x`，有：

          ![](https://latex.codecogs.com/gif.latex?%5Clim_%7B%20n%5Crightarrow%20%5Cinfty%7DP%5C%7B%5Cfrac%7BX-np%7D%7B%5Csqrt%7Bnp%281-p%29%7D%7D%5Cleq%20x%5C%7D%3D%5Cint_%7B-%5Cinfty%7D%5E%7Bx%7D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%7De%5E%7B-%5Cfrac%7Bt%5E2%7D%7B2%7D%7Ddt)
          
          该正态近似很重要，它提供了计算二项概率和的一种实用且简便的近似方法；

### 统计量及其抽样分布

- 常用统计量
    - 当`n`充分大时，有定理可以保证经验分布函数`F_n(x)`很靠近总体分布函数`F(X)`，通常把经验分布函数的各阶**矩**称为样本各阶矩；
    - 样本均值
    
      ![](https://latex.codecogs.com/gif.latex?%5Coverline%7BX%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7DX_i)

    - 样本方差

      ![](https://latex.codecogs.com/gif.latex?S%5E2%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5E2)

    - 样本变异系数

      ![](https://latex.codecogs.com/gif.latex?V%3D%5Cfrac%7BS%7D%7B%5Coverline%7BX%7D%7D)

    - 样本k阶矩
    
      ![](https://latex.codecogs.com/gif.latex?m_k%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7DX_i%5Ek)

      显然`m_1=\overline{X}`

    - 样本k阶中心矩

      ![](https://latex.codecogs.com/gif.latex?v_k%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5Ek)

      显然`v_2=S^2`

    - 样本偏度

      ![](https://latex.codecogs.com/gif.latex?%5Calpha_3%3D%5Cfrac%7B%5Csqrt%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5E3%7D%7B%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5E2%29%5E%5Cfrac%7B3%7D%7B2%7D%7D)

    - 样本峰度

      ![](https://latex.codecogs.com/gif.latex?%5Calpha_4%3D%5Cfrac%7B%28n-1%29%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5E4%7D%7B%5B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28X_i-%5Coverline%7BX%7D%29%5E2%5D%5E2-3%7D)

- 抽样分布
    - 抽样分布中我们主要研究精确抽样分布，即在总体的分布类型已知时，对所有数据都能导出统计量的分布的数学表达式；而已求出的精确抽样分布并不多，在实际应用中，常采用极限分布作为抽样分布的一种近似，称为渐近分布；精确的抽样分布大多是在**正态总体**的情况下得到的，因此也可以说是**由正态分布导出的**，下面详细介绍；
    - `\Chi^2`分布
        
      设随机变量相互独立，且服从标准正态分布，则他们的平方和服从自由度为`n`的`\Chi^2`分布；

      ![](https://latex.codecogs.com/gif.latex?E%28%5Cchi%5E2%29%3Dn%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20D%28%5Cchi%5E2%29%3D2n%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20%5Cchi%5E2%28n_1%29&plus;%5Cchi%5E2%28n_2%29%5Csim%20%5Cchi%5E2%28n_1&plus;n_2%29)

      ![](https://i.loli.net/2020/02/05/ehkjqmSLDOH49ZA.jpg)

    - `t`分布
      
      设随机变量`X`服从标准正态分布，则随机变量`Y`服从卡方分布，且`X`和`Y`独立，则自由度为`n`的`t`分布如下；

      ![](https://latex.codecogs.com/gif.latex?t%3D%5Cfrac%7BX%7D%7B%5Csqrt%7BY/n%7D%7D)

      ![](https://latex.codecogs.com/gif.latex?E%28t%29%3D0%2C%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20n%5Cgeq2)

      ![](https://latex.codecogs.com/gif.latex?D%28t%29%3D%5Cfrac%7Bn%7D%7Bn-2%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20n%5Cgeq3)

      ![](https://i.loli.net/2020/02/05/hN8LDsWYX1VHjkf.jpg)

    - `F`分布

      设随机变量`Y`和`Z`相互独立，且`Y`和`Z`分别服从自由度`m`为`n`和的卡方分布，则称`X`服从第一自由度为`m`，第二自由度为`n`的`F`分布；

      ![](https://latex.codecogs.com/gif.latex?X%3D%5Cfrac%7BY/m%7D%7BZ/n%7D%3D%5Cfrac%7BnY%7D%7BmZ%7D)

      ![](https://latex.codecogs.com/gif.latex?E%28X%29%3D%5Cfrac%7Bn%7D%7Bn-2%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20n%3E2)

      ![](https://latex.codecogs.com/gif.latex?D%28X%29%3D%5Cfrac%7B2n%5E2%28m&plus;n-2%29%7D%7Bm%28n-2%29%28n-4%29%7D%2C%5C%3A%20%5C%3A%20%5C%3A%20%5C%3A%20n%3E4)

      ![](https://i.loli.net/2020/02/05/92YQex4TRorzVDl.jpg)

      **如果随机变量`X`服从`t`分布，则`X^2`服从`F(1,n)`的`F`分布，这在回归分析中的回归系数显著性检验中有用。*

- 样本均值的分布
    
    当总体服从正态分布`N(\mu,\sigma^2)`时，样本均值`\overline{X}`的抽样分布服从如下分布；

    ![](https://latex.codecogs.com/gif.latex?%5Coverline%7BX%7D%5Csim%20N%28%5Cmu%2C%5Cfrac%7B%5Csigma%5E2%7D%7Bn%7D%29)

- 中心极限定理(central limit theorem)

    设从均值为`\mu`，方差为`\sigma^2`的任意一个总体中抽取样本量为`n`的样本，当`n`充分大时，样本均值的抽样分布近似服从均值为`\mu`，方差为`\sigma^2/n`的正态分布；

    ![](https://i.loli.net/2020/02/05/xzhnb8jfaEqNg4K.jpg)