---
layout:     post
title:      知识树：浅析Spark内部实现
subtitle:   MSBD5003 阶段性知识总结
date:       2020-10-21
author:     Tex
header-img: img/post-bg-sparkLogo.png
catalog: true
tags:
    - 大数据分析 (Big Data Analytics)
    - 大数据计算 (Big Data Computing)

---

>SparkRDD内部涉及最关键的问题在于分区，了解清楚分区机制是如何善用Spark的关键。

### 分区基础

每个RDD在初始化以及后续存储中都会分区 (Partitioning)，分区的目的在于并行 (Parallelism)。初始化时分区总是平衡的，但某些操作后，分区就不再平衡了。因此明确何时重分区，怎样重分区，是使程序高效的关键。

RDD中有两种重分区方法，如下

```python
rdd.repartition(n)
rdd.partitionBy(n,f)
```

第一种方法不会产生分区器，而仅仅是使数据平衡；第二种方法是通过哈希分区器和自定义分区函数来实现，合理编写分区函数可以应对哈希分区器的缺陷。

### 分区方法

就分区性质来说，同一RDD的分区绝对不会散列在多台机器上，且分区数量是可以指定的，默认的分区数与所有执行器总核数相同（从HDFS/WASB加载数据时除外，因为HDFS中file的分区信息会被继承）。

#### 哈希分区 (Hash Partitioning)

通常情况下，哈希分区会将键值对 $(k,v)$散列分配到分区$p$上，其中
$$
p = k.hashcode()\;\;\%\;\;numPartitions  
$$
使用哈希分区需要注意不良输入，这会导致哈希分区的不正常工作，即相同的key很多的情况下，会导致数据倾斜。

#### 区间分区 (Range Partitioning)

正如该实例，

```
An RDD with keys [8, 96, 240, 400, 401, 800], 
Number of partitions: 4
In this case, hash partitioning distributes the keys as follows among the partitions:
partition 0: [8, 96, 240, 400, 800]
partition 1: [401]
partition 2: []
partition 3: []
```

为了避免这种哈希分区法不能应对的状况，提出了一种基于对key进行排序的分区法，首先将key切分成若干区间，每段区间对应一个分区。区间分区保证各分区间key的有序性，且每个分区基本平衡，但单个分区内的key的有序性是不保证的。

#### 分区法的选择

在Spark的内部实现中，规定了某些转换操作对应选择的分区算法，例如，`sortByKey`使用`RangePartitioner`，`groupByKey`使用`HashPartitioner`。

除此之外，用户可以通过`partitionBy()`方法创建自定义的分区函数。

#### 转换操作后的分区

在对RDD的转换操作中，某些情况下数据的分区性质是得以继承的，包括`mapValues`/`flatMapValues`/`filter`，而其他转换操作均无法保留前一个RDD的分区性质。以`map`转换操作为例，经过转换后可能会改变key，因此原RDD的分区信息无法保留。为此Spark设计了`mapValues`方法，在映射过程中不改变key，进而达到可以保留分区信息的目的。

![](/home/tian/Github/edwardtex.github.io/img/post-sql-2.png)

上述性质可由如下实例说明，

```python
data = [8, 96, 240, 400, 1, 800, 4, 12]
rdd = sc.parallelize(zip(data, data),4)
print(rdd.partitioner)
print(rdd.glom().collect())
rdd = rdd.reduceByKey(lambda x,y: x+y)
print(rdd.glom().collect())
print(rdd.partitioner)
print(rdd.partitioner.partitionFunc)

rdd1 = rdd.map(lambda x: (x[0], x[1]+1))
print(rdd1.glom().collect())
print(rdd1.partitioner)

rdd2 = rdd.mapValues(lambda x: x+1)
print(rdd2.partitioner.partitionFunc)

rdd = rdd.sortByKey()
print(rdd.glom().collect())
print(rdd.partitioner.partitionFunc)
rdd3 = rdd.mapValues(lambda x: x+1)
print(rdd3.partitioner.partitionFunc)
```

输出为

```
None
[[(8, 8), (96, 96)], [(240, 240), (400, 400)], [(1, 1), (800, 800)], [(4, 4), (12, 12)]]
[[(8, 8), (96, 96), (240, 240), (400, 400), (800, 800), (4, 4), (12, 12)], [(1, 1)], [], []]
<pyspark.rdd.Partitioner object at 0x7f7838d768b0>
<function portable_hash at 0x7f783970b700>
[[(8, 9), (96, 97), (240, 241), (400, 401), (800, 801), (4, 5), (12, 13)], [(1, 2)], [], []]
None
<function portable_hash at 0x7f783970b700>
[[(1, 1), (4, 4), (8, 8)], [(12, 12), (96, 96)], [(240, 240), (400, 400)], [(800, 800)]]
<function RDD.sortByKey.<locals>.rangePartitioner at 0x7f7838d31f70>
<function RDD.sortByKey.<locals>.rangePartitioner at 0x7f7838d31f70>
```

首先一个reduceByKey生成一个hash partitioner，经过map后，为none，但再mapValue又会回到hash partitioner(retain机制)。之后一个sortByKey生成一个range partitioner，mapValue保持range partitioner。

### 洗牌操作 (Shuffle Operations)

在Spark内部有必要的广依赖(Wide Dependencies)，通过某些转换操作内部的洗牌操作实现，包括`reduceByKey`/`repartition`/`substract`/`coalesce`/`join(based on different partitioner)`，经过洗牌操作的结果在Spark内部会自动缓存。

其他重要的内部实现还包括工作(job)内部的任务(task)调度，以及内存管理等。

