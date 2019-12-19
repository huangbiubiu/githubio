---
title: 'Faster Data Loading in PyTorch'
date: 2019-12-19 19:38:57
tags:
 - PyTorch
 - TUTORIAL
 - As Fast As Possible
---

*不要让数据加载限制你的训练速度！*

教程：如何避免训练速度被数据加载拖累

<!--more-->

数据加载是神经网络训练的重要步骤。我们会花费大量金钱来购买更加昂贵的GPU，以求获得更快的训练（推理）速度。然而，数据加载的时间花费却获得较少的关注。如何优化数据加载的过程，从而充分利用GPU？

## 是不是：数据加载的时间是你的瓶颈吗？

「先问是不是」：解决这个问题之前，我们首先需要知道数据加载的时间是否真的拖慢了训练速度。下面介绍几个方法来查看训练中数据加载花费的时间。

### 直接测量: `time.time()`

最为直观的方法就是测量数据加载所需要的时间花费。Python中可以使用`time.time()`来返回当前的时间。在数据加载前后分别测量一次当前时间，相减后即为数据加载时间。PyTorch提供了一个很好的[示例](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L277-L280)：

```python
# only code snippet
# could not run directly

end = time.time()
for i, (images, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

```

> 在Python中，更加精确地测量代码运行时间的方法是`timeit.timeit()`。`timeit`允许你选择不同的时间函数（例如[`time.process_time()`](https://docs.python.org/3.7/library/time.html#time.process_time)来测量*CPU时间*而不是*当前时间*），以及重复多次运行来消除测量误差等，详见StackOverflow的[相关讨论](https://stackoverflow.com/questions/17579357/time-time-vs-timeit-timeit). 而对于本任务，我相信`time.time()`的精确度是足够的，而且只需要对代码做微小的改动即可。有关使用`timeit.timeit()`测量代码速度的例子，可以参考[文档](https://docs.python.org/3.7/library/timeit.html#basic-examples)以及[这篇博客](https://huangbiubiu.github.io/2019/BEST-PRACTICE-PyTorch-TensorDataset/#%E4%BB%A3%E7%A0%811-%E8%AF%84%E4%BC%B0%E4%BB%A3%E7%A0%81%E8%BF%90%E8%A1%8C%E9%80%9F%E5%BA%A6)。

> **注意** 此方法应当运行多次后，以测量结果稳定后的数值为准。

### 使用Profile工具: cProfile

[cProfile](https://docs.python.org/3.7/library/profile.html)是Python提供的一个分析器，可以测量出不同函数被调用的次数，时间等信息。

这是一个数据加载时间缓慢的代码片段（来自于[这篇文章](https://huangbiubiu.github.io/2019/BEST-PRACTICE-PyTorch-TensorDataset/)）:

```python
import timeit

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_data() -> DataLoader:
    batch_size = 8192

    data_all = np.random.rand(batch_size * 100, 128)  # demo input
    dataset = TensorDataset(torch.from_numpy(data_all))
    data_loader = DataLoader(dataset=dataset,
                             shuffle=True,
                             batch_size=8192)

    return data_loader


def iterate_data(dataloader):
    for i, x in enumerate(dataloader):
        # training
        pass


dataloader = prepare_data()
if __name__ == '__main__':
    print(timeit.timeit('iterate_data(dataloader)', globals=globals(), number=5))

```

将这段代码保存为`test.py`文件，执行命令:

```
python -m cProfile -s time test.py
```

程序运行结束后得到如下结果（截取部分）:
```
44.836229782085866
         20666773 function calls (20660872 primitive calls) in 50.746 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  8192000   20.637    0.000   20.637    0.000 dataset.py:162(<genexpr>)
  4096000    8.601    0.000   29.238    0.000 dataset.py:161(__getitem__)
      500    8.044    0.016    8.044    0.016 {built-in method stack}
      500    3.585    0.007   32.823    0.066 fetch.py:44(<listcomp>)
    43/41    2.715    0.063    2.721    0.066 {built-in method _imp.create_dynamic}
      505    1.417    0.003    2.467    0.005 sampler.py:198(__iter__)
        1    1.303    1.303    1.303    1.303 {method 'rand' of 'numpy.random.mtrand.RandomState' objects}
      505    0.847    0.002   44.629    0.088 dataloader.py:344(__next__)
      404    0.844    0.002    0.844    0.002 {method 'read' of '_io.FileIO' objects}
4101414/4101115    0.354    0.000    0.354    0.000 {built-in method builtins.len}
        1    0.328    0.328    0.328    0.328 {built-in method mkl._py_mkl_service.get_version}
  4101636    0.317    0.000    0.317    0.000 {method 'append' of 'list' objects}
 1000/500    0.234    0.000    8.347    0.017 collate.py:42(default_collate)
        5    0.202    0.040    0.202    0.040 {method 'tolist' of 'torch._C._TensorBase' objects}
      404    0.184    0.000    1.028    0.003 <frozen importlib._bootstrap_external>:914(get_data)
        5    0.178    0.036    0.178    0.036 {built-in method randperm}
      500    0.143    0.000   41.313    0.083 fetch.py:42(fetch)
        5    0.139    0.028   44.830    8.966 test.py:20(iterate_data)
        1    0.092    0.092    0.092    0.092 {built-in method from_numpy}
      500    0.058    0.000    8.108    0.016 collate.py:79(<listcomp>)
      404    0.051    0.000    0.051    0.000 {built-in method marshal.loads}
        5    0.038    0.008    0.038    0.008 {method 'random_' of 'torch._C._TensorBase' objects}
        9    0.035    0.004    0.035    0.004 {method 'readline' of '_io.BufferedReader' objects}
```

可以看到每个函数被调用的时间和次数。从这个结果我们可以看到，在100次迭代中，[`__getitem__`](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class)函数被调用了4096000次，花费时间29.238s。

### 间接观察：nvidia-smi

我们也可以通过不断观察`nvidia-smi`的GPU利用率字段来推断GPU是否处于满负荷状态：

`watch -n0 nvidia-smi`

[该命令](https://en.wikipedia.org/wiki/Watch_(Unix))每0.1s刷新一次`nvidia-smi`的状态。如果发现`Volatile GPU-Util`字段不是一直处于100%，有几种可能：

1. 程序对于GPU太弱了，GPU不需要满负荷运转就可以轻松应付你的程序；
2. 这个字段不能准确反映GPU利用率，请参考[这里的讨论](https://stackoverflow.com/a/40938696/5634636)；
3. 数据加载等操作阻塞了GPU的运算。

第三种情况才是我们需要关注的情况。这种情况下，GPU利用率会呈现“过山车”式的曲线，一会达到100%（GPU运算，模型推理过程），一会非常低（被数据加载代码阻塞）。


## 为什么：什么操作会拖慢数据加载？

### 数据增强和预处理

普通的数据增强(例如[`torchvision.transforms`](https://pytorch.org/docs/stable/torchvision/transforms.html)中的transformer)通常比较快。如果你使用了额外的数据增强和预处理方法，请着重关注他们的执行效率。这样的速度减缓通常可以在上文提到的`cProfile`的结果中反映出来。

### IO

IO通常包括磁盘IO和网络IO。深度学习的训练中，通常不会涉及大量的网络吞吐。多数情况下，可能导致的网络IO瓶颈是分布式训练的过程中。

#### 磁盘IO

磁盘IO瓶颈常见于在大数据集和高性能显卡的环境下。这样的情况下，GPU运算速度非常快，而数据集很大，导致数据加载的速度跟不上数据运算的速度。

如何判断你的程序遇到了磁盘IO的瓶颈？一种方法是检查系统的`iowait`。`iowait`表示CPU等待磁盘的时间，有关具体的含义，请参考[这里的讨论](https://serverfault.com/questions/12679/can-anyone-explain-precisely-what-iowait-is). 

可以通过命令

```
iostat -x 1
```

观察每个磁盘的IO情况以及总的CPU iowait:

```
avg-cpu:  %user   %nice %system %iowait  %steal   %idle
           0.12    0.00    0.21    0.04    0.00   99.62

Device:         rrqm/s   wrqm/s     r/s     w/s    rkB/s    wkB/s avgrq-sz avgqu-sz   await r_await w_await  svctm  %util
loop0             0.00     0.00    0.00    0.00     0.00     0.00     0.00     0.00    0.00    0.00    0.00   0.00   0.00
sdg               0.00     0.00    0.00    0.00     0.00     0.00     0.00     0.00    0.00    0.00    0.00   0.00   0.00
sdb               0.00     0.00    0.00    0.00     0.00     0.00     0.00     0.00    0.00    0.00    0.00   0.00   0.00
sdd               0.00     0.00    0.00    0.00     0.00     0.00     0.00     0.00    0.00    0.00    0.00   0.00   0.00
dm-0              0.00     0.00    0.00    1.00     0.00     4.00     8.00     0.00    0.00    0.00    0.00   0.00   0.00
dm-1              0.00     0.00    0.00    0.00     0.00     0.00     0.00     0.00    0.00    0.00    0.00   0.00   0.00
```

> iostat没有在Ubuntu系统上预先安装。对于Ubuntu系统，请预先安装此程序：`sudo apt-get install sysstat`

一般来说，如果`iowait`长期保持100%，就是磁盘瓶颈的警告。也可以通过观察`avgqu-sz`字段判断IO队列长度。

#### 分布式训练

分布式训练有可能会出现网络IO的瓶颈。Linux有大量监控网络流量的方法，例如[`nethogs`](https://github.com/raboof/nethogs)，以及这里的讨论https://askubuntu.com/questions/257263/how-to-display-network-traffic-in-the-terminal。

一般来说，在分布式训练时，如果网络流量长期处于顶点（例如千兆网络一直被占满），那么就有理由猜测可能是网络限制了分布式训练的过程，包括数据和模型参数的传输过程。此情况可能在大数据集和大模型训练的时候发生。

### PyTorch的一些坑

例如，PyTorch的`TensorDataset`配合`DataLoader`使用可能会发生数据加载过慢的情况，分析及解决方案请参考[这篇博文](https://huangbiubiu.github.io/2019/BEST-PRACTICE-PyTorch-TensorDataset/).

## 怎么做：如何提高数据加载的速度

面对不同的原因，列出不同的解决方法如下。

### 数据增强和预处理

1. 考虑提高运行效率，包括但不限于使用多进程(线程)处理等
2. 考虑将数据增强和预处理操作离线进行，即预先处理好数据存在磁盘中，在训练时直接加载处理好的数据。

### IO

#### 磁盘IO

1. 将数据存储在SSD中通常能解决此问题
2. 如果内存足够大，考虑将所有数据加载到内存中（通常不太可行）

#### 分布式训练

1. 升级为高速网络连接，如万兆网络、[InfiniBand](https://zh.wikipedia.org/wiki/InfiniBand)
2. 使用单机训练

