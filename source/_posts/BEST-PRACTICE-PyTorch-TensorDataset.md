---
title: 'Faster PyTorch TensorDataset'
date: 2019-12-13 22:01:22
tags:
 - PyTorch
 - BEST PRACTICE
 - As Fast As Possible
---

<!-- # [BEST PRACTICE] PyTorch TensorDataset -->

一个更加快速的`TensorDataset`使用方法, 70x速度提升!

<!--more-->

## Background

PyTorch的[`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)类提供了非常好用的数据加载接口。`TensorDataset`继承了`Dataset`，提供了已经完全加载到内存中的矩阵的数据读取接口。一个普遍的使用方法是这样的:

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


data_all = np.random.rand(100000, 128)  # demo input
dataset = TensorDataset(torch.from_numpy(data_all))
data_loader = DataLoader(dataset=dataset, 
                            shuffle=True,
                            batch_size=8192)

for x in data_loader:
    # training
    pass
```

测量一下数据读取的速度(代码见附录1), 5次运行结果的平均值（单位是秒）:
> 37.70331398304552

## Problem

5次运行的平均速度是37.7s， `batch size`是8192，一个`epoch`100个step，平均每个step花费的时间是0.3s. 这个时间对于很多训练任务是无法接受的。作为参考，我们使用4*Tesla V100在ImageNet上训练ResNet 50，每个step的时间是0.27s. 在这个情况下，上述方法读取数据的时间已经超过了模型forward和backward的时间，极大拖慢了运行速度. 

问题出在哪里？`TensorDataset`中，数据全部存储在内存中，每次需要数据的时候直接从内存中取出相应的数据即可，不存在IO瓶颈的问题。

问题在于，对于`DataLoader`，每次调用`Dataset`中一个值的时候，循环地调用`Dataset`的`__getitem__`函数，类似于以下这种写法:

```python
# only a simplified demo
def get_next_batch():
    results = []
    for i in range(indices):
        results.append(dataset[i])

    return torch.cat(results) # default collate_fn
```

这样写对于需要从磁盘中读取的数据是没有问题的，但是对于`Tensor`，我们知道有更高效的写法:

```python
def get_next_batch():
    return data[indices]
```

这个问题在PyTorch issue中已有[相关讨论](https://github.com/pytorch/pytorch/issues/4959).

## Solution

那么关键就在于如何在尽可能少地改动代码的情况下保证`DataLoader`使用自定义的index. 我们使用[`Sampler`](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)控制`DataLoader`的采样方法，一次返回一批`Tensor`，而不是一次返回一条数据然后再concat起来. 此方法参考了[@fmassa的回复](https://github.com/pytorch/pytorch/issues/4959#issuecomment-362424598)。

```python
class TensorSampler(Sampler):
    def __init__(self, data_source: Sized, batch_size=8192):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long().split(self.batch_size))
```

这样一次`__getitem__`就会返回一个batch的数据。此时需要[禁用`DataLoader`的自动batch](https://pytorch.org/docs/stable/data.html#disable-automatic-batching), 由Sampler来控制batch:

```python
data_loader = DataLoader(dataset=dataset, shuffle=False,
                        batch_size=None, batch_sampler=None,
                        sampler=TensorSampler(data_source=dataset,
                                              batch_size=batch_size))
```

修改后的代码参见附录2. 

> Note: 在使用自定义Sampler时，`DataLoader`的shuffle选项将不可用。


## Experiment

重新运行优化过的代码，计算运行时间为（评估代码见附录2）:

> 0.5254814000800252

速度提升了**70**倍.


## Summary

在使用`TensorDataset`时，应尽量避免直接使用`Dataloader`，否则`Dataloader`的auto-batch机制会导致数据加载非常缓慢。一种可行的方法是使用自定义的`Sampler`控制每次从`Dataset`中的采样方式，一次直接取出一个batch的数据.


## Further Reading

1. PyTorch社区的相关讨论 https://github.com/pytorch/pytorch/issues/4959
2. `torch.utils.data`文档 https://pytorch.org/docs/stable/data.html#disable-automatic-batching

## Appendix

### 代码1: 评估代码运行速度

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

### 代码2: 评估改进后代码运行速度

```python
import timeit
from typing import Sized

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler


class TensorSampler(Sampler):
    def __init__(self, data_source: Sized, batch_size=8192):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long().split(self.batch_size))


def prepare_data() -> DataLoader:
    batch_size = 8192

    data_all = np.random.rand(batch_size * 100, 128)  # demo input
    dataset = TensorDataset(torch.from_numpy(data_all))

    data_loader = DataLoader(dataset=dataset, shuffle=False,
                             batch_size=None, batch_sampler=None,
                             sampler=TensorSampler(data_source=dataset,
                                                   batch_size=batch_size))

    return data_loader


def iterate_data(dataloader):
    for i, x in enumerate(dataloader):
        # training
        pass


dataloader = prepare_data()
if __name__ == '__main__':
    print(timeit.timeit('iterate_data(dataloader)', globals=globals(), number=5))

```



