---
title: 'Implementation of Map in C++'
date: 2019-12-13 22:01:22
tags:
 - C++
 - BEST PRACTICE
 - As Fast As Possible
---

<!-- # Implementation of Map in C++ -->

Map (Dictionary) 是一种常见的抽象数据类型。本文探讨了C++中的`std::map`和Python对应的数据结构`dict`的具体实现差异。

<!--more-->

[Map（或Dictionary）](https://www.quora.com/What-is-a-map-data-structure-How-does-it-store-data)是一种用来查找的抽象数据类型（ADT）。在C++ STL中对应的类是`std::map`, Java中使用`java.util.Map`来表达这一类型，不同的实现方法对应着不同的子类，如`HashMap`，`TreeMap`等; 而Python则使用`dict`来实现。

> Further Reading: [ADT vs. Data Structure](https://softwareengineering.stackexchange.com/questions/148747/abstract-data-type-and-data-structure)




