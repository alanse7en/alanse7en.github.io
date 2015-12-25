---
layout: post
title: "Caffe代码解析(1)"
modified: 2015-12-25 16:20:45 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: 
  credit: 
  creditlink: 
comments: true
share: true
---

使用Caffe做各种实验也有一段时间了，除了Caffe支持的各种layer之外，在自己的使用中开始遇到一些需要自己定义的网络，也有一些新的模型，比如2015ImageNet的冠军里面的shortcut结构，都需要实现新的layer。为了去改造Caffe，首先打算学习一下Caffe的代码，接下来一系列的博客将记录一下这个学习的过程。

Caffe主要包含了4个大类:`Solver`, `Net`, `Layer`, `Blob`。

其中Solver这个类有几个子类分别实现了不同的优化方法：`SGDSolver`, `NesterovSolver`, `AdaGradSolver`, `RMSPropSolver`, `AdaDeltaSolver`和`AdamSolver`。具体每个Solver对应的优化方法参考：<a href = "http://caffe.berkeleyvision.org/tutorial/solver.htm">Caffe Solver Methods</a>。
类似地Layer这个类派生出了很多子类，这些子类实现了Data的读取和Convolution, Pooling, InnerProduct等各种功能的layer。
Blob则是Caffe对数据的封装，在整个网络的计算中，不管是数据还是网络的参数和梯度都是这个类的对象，均为num*channel*width*height形式的数据。可以看出Caffe的整个代码结构是很清楚的，也很方便添加新的优化方法和自定义功能的layer。

除了清晰的代码结构，让Caffe变得易用更应该归功于<a href = "https://developers.google.com/protocol-buffers/">`Google Protocol Buffer`</a>的使用。`Google Protocol Buffer`是Google开发的一个用于serializing结构化数据的开源工具:

> Protocol buffers are a language-neutral, platform-neutral extensible mechanism for serializing structured data.

Caffe使用这个工具来定义Solver和Net，以及Net中每一个layer的参数。这使得只是想使用Caffe目前支持的Layer(已经非常丰富了)来做一些实验或者demo的用户可以不去和代码打交道，只需要在`*.prototxt`文件中描述自己的Solver和Net即可。下一篇文章将通过一个简单的例子来展示`Google Protocol Buffer`的作用和便捷之处。