---
layout: post
title: "Caffe代码解析(1)"
modified: 2015-12-25 16:20:45 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/
comments: true
share: true
---
Caffe是一个基于C++和cuda开发的深度学习框架。其使用和开发的便捷特性使其成为近年来机器学习和计算机视觉领域最广为使用的框架。

笔者使用Caffe做各种实验也有一段时间了，除了Caffe支持的各种计算方式(卷积/pooling/全连接等)之外，在自己的使用中开始遇到一些需要自己定义的网络，也有一些新的模型，比如2015ImageNet的冠军里面的shortcut结构，都需要实现新的layer来完成计算。为了去改造Caffe，首先学习Caffe的源码，接下来一系列的博客将记录和分享这个学习的过程。作为一个C++的菜鸟，如有错误，希望读者指出。

首先，简单介绍一下Caffe的代码结构。Caffe主要包含了4个大类:

* Solver: An interface for classes that perform optimization on Nets
* Net: Connects Layers together into a directed acyclic graph (DAG) specified by a NetParameter
* Layer: An interface for the units of computation which can be composed into a Net
* Blob: A wrapper around SyncedMemory holders serving as the basic computational unit through which Layers, Nets, and Solvers interact

其中Solver这个类实现了优化函数的封装，其中有一个protected的成员:shared_ptr<Net<Dtype> > net_;，这个成员是一个指向Net类型的智能指针（shared_ptr），Solver正是通过这个指针来和网络Net来交互并完成模型的优化。不同的子类分别实现了不同的优化方法：SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver和AdamSolver。具体每个Solver对应的优化方法参考：<a href = "http://caffe.berkeleyvision.org/tutorial/solver.htm">Caffe Solver Methods</a>。
类似地Layer这个类派生出了很多子类，这些子类实现了Data的读取和Convolution, Pooling, InnerProduct等各种功能的layer。
Net则是对整个网络的一个封装，其中有一个成员为:vector<shared_ptr<Layer<Dtype> > > layers_;，这个vector中包含了整个网络中每一层layer的智能指针，Net通过调用这些layer各自的forward()和backward()接口实现了网络整体的ForwardBackward()。
Blob则是Caffe对数据的封装，在整个网络的计算中，不管是数据还是网络的参数和梯度都是这个类的对象，均为num\*channel\*width\*height形式的数据。

目前，初步的打算是从外部接口逐渐深入，首先学习caffe的主函数的接口，然后是Solver特别是默认使用的SGDSolver的具体实现，调用了哪些Net的接口等；接下来学习和了解Net是如何封装各个Layer来组成一个整体的网络，还有就是Net中如何利用Layer的接口完成数据的forward和backward的传导；最后具体了解不同的Layer如何实现自定义的forward()和backward()接口，完成最重要的计算。虽然目前Caffe已经实现了多GPU并行化的功能，但是在这个学习的过程中，我将暂时忽略这一部分的代码，而集中注意力到前面所述的这几部分内容上。

除了清晰的代码结构，让Caffe变得易用更应该归功于<a href = "https://developers.google.com/protocol-buffers/">Google Protocol Buffer</a>的使用。Google Protocol Buffer是Google开发的一个用于serializing结构化数据的开源工具:

> Protocol buffers are a language-neutral, platform-neutral extensible mechanism for serializing structured data.

Caffe使用这个工具来定义Solver和Net，以及Net中每一个layer的参数。这使得只是想使用Caffe目前支持的Layer(已经非常丰富了)来做一些实验或者demo的用户可以不去和代码打交道，只需要在*.prototxt文件中描述自己的Solver和Net即可，再通过Caffe提供的command line interfaces就可以完成模型的train/finetune/test等功能。下一篇文章将通过一个简单的例子来展示Google Protocol Buffer的作用和便捷之处。