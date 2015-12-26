---
layout: post
title: "Caffe代码解析(2)"
modified: 2015-12-25 20:24:49 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/ 
comments: 
share: 
---

在Caffe中定义一个网络是通过编辑一个prototxt文件来完成的，一个简单的网络定义文件如下：

{% highlight cpp %}
name: "ExampleNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "path/to/train_database"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "conv1"
  top: "ip1"
  inner_product_param {
    num_output: 500
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip1"
  bottom: "label"
  top: "loss"
}
{% endhighlight %}
这个网络定义了一个`name`为`ExampleNet`的网络，这个网络的输入数据是`LMDB`数据，`batch_size`为64，包含了一个卷积层和一个全连接层，训练的`loss function`为`SoftmaxWithLoss`。通过这种简单的`key: value`描述方式，用户可以很方便的定义自己的网络，利用Caffe来训练和测试网络，验证自己的想法。

Caffe中定义了丰富的layer类型，每个类型都有对应的一些参数来描述这一个layer。为了说明的方便，接下来将通过一个简单的例子来展示Caffe是如何使用`Google Protocol Buffer`来完成`Solver`和`Net`的定义。

首先我们需要了解`Google Protocol Buffer`定义data schema的方式，`Google Protocol Buffer`通过一种类似于C++的语言来定义数据结构，下面是官网上一个典型的AddressBook例子：

{% highlight cpp %}
package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phone = 4;
}

message AddressBook {
  repeated Person person = 1;
}
{% endhighlight %}