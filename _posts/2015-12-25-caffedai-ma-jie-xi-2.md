---
layout: post
title: "Caffe代码分析(2)"
modified: 2015-12-25 20:24:49 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/ 
comments: 
share: 
---

本文将通过一个简单的例子来展示Caffe是如何使用`Google Protocol Buffer`来完成`Solver`和`Net`的定义。

首先我们需要了解`Google Protocol Buffer`定义data schema的方式，下面是官网上一个典型的AddressBook例子：

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