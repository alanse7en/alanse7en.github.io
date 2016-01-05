---
layout: post
title: "Caffe代码(3)"
modified: 2016-01-05 16:31:39 +0800
tags: [Caffe,C++,Deep Learning]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/
comments: true
share: true
---

在上文对Google Protocol Buffer进行了简单的介绍之后，本文将对caffe的Command Line Interfaces进行分析。

本文将主要分为四部分的内容：

* Google Flags的使用
* Register Brew Function的宏的定义和使用
* `train()`函数的具体实现
* `SolverParam`的具体解析过程

### Google Flags的使用

从<a href = "http://caffe.berkeleyvision.org/tutorial/interfaces.html">Caffe官网</a>中可以看到，caffe的Command Line Interfaces一共提供了四个功能：train/test/time/device_query，而Interfaces的输入除了这四种功能还可以输入诸如-solver/-weights/-snapshot/-gpu等参数。这些参数的解析是通过Google Flags这个工具来完成的。

在caffe.cpp（位于/CAFFE_ROOT/tools/caffe.cpp）的开头，我们可以看到很多这样的宏：

{% highlight cpp %}
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
{% endhighlight %}

这个宏的使用方式为`DEFINE_xxx(name, default_value, instruction);`，这样就定义了一个xxx类型名为FLAGS_name的标志，如果用户没有在Command Line中用户没有提供其值，那么会默认为`default_value`，instruction是这个标志含义的说明。因此，上面的代码定义了一个string类型的名为FLAGS_gpu的标志，如果在Command Line中用户没有提供值，那么会默认为空字符串，根据说明可以得知这个标志是提供给用户来指定caffe将使用的GPU的。其余的定义也是类似的理解方式就不一一列举了。

解析这些标志的代码在caffe.cpp中的`main()`中调用了/CAFFE_ROOT/src/common.cpp中的`GlobalInit(&argc, &argv)`函数：

{% highlight cpp lineos%}
void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}
{% endhighlight %}

第三行的函数就是Google Flags用来解析输入的参数的，前两个参数分别是指向`main()`的`argc`和`argv`的指针，第三个参数为`true`，表示在解析完所有的标志之后将这些标志从`argv`中清除，因此在解析完成之后，`argc`的值为2，`argv[0]`为main，`argv[1]`为train/test/time/device_query中的一个。