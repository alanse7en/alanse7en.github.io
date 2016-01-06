---
layout: post
title: "Caffe代码解析(3)"
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

{% highlight cpp linenos %}
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

### Register Brew Function的宏的定义和使用

Caffe在Command Line Interfaces中一共提供了4种功能:train/test/time/device_query，分别对应着四个函数，这四个函数的调用是通过一个叫做`g_brew_map`的全局变量来完成的：

{% highlight cpp linenos %}
// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
{% endhighlight %}

`g_brew_map`是一个key为string类型，value为BrewFunction类型的一个全局的map，BrewFunction是一个函数指针类型，指向的是参数为空，返回值为int的函数，也就是train/test/time/device_query这四个函数的类型。在train等四个函数实现的后面都紧跟着这样一句宏的调用：`RegisterBrewFunction(train)`;

其中使用的宏的具体定义为：

{% highlight cpp linenos %}
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}
{% endhighlight %}

以train函数为例子，`RegisterBrewFunction(train)`这个宏的作用是定义了一个名为`__Register_train`的类，在定义完这个类之后，定义了一个这个类的变量，会调用构造函数，这个类的构造函数在前面提到的`g_brew_map`中添加了key为"train"，value为指向train函数的指针的一个元素。

然后函数的调用在`main()`函数中是通过下面的这段代码实现的，在完成初始化(GlobalInit)之后，有这样一句代码：

{% highlight cpp linenos %}
// main()中的调用代码
return GetBrewFunction(caffe::string(argv[1]))();
// BrewFunction的具体实现
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}
{% endhighlight %}

还是以train函数为例子，如果我们在Command Line中输入了`caffe train <args>`，经过Google Flags的解析argv[1]=train，因此，在`GetBrewFunction`中会通过`g_brew_map`返回一个指向train函数的函数指针，最后在main函数中就通过这个返回的函数指针完成了对train函数的调用。

总结一下：`RegisterBrewFunction`这个宏在每一个实现主要功能的函数之后将这个函数的名字和其对应的函数指针添加到了`g_brew_map`中，然后在main函数中，通过`GetBrewFunction`得到了我们需要调用的那个函数的函数指针，并完成了调用。

### `train()`函数的具体实现

接下来我们仔细地分析一下在`train()`的具体实现。

首先是这样的一段代码：

{% highlight cpp linenos %}
CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
    << "Give a snapshot to resume training or weights to finetune "
    "but not both.";
{% endhighlight %}

这段代码的第一行使用了glog的`CHECK_GT`宏（含义为check greater than），检查`FLAGS_solver`的size是否大于0，如果小于或等于0则输出提示："Need a solver definition to train"。`FLAGS_solver`是最开始通过`DEFINE_string`定义的标志，如果我们希望训练一个模型，那么自然应该应该提供对应的solver的路径，这一句话正是在确保我们提供了这样的标志。这样的检查语句在后续的代码中会经常出现，将不再一一详细解释，如果有不清楚含义的glog宏可以去看看<a href=http://google-glog.googlecode.com/svn/trunk/doc/glog.html>文档</a>。
与第一行代码类似，第二行代码是确保用户没有同时提供snapshot和weights参数，这两个参数都是继续之前的训练或者进行fine-tuning的。

然后出现了`SolverParameter solver_param`的声明和解析的代码：

{% highlight cpp linenos %}
caffe::SolverParameter solver_param;
caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
{% endhighlight %}

`SolverParameter`是通过`Google Protocol Buffer`自动生成的一个类，如果有不清楚的可以参考<a href=http://alanse7en.github.io/caffedai-ma-jie-xi-2/>上一篇文章</a>。而具体的解析函数将在下一部分具体解释。

接下来这一部分的代码是根据用户的设置来选择caffe工作的模式（GPU或CPU）以及使用哪些GPU（caffe已经支持了多GPU同时工作！具体使用参考：<a href=http://caffe.berkeleyvision.org/tutorial/interfaces.html>官网tutorial的Parallelism部分</a>）：

{% highlight cpp linenos %}
// If the gpus flag is not provided, allow the mode and device to be set
// in the solver prototxt.
if (FLAGS_gpu.size() == 0
    && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    if (solver_param.has_device_id()) {
        FLAGS_gpu = ""  +
            boost::lexical_cast<string>(solver_param.device_id());
    } else {  // Set default GPU if unspecified
        FLAGS_gpu = "" + boost::lexical_cast<string>(0);
    }
}
vector<int> gpus;
get_gpus(&gpus);
if (gpus.size() == 0) {
  LOG(INFO) << "Use CPU.";
  Caffe::set_mode(Caffe::CPU);
} else {
  ostringstream s;
  for (int i = 0; i < gpus.size(); ++i) {
    s << (i ? ", " : "") << gpus[i];
  }
  LOG(INFO) << "Using GPUs " << s.str();

  solver_param.set_device_id(gpus[0]);
  Caffe::SetDevice(gpus[0]);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_solver_count(gpus.size());
}
{% endhighlight %}