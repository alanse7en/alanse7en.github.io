---
layout: post
title: "Caffe代码解析(4)"
modified: 2016-01-13 15:15:57 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/
comments: true
share: true 
---

在上文对Command Line Interfaces进行了简单的介绍之后，本文将对caffe的Solver相关的代码进行分析。

本文将主要分为四部分的内容：

* Solver的初始化（Register宏和构造函数）
* `Solver::Solve()`具体实现
* `SIGINT`和`SIGHUP`信号的处理

## Solver的初始化（Register宏和构造函数）

{% highlight cpp %}
shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
{% endhighlight %}

caffe.cpp中的train函数中通过上面的代码定义了一个指向`Solver<float>`的shared_ptr。其中主要是通过调用`SolverRegistry`这个类的静态成员函数`CreateSolver`得到一个指向`Solver`的指针来构造shared_ptr类型的`solver`。而且由于C++多态的特性，尽管`solver`是一个指向基类`Solver`类型的指针，通过`solver`这个智能指针来调用各个成员函数会调用到各个子类(`SGDSolver`等)的函数。

下面我们就来具体看一下`SolverRegistry`这个类的代码，以便理解是如何通过同一个函数得到不同类型的Solver：

{% highlight cpp linenos %}
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    return registry[type](param);
  }
  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }
 private:
  SolverRegistry() {}
  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};
{% endhighlight %}

首先需要注意的是这个类的构造函数是private的，也就是用我们没有办法去构造一个这个类型的变量，这个类也没有数据成员，所有的成员函数也都是static的，可以直接调用。

我们首先从`CreateSolver`函数入手，这个函数先定义了string类型的变量type，表示Solver的类型('SGD'/'Nestrov'等)，然后定义了一个key类型为string，value类型为`Creator`的map：registry，其中`Creator`是一个函数指针类型，指向的函数的参数为`SolverParameter`类型，返回类型为`Solver<Dtype>*`(见第2行和第3行)。如果是一个已经register过的Solver类型，那么`registry.count(type)`应该为1，然后通过registry这个map返回了我们需要类型的Solver的creator，并调用这个creator函数，将creator返回的`Solver<Dtype>*`返回。

上面的代码中，`Registry`这个函数中定义了一个static的变量g_registry，这个变量是一个指向`CreatorRegistry`这个map类型的指针，然后直接返回，因为这个变量是static的，所以即使多次调用这个函数，也只会定义一个g_registry，而且在其他地方修改这个map里的内容，是存储在这个map中的。事实上各个Solver的register的过程正是往g_registry指向的那个map里添加以Solver的type为key，对应的Creator函数指针为value的内容。

下面我们具体来看一下Solver的register的过程：

{% highlight cpp linenos %}
template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
}
// register SGD Solver
REGISTER_SOLVER_CLASS(SGD);
{% endhighlight %}

在sgd_solver.cpp(SGD Solver对应的cpp文件)末尾有上面第xx行的代码，使用了`REGISTER_SOLVER_CLASS`这个宏，这个宏会定义一个名为`Creator_SGDSolver`的函数，这个函数即为`Creator`类型的指针指向的函数，在这个函数中调用了`SGDSolver`的构造函数，并将构造的这个变量得到的指针返回，这也就是Creator类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。然后在这个宏里又调用了`REGISTER_SOLVER_CREATOR`这个宏，这里分别定义了`SolverRegisterer`这个模板类的float和double类型的static变量，这会去调用各自的构造函数，而在`SolverRegisterer`的构造函数中调用了之前提到的`SolverRegistry`类的`AddCreator`函数，这个函数就是将刚才定义的`Creator_SGDSolver`这个函数的指针存到g_registry指向的map里面。类似地，所有的Solver对应的cpp文件的末尾都调用了这个宏来完成注册，在所有的Solver都注册之后，我们就可以通过之前描述的方式，通过g_registry得到对应的Creator函数的指针，并通过调用这个Creator函数来构造对应的Solver。