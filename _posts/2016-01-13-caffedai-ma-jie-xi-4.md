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

本文将主要分为三部分的内容：

* Solver的初始化（Register宏和构造函数）
* `SIGINT`和`SIGHUP`信号的处理
* `Solver::Solve()`具体实现


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

我们首先从`CreateSolver`函数(第15行)入手，这个函数先定义了string类型的变量type，表示Solver的类型('SGD'/'Nestrov'等)，然后定义了一个key类型为string，value类型为`Creator`的map：registry，其中`Creator`是一个函数指针类型，指向的函数的参数为`SolverParameter`类型，返回类型为`Solver<Dtype>*`(见第2行和第3行)。如果是一个已经register过的Solver类型，那么`registry.count(type)`应该为1，然后通过registry这个map返回了我们需要类型的Solver的creator，并调用这个creator函数，将creator返回的`Solver<Dtype>*`返回。

上面的代码中，`Registry`这个函数（第5行）中定义了一个static的变量g_registry，这个变量是一个指向`CreatorRegistry`这个map类型的指针，然后直接返回，因为这个变量是static的，所以即使多次调用这个函数，也只会定义一个g_registry，而且在其他地方修改这个map里的内容，是存储在这个map中的。事实上各个Solver的register的过程正是往g_registry指向的那个map里添加以Solver的type为key，对应的Creator函数指针为value的内容。

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

在sgd_solver.cpp(SGD Solver对应的cpp文件)末尾有上面第24行的代码，使用了`REGISTER_SOLVER_CLASS`这个宏，这个宏会定义一个名为`Creator_SGDSolver`的函数，这个函数即为`Creator`类型的指针指向的函数，在这个函数中调用了`SGDSolver`的构造函数，并将构造的这个变量得到的指针返回，这也就是Creator类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。然后在这个宏里又调用了`REGISTER_SOLVER_CREATOR`这个宏，这里分别定义了`SolverRegisterer`这个模板类的float和double类型的static变量，这会去调用各自的构造函数，而在`SolverRegisterer`的构造函数中调用了之前提到的`SolverRegistry`类的`AddCreator`函数，这个函数就是将刚才定义的`Creator_SGDSolver`这个函数的指针存到g_registry指向的map里面。类似地，所有的Solver对应的cpp文件的末尾都调用了这个宏来完成注册，在所有的Solver都注册之后，我们就可以通过之前描述的方式，通过g_registry得到对应的Creator函数的指针，并通过调用这个Creator函数来构造对应的Solver。

## `SIGINT`和`SIGHUP`信号的处理

Caffe在train或者test的过程中都有可能会遇到系统信号(用户按下ctrl+c或者关掉了控制的terminal)，我们可以通过对`sigint_effect`和`sighup_effect`来设置遇到系统信号的时候希望进行的处理方式：

> caffe train --solver=/path/to/solver.prototxt --sigint_effect=EFFECT --sighup_effect=EFFECT

在caffe.cpp中定义了一个GetRequesedAction函数来将设置的string类型的标志转变为枚举类型的变量：

{% highlight cpp linenos %}
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}
// SolverAction::Enum的定义
namespace SolverAction {
  enum Enum {
    NONE = 0,  // Take no special action.
    STOP = 1,  // Stop training. snapshot_after_train controls whether a
               // snapshot is created.
    SNAPSHOT = 2  // Take a snapshot, and keep training.
  };
}
{% endhighlight %}

其中SolverAction::Enum的定义在solver.hpp中，这是一个定义为枚举类型的数据类型，只有三个可能的值，分别对应了三种处理系统信号的方式：NONE(忽略信号什么都不做)/STOP(停止训练)/SNAPSHOT(保存当前的训练状态，继续训练)。在caffe.cpp中的train函数里Solver设置如何处理系统信号的代码为：

{% highlight cpp lineos %}
caffe::SignalHandler signal_handler(
      GetRequestedAction(FLAGS_sigint_effect),
      GetRequestedAction(FLAGS_sighup_effect));

solver->SetActionFunction(signal_handler.GetActionFunction());
{% endhighlight %}

FLAGS_sigint_effect和FLAGS_sighup_effect是通过gflags定义和解析的两个Command Line Interface的输入参数，分别对应遇到sigint和sighup信号的处理方式，如果用户不设定(大部分时候我自己就没设定)，sigint的默认值为"stop"，sighup的默认值为"snapshot"。`GetRequestedAction`函数会将string类型的FLAGS_xx转为SolverAction::Enum类型，并用来定义一个`SignalHandler`类型的对象signal_handler。我们可以看到这部分代码都依赖于`SignalHandler`这个类的接口，我们先来看看这个类都做了些什么：

{% highlight cpp lineos %}
// header file
class SignalHandler {
 public:
  // Contructor. Specify what action to take when a signal is received.
  SignalHandler(SolverAction::Enum SIGINT_action,
                SolverAction::Enum SIGHUP_action);
  ~SignalHandler();
  ActionCallback GetActionFunction();
 private:
  SolverAction::Enum CheckForSignals() const;
  SolverAction::Enum SIGINT_action_;
  SolverAction::Enum SIGHUP_action_;
};
// source file
SignalHandler::SignalHandler(SolverAction::Enum SIGINT_action,
                             SolverAction::Enum SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action) {
  HookupHandler();
}
void HookupHandler() {
  if (already_hooked_up) {
    LOG(FATAL) << "Tried to hookup signal handlers more than once.";
  }
  already_hooked_up = true;
  struct sigaction sa;
  sa.sa_handler = &handle_signal;
  // ...
}
static volatile sig_atomic_t got_sigint = false;
static volatile sig_atomic_t got_sighup = false;
void handle_signal(int signal) {
  switch (signal) {
  case SIGHUP:
    got_sighup = true;
    break;
  case SIGINT:
    got_sigint = true;
    break;
  }
}
ActionCallback SignalHandler::GetActionFunction() {
  return boost::bind(&SignalHandler::CheckForSignals, this);
}
SolverAction::Enum SignalHandler::CheckForSignals() const {
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SolverAction::NONE;
}
bool GotSIGINT() {
  bool result = got_sigint;
  got_sigint = false;
  return result;
}
bool GotSIGHUP() {
  bool result = got_sighup;
  got_sighup = false;
  return result;
}
// ActionCallback的含义
typedef boost::function<SolverAction::Enum()> ActionCallback;
{% endhighlight %}

`SignalHandler`这个类有两个数据成员，都是`SolverAction::Enum`类型的，分别对应sigint和sighup信号，在构造函数中，用解析FLAGS_xx得到的结果分别给两个成员赋值，然后调用了`HookupHandler`函数，这个函数的主要作用是定义了一个`sigaction`类型(应该是系统级别的代码)的对象sa，然后通过sa.sa_handler = &handle_signal来设置，当有遇到系统信号时，调用`handle_signal`函数来处理，而我们可以看到这个函数的处理很简单，就是判断一下当前的信号是什么类型，如果是sigint就将全局的static变量got_sigint变为true，sighup的处理类似。

在根据用户设置（或者默认值）的参数定义了signal_handler之后，solver通过`SetActionFunction`来设置了如何处理系统信号。这个函数的输入为signal_handler的`GetActionFunction`的返回值，根据上面的代码我们可以看到，`GetActionFunction`会返回signal_handler这个对象的CheckForSignals函数的地址(boost::bind的具体使用请参考boost官方文档)。而在`Solver`的`SetActionFunction`函数中只是简单的把`Solver`的一个成员action_request_function_赋值为输入参数的值，以当前的例子来说就是，solver对象的action_request_function_指向了signal_handler对象的CheckForSignals函数的地址。其中的ActionCallback是一个函数指针类型，指向了参数为空，返回值为SolverAction::Enum类型的函数(boost::function具体用法参考官方文档)。

总结起来，我们通过定义一个`SignalHandler`类型的对象，告知系统在遇到系统信号的时候回调`handle_signal`函数来改变全局变量got_sigint和got_sighup的值，然后通过`Solver`的接口设置了其遇到系统函数将调用signal_handler的Check函数，这个函数实际上就是去判断当前是否遇到了系统信号，如果遇到某个类型的信号，就返回我们之前设置的处理方式(`SolverAction::Enum`类型)。剩余的具体处理再交给`Solver`的其它函数，后面会具体分析。