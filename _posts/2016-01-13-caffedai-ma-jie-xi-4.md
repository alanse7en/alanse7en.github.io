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

* `Solver`的初始化（Register宏和构造函数）
* `SIGINT`和`SIGHUP`信号的处理
* `Solver::Solve()`具体实现
* `SGDSolver::ApplyUpdate`具体实现


## Solver的初始化（Register宏和构造函数）

{% highlight cpp %}
shared_ptr<caffe::Solver<float> >
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
{% endhighlight %}

caffe.cpp中的train函数中通过上面的代码定义了一个指向`Solver<float>`的shared_ptr。其中主要是通过调用`SolverRegistry`这个类的静态成员函数`CreateSolver`得到一个指向`Solver`的指针来构造shared_ptr类型的`solver`。而且由于C++多态的特性，尽管`solver`是一个指向基类`Solver`类型的指针，通过`solver`这个智能指针来调用各个成员函数会调用到各个子类(`SGDSolver`等)的函数。具体的过程如下面的流程图所示：

<figure>
  <img src="/images/solver_creator.png" alt="">
  <figcaption>Create solver</figcaption>
</figure>

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

上面的代码中，`Registry`这个函数（第5行）中定义了一个static的变量g_registry，这个变量是一个指向`CreatorRegistry`这个map类型的指针，然后直接返回，因为这个变量是static的，所以即使多次调用这个函数，也只会定义一个g_registry，而且在其他地方修改这个map里的内容，是存储在这个map中的。事实上各个Solver的register的过程正是往g_registry指向的那个map里添加以Solver的type为key，对应的Creator函数指针为value的内容。Register的过程如流程图所示：

<figure>
  <img src="/images/solver_register.png" alt="">
  <figcaption>Register Solver</figcaption>
</figure>

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

在sgd_solver.cpp(SGD Solver对应的cpp文件)末尾有上面第24行的代码，使用了`REGISTER_SOLVER_CLASS`这个宏，这个宏会定义一个名为`Creator_SGDSolver`的函数，这个函数即为`Creator`类型的指针指向的函数，在这个函数中调用了`SGDSolver`的构造函数，并将构造的这个变量得到的指针返回，这也就是Creator类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。然后在这个宏里又调用了`REGISTER_SOLVER_CREATOR`这个宏，这里分别定义了`SolverRegisterer`这个模板类的float和double类型的static变量，这会去调用各自的构造函数，而在`SolverRegisterer`的构造函数中调用了之前提到的`SolverRegistry`类的`AddCreator`函数，这个函数就是将刚才定义的`Creator_SGDSolver`这个函数的指针存到g_registry指向的map里面。类似地，所有的Solver对应的cpp文件的末尾都调用了这个宏来完成注册，在所有的Solver都注册之后，我们就可以通过之前描述的方式，通过g_registry得到对应的Creator函数的指针，并通过调用这个Creator函数来构造对应的Solver。Register和Create对应的流程图如下所示：

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

## `Solver::Solve()`具体实现

`Solve`函数实现了具体的网络的优化过程，下面我们来具体分析一下这部分的代码，分析见注释：

{% highlight cpp lineos %}
void Solver<Dtype>::Solve(const char* resume_file) {
// 检查当前是否是root_solver(多GPU模式下，只有root_solver才运行这一部分的代码)
  CHECK(Caffe::root_solver());
// 然后输出learning policy(更新学习率的策略)
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();
// requested_early_exit_`一开始被赋值为false，也就是现在没有要求在优化结束前退出
  requested_early_exit_ = false;
// 判断`resume_file`这个指针是否NULL，如果不是则需要从resume_file存储的路径里读取之前训练的状态
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
// 然后调用了'Step'函数，这个函数执行了实际的逐步的迭代过程
  Step(param_.max_iter() - iter_);
// 迭代结束或者遇到系统信号提前结束后，判断是否需要在训练结束之后snapshot
// 这个可以在solver.prototxt里设置
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
// 如果在`Step`函数的迭代过程中遇到了系统信号，且我们的处理方式设置为`STOP`，
// 那么`requested_early_exit_`会被修改为true，迭代提前结束，输出相关信息
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
// 判断是否需要输出最后的loss
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
// 判断是否需要最后Test
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}
{% endhighlight %}

下面继续分析具体的迭代过程发生的`Step`函数：

{% highlight cpp lineos %}
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  vector<Blob<Dtype>*> bottom_vec;
// 设置开始的迭代次数(如果是从之前的snapshot恢复的，那iter_等于snapshot时的迭代次数)和结束的迭代次数
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
// 输出的loss为前average_loss次loss的平均值，在solver.prototxt里设置，默认为1，
// losses存储之前的average_loss个loss，smoothed_loss为最后要输出的均值
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;
// 迭代
  while (iter_ < stop_iter) {
  // 清空上一次所有参数的梯度
    net_->ClearParamDiffs();
// 判断是否需要测试
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
    // 判断是否需要提前结束迭代
      if (requested_early_exit_) {
        break;
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    // 判断当前迭代次数是否需要显示loss等信息
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    Dtype loss = 0;
    // iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size，
    // 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size，这个loss是通过调用`Net::ForwardBackward`函数得到的
    // 这个设置我的理解是在GPU的显存不够的时候使用，比如我本来想把batch_size设置为128，但是会out_of_memory，
    // 借助这个方法，可以设置batch_size=32，iter_size=4，那实际上每次迭代还是处理了128个数据
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward(bottom_vec);
    }
    loss /= param_.iter_size();
    // 计算要输出的smoothed_loss，如果losses里还没有存够average_loss个loss则将当前的loss插入，如果已经存够了，则将之前的替换掉
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    // 输出当前迭代的信息
    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    // 执行梯度的更新，这个函数在基类`Solver`中没有实现，会调用每个子类自己的实现，后面具体分析`SGDSolver`的实现
    ApplyUpdate();
    // 迭代次数加1
    ++iter_;
    // 调用GetRequestedAction，实际是通过action_request_function_函数指针调用之前设置好(通过`SetRequestedAction`)的
    // signal_handler的`CheckForSignals`函数，这个函数的作用是
    // 会根据之前是否遇到系统信号以及信号的类型和我们设置(或者默认)的方式返回处理的方式
    SolverAction::Enum request = GetRequestedAction();
    // 判断当前迭代是否需要snapshot，如果request等于`SNAPSHOT`则也需要
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }
    // 如果request为`STOP`则修改`requested_early_exit_`为true，之后就会提前结束迭代
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      break;
    }
  }
}
{% endhighlight %}

## `SGDSolver::ApplyUpdate`具体实现

每一组网络中的参数的更新都是在不同类型的Solver自己实现的`ApplyUpdate`函数中完成的，下面我们就以最常用的SGD为例子来分析这个函数具体的功能：

{% highlight cpp lineos %}
template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  // GetLearningRate根据设置的lr_policy来计算当前迭代的learning rate的值
  Dtype rate = GetLearningRate();
  // 判断是否需要输出当前的learning rate
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  // 避免梯度爆炸，如果梯度的二范数超过了某个数值则进行scale操作，将梯度减小
  ClipGradients();
  // 对所有可更新的网络参数进行操作
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    // 将第param_id个参数的梯度除以iter_size，这一步的作用是保证实际的batch_size=iter_size*设置的batch_size
    Normalize(param_id);
    // 将正则化部分的梯度降入到每个参数的梯度中 
    Regularize(param_id);
    // 计算SGD算法的梯度(momentum等)
    ComputeUpdateValue(param_id, rate);
  }
  // 调用`Net::Update`更新所有的参数
  this->net_->Update();
}
{% endhighlight %}

下面我们继续具体分析一下`Normalize`/`Regularize`/`ComputeUpdateValue`的实现，我们均以CPU的代码为例子，GPU部分的处理原理是一样的：

#### Normalize

{% highlight cpp lineos %}
template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  // 如果iter_size的值为1，则不需要任何处理直接return
  if (this->param_.iter_size() == 1) { return; }
  // 通过net_返回所有可以学习的参数，是一个vector<shared_ptr<Blob<Dtype> > >
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  // 要乘以的系数等于1/iter_size
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // caffe_scal在/CAFFE_ROOT/src/caffe/util/math_functions.cpp中
    // 是blas的scale函数的一个封装，第一个参数是数据的个数，第二个参数是乘以的系数，
    // 第三个参数是数据的指针
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: { 
    // GPU代码略
  }
}
{% endhighlight %}

#### Regularize

{% highlight cpp lineos %}
template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  // 获取所有可以学习的参数的vector
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  // 获取所有的参数对应的weight_decay的vector
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  // 模型整体的weight_decay数值
  Dtype weight_decay = this->param_.weight_decay();
  // 获取正则化的类型：L1 或 L2
  string regularization_type = this->param_.regularization_type();
  // 实际的weight_decay等于整体模型的数值乘以具体每个参数的数值
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // 如果weight_decay不为0，则计算
    if (local_decay) {
      if (regularization_type == "L2") {
        // L2的梯度为diff_ = weight_decay*data_ + diff_
        // caffe_axpy的功能是 y = a*x + y
        // 第一个参数是数据的个数，第二个是上式的a，第三个是x的指针，第四个是y的指针
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        // L1的梯度为diff_ = diff_ + sign(data_)
        // temp_ = sign(data_)
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        // 将temp_加到diff_中 diff_ = weight_decay*temp_ + diff_
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
// GPU代码略
}
{% endhighlight %}

#### ComputeUpdatedValue

{% highlight cpp lineos %}
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  // 获取所有可以更新的参数的vector
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  // 获取所有参数对应的learning_rate的vector
  const vector<float>& net_params_lr = this->net_->params_lr();
  // 获取momentum数值
  Dtype momentum = this->param_.momentum();
  // 实际的learning_rate为全局的learning_rate乘以每个参数对应的learning_rate
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // 关于SGD的公式参考caffe官网tutorial的Solver部分
    // history_存储了上一次的梯度，下面这个函数：
    // history_ = learning_rate*diff_ + momentum*history
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    // 把当前的梯度拷贝给参数Blob的diff_
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
    // GPU代码略
  }
}
{% endhighlight %}

至此`Solver`主要的代码都已经分析完了，总结起来主要有：(1)solver_factory的register和create不同类型Solver的机制，(2)通过signal_handler来获取系统信号，并根据用户或默认的设置进行相应的处理，(3)`Solver::Solve`函数的具体实现的分析，(4)`SGDSolver::ApplyUpdate`函数的具体实现。前面三个部分都属于基类的，最后一个是SGDSolver这个子类的，如果用户想要实现自己的Solver类，也应该类似地去继承基类，并实现自己的`ApplyUpdate`函数，在代码的末尾通过register宏完成注册，便可以被成功的调用。