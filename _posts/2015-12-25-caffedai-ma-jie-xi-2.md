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

{% highlight cpp linenos %}
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

{% highlight cpp linenos %}
// AddressBook.proto
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
第1行的`package tutorial`类似于C++里的`namespace`，`message`可以简单的理解为一个`class`，`message`可以嵌套定义。每一个`field`除了一般的`int32`和`string`等类型外，还有一个属性来表明这个`field`是`required`,`optional`或者'repeated'。`required`的`field`必须存在，相对应的`optional`的就可以不存在，`repeated`的`field`可以出现0次或者多次。这一点对于`Google Protocol Buffer`的兼容性很重要，比如新版本的`AddressBook`添加了一个`string`类型的`field`，只有把这个`field`的属性设置为`optional`，就可以保证新版本的代码读取旧版本的数据也不会出错，新版本只会认为旧版本的数据没有提供这个`optional field`，会直接使用`default`。同时我们也可以定义`enum`类型的数据。每个`field`等号右侧的数字可以理解为在实际的binary encoding中这个`field`对应的key值，通常的做法是将经常使用的`field`定义为0-15的数字，可以节约存储空间（涉及到具体的encoding细节，感兴趣的同学可以看看官网的解释），其余的`field`使用较大的数值。

在定义好了data schema之后，需要使用`protoc compiler`来编译定义好的`proto`文件。常用的命令为：

> protoc  -I=/protofile/directory  --cpp_out=/output/directory  /path/to/protofile 

`-I`之后为`proto`文件的路径，`--cpp_out`为编译生成的`.h`和`.cc`文件的路径，最后是`proto`文件的路径。编译之后会生成`AddressBook.pb.h`和`AddressBook/pb.cc`文件，其中包含了大量的接口函数，用户可以利用这些接口函数获取和改变某个`field`的值。对应上面的data schema定义，有这样的一些接口函数：

{% highlight cpp linenos %}
// name
inline bool has_name() const;
inline void clear_name();
inline const ::std::string& name() const;
inline void set_name(const ::std::string& value);
inline void set_name(const char* value);
inline ::std::string* mutable_name();

// email
inline bool has_email() const;
inline void clear_email();
inline const ::std::string& email() const;
inline void set_email(const ::std::string& value);
inline void set_email(const char* value);
inline ::std::string* mutable_email();

// phone
inline int phone_size() const;
inline void clear_phone();
inline const ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >& phone() const;
inline ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >* mutable_phone();
inline const ::tutorial::Person_PhoneNumber& phone(int index) const;
inline ::tutorial::Person_PhoneNumber* mutable_phone(int index);
inline ::tutorial::Person_PhoneNumber* add_phone();
{% endhighlight %}