---
layout: post
title: "Caffe代码解析(2)"
modified: 2015-12-25 20:24:49 +0800
tags: [Caffe,Deep Learning,C++]
image:
  feature: abstract-17.png
  credit: 
  creditlink: http://www.rafaelgrossmann.com/deep-really-deep-learning/ 
comments: true
share: true
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
第2行的`package tutorial`类似于C++里的`namespace`，`message`可以简单的理解为一个`class`，`message`可以嵌套定义。每一个`field`除了一般的`int32`和`string`等类型外，还有一个属性来表明这个`field`是`required`,`optional`或者'repeated'。`required`的`field`必须存在，相对应的`optional`的就可以不存在，`repeated`的`field`可以出现0次或者多次。这一点对于`Google Protocol Buffer`的兼容性很重要，比如新版本的`AddressBook`添加了一个`string`类型的`field`，只有把这个`field`的属性设置为`optional`，就可以保证新版本的代码读取旧版本的数据也不会出错，新版本只会认为旧版本的数据没有提供这个`optional field`，会直接使用`default`。同时我们也可以定义`enum`类型的数据。每个`field`等号右侧的数字可以理解为在实际的binary encoding中这个`field`对应的key值，通常的做法是将经常使用的`field`定义为0-15的数字，可以节约存储空间（涉及到具体的encoding细节，感兴趣的同学可以看看官网的解释），其余的`field`使用较大的数值。

类似地在`caffe/src/caffe/proto/`中有一个`caffe.proto`文件，其中对layer的部分定义为：

{% highlight cpp linenos %}
message LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the layer type
  repeated string bottom = 3; // the name of each bottom blob
  repeated string top = 4; // the name of each top blob
//  other fields
}
{% endhighlight %}

在定义好了data schema之后，需要使用`protoc compiler`来编译定义好的`proto`文件。常用的命令为：

> protoc  -I=/protofile/directory  --cpp_out=/output/directory  /path/to/protofile 

`-I`之后为`proto`文件的路径，`--cpp_out`为编译生成的`.h`和`.cc`文件的路径，最后是`proto`文件的路径。编译之后会生成`AddressBook.pb.h`和`AddressBook/pb.cc`文件，其中包含了大量的接口函数，用户可以利用这些接口函数获取和改变某个`field`的值。对应上面的data schema定义，有这样的一些接口函数：

{% highlight cpp linenos %}
// name
inline bool has_name() const;
inline void clear_name();
inline const ::std::string& name() const;  //getter
inline void set_name(const ::std::string& value);  //setter
inline void set_name(const char* value);  //setter
inline ::std::string* mutable_name();

// email
inline bool has_email() const;
inline void clear_email();
inline const ::std::string& email() const; //getter
inline void set_email(const ::std::string& value);  //setter
inline void set_email(const char* value);  //setter
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

每个类都有对应的`setter`和`getter`，因为`phone`是`repeated`类型的，所以还多了通过`index`来获取和改变某一个元素的`setter`和`getter`，`phone`还有一个获取数量的`phone_size`函数。

官网上的tutorial是通过`bool ParseFromIstream(istream* input);`来从binary的数据文件里解析数据，为了更好地说明Caffe中读取数据的方式，我稍微修改了代码，使用了和`Caffe`一样的方式通过`TextFormat::Parse(ZeroCopyInputStream* input, Message* output);`来解析文本格式的数据。具体的代码如下：

{% highlight cpp lineos %}
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "addressBook.pb.h"

using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

// Iterates though all people in the AddressBook and prints info about them.
void ListPeople(const tutorial::AddressBook& address_book) {
  for (int i = 0; i < address_book.person_size(); i++) {
    const tutorial::Person& person = address_book.person(i);

    cout << "Person ID: " << person.id() << endl;
    cout << "  Name: " << person.name() << endl;
    if (person.has_email()) {
      cout << "  E-mail address: " << person.email() << endl;
    }

    for (int j = 0; j < person.phone_size(); j++) {
      const tutorial::Person::PhoneNumber& phone_number = person.phone(j);

      switch (phone_number.type()) {
        case tutorial::Person::MOBILE:
          cout << "  Mobile phone #: ";
          break;
        case tutorial::Person::HOME:
          cout << "  Home phone #: ";
          break;
        case tutorial::Person::WORK:
          cout << "  Work phone #: ";
          break;
      }
      cout << phone_number.number() << endl;
    }
  }
}

// Main function:  Reads the entire address book from a file and prints all
//   the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " ADDRESS_BOOK_FILE" << endl;
    return -1;
  }

  tutorial::AddressBook address_book;

  {
    // Read the existing address book.
    int fd = open(argv[1], O_RDONLY);
    FileInputStream* input = new FileInputStream(fd);
    if (!google::protobuf::TextFormat::Parse(input, &address_book)) {
      cerr << "Failed to parse address book." << endl;
      delete input;
      close(fd);
      return -1;
    }
  }

  ListPeople(address_book);

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
{% endhighlight %}

读取和解析数据的代码：

{% highlight cpp linenos %}
int fd = open(argv[1], O_RDONLY);
FileInputStream* input = new FileInputStream(fd);
if (!google::protobuf::TextFormat::Parse(input, &address_book)) {
  cerr << "Failed to parse address book." << endl;
}
{% endhighlight %}

这一段代码将input解析为我们设计的数据格式，写入到`address_book`中。之后再调用`ListPeople`函数输出数据，来验证数据确实是按照我们设计的格式来存储和读取的。`ListPeople`函数中使用了之前提到的各个`getter`接口函数。

{% highlight cpp linenos %}
# ExampleAddressBook.prototxt
person {
  name: "Alex K"
  id: 1
  email: "kongming.liang@abc.com"
  phone {
    number: "+86xxxxxxxxxxx"
    type: MOBILE
  }
}

person {
  name: "Andrew D"
  id: 2
  email: "xuesong.deng@vipl.ict.ac.cn"
  phone {
    number: "+86xxxxxxxxxxx"
    type: MOBILE
  }
  phone {
    number: "+86xxxxxxxxxxx"
    type: WORK
  }
}
{% endhighlight %}

上面的文件的解析结果如图所示：

<figure>
  <img src="/images/listPerson.png" alt="">
</figure>