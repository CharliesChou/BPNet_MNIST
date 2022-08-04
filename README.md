# BPNet_MNIST
使用C++语言实现的BP神经网络，并且验证了MNIST数据集。

程序说明：

1.该程序的正向传递函数为：forward_propagation()；

2.该程序的反向传递函数可以使用两种方法：

    2.1：方法一：使用 backward_propagation() ；

    2.2：方法二：先使用 calcu_delta() 函数计算出偏导 delta，紧接着使用 backward_propagation_use_delta();

    2.3：说明：方法一代码比较容易阅读，按照反向传递公式一次性更新，但运行速度慢，因为在更新过程中存在大量重复运算；方法二运行速度快，将需要重复运算的delta偏导值先运算出来保存在数组中，然后再利用数组中的值来更新各个参数，减少了运算次数，速率可以提高3-4倍；可以在训练中使用其中一种方法即可。

3.show_pic_info()方法用来显示一个图片的信息，并且进行了重载。

4.程序代码（MNIST数据集请自行下载并且放在源文件同目录下），编辑器为codeblocks，编译器为codeblocks自带的GUN GCC Complier
版权声明：本文为CSDN博主「Charles Chou」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_43249548/article/details/126148304
