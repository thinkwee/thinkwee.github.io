---
title: Tensorflow学习
date: 2018-01-10 09:15:39
tags:
	- tensorflow
	- machinelearning
	- code
categories:
	- 机器学习
mathjax: true
photo: http://ojtdnrpmt.bkt.clouddn.com/blog/180307/el4J0a52f0.jpg?imageslim
---

-	tensorflow 学习
-	应用于深度学习
-	Stanford CS20SI

<!--more-->

# CS20SI
-	课程代码：[Tensorflow_Learn](https://github.com/thinkwee/Tensorflow_Learn)
-	主要是讲解session、op、graph等基本概念
-	介绍了如何处理数据，计算梯度，使用summary记录中间数据并在tensorboard中可视化
-	B站上有17年的视频：[CS20SI - Tensorflow for Deep Learning Research](https://www.bilibili.com/video/av15898988/)
-	有一些关于图像风格迁移的略过了，待补充

# TF中的坑
-	记录自己错过的地方
-	int() argument must be a string, a bytes-like object or a number, not 'Tensor'
 -	假如一些函数的参数指定为整型,不能为tensor，则传整型的tensor也不行
 -	解决：调用eval传值：tensor.Variable.eval()
-	Attempting to use uninitialized value
 -	run模型之前先sess.run(tf.global_variables_initializer())
 -	如果是回复模型出现错误，说明恢复之前先build graph了，且其中包含了有初始值的tf.variable，最好是恢复整个图和张量，而不是建一个一样的图再恢复张量
-	提示nested tuple相关错误
 -	搜索tf.unstack()查看其用法
 -	用for i in variable展开
-	tensor object is not iterable
 -	tensor对象是不可遍历的，不能用for i in tensor 或者tensor[i]
 -	解决：尝试调用其函数或者使用np.function()获取可遍历的值
-	在seq2seq中使用LuongAttention报错
 -	因为我是从无注意力的seq2seq中直接加了attention_wrapper才会报错
 -	解码端的初始状态应先置0，再复制成编码端最终输出的状态
 ```Python
 decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state) 
 ```


