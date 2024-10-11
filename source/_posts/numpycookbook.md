---

title: Numpy Cookbook
date: 2017-01-23 20:12:40
tags: [math,machine learning,python,code]
categories: Python

---

Cookbook网址：[Numpy Cookbook](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
Numpy的一些语法查询和总结
持续更新

<!--more-->

- shape()
  shape是numpy函数库中的方法，用于查看矩阵或者数组的维素
  shape(array) 若矩阵有m行n列，则返回(m,n)
  array.shape[0] 返回矩阵的行数m，参数为1的话返回列数n

- tile()
  tile是numpy函数库中的方法，用法如下:
  tile(A,(m,n))  将数组A作为元素构造出m行n列的数组

- sum()
  sum()是numpy函数库中的方法
  array.sum(axis=1)按行累加，axis=0为按列累加

- argsort()
  argsort()是numpy中的方法，得到矩阵中每个元素的排序序号 
  A=array.argsort()  A[0]表示排序后 排在第一个的那个数在原来数组中的下标

- dict.get(key,x)
  Python中字典的方法，get(key,x)从字典中获取key对应的value，字典中没有key的话返回0

- sorted()

- numpy中有min()、max()方法，用法如下
  array.min(0)  返回一个数组，数组中每个数都是它所在列的所有数的最小值
  array.min(1)  返回一个数组，数组中每个数都是它所在行的所有数的最小值

- listdir('str')
  strlist=listdir('str')  读取目录str下的所有文件名，返回一个字符串列表

- split()
  python中的方法，切片函数
  string.split('str')以字符str为分隔符切片，返回list

- zeros
  a=np.zeros((m,n), dtype=np.int) #创建数据类型为int型的大小为m*n的零矩阵