---
title: pandas 数据处理基础
date: 2017-02-04 21:02:39
tags: [math,machinelearning,python,code]
categories: Python
---
***
以泰坦尼克号的数据为例介绍一下前期对数据的基础操作。
数据在这：
# 引入库
```python
import csv as csv 
import pandas as pd
import numpy as np
```
# 读取文件
```python
train = pd.read_csv(r"文件目录") 
```
此时数据的样式是：
![](http://ojtdnrpmt.bkt.clouddn.com/blog/20170205/103436937.JPG)


<!--more-->

# 数据概览
-	describe 显示整体数据常见属性
```python
print(train.describe())
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222443347.JPG)
-	head tail 显示首尾一些数据
```python
print(train.head(5))
print(train.tail(3))
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222504633.JPG)
-	index：索引，默认自建整型索引；columns：列；values：数据数值
```python
print(train.index)
print(train.columns)
print(train.values)
```
# 数据操作
-	T：数据的转置
```python
print(train.T)
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222544174.JPG)
-	sort：可以按索引或者值进行排序，axis选择维度(行还是列),ascending选择升序或者降序,Nan永远排在最后，无论升序降序
```python
print(train.sort_index(axis=0,ascending=True))
print(train.sort_values(by="Age",ascending=False))
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222551913.JPG)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222617681.JPG)

# 数据选择
-	按照标签选择，选择列，行切片
```python
print(train['Age'])
print(train[0:9])
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222634103.JPG)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222640068.JPG)
-	利用loc自由选择某些行某些列，可以用at替代
```python
print(train.loc[train.index[4:6]])
print(train.loc[:,['Age','Fare']])
print(train.loc[3:5,['Age','Fare']])
print(train.loc[4,'Age'])
print(train.at[4,'Age'])
```
-	利用iloc按照位置进行选择
```python
print(train.iloc[5])
print(train.iloc[3:5,2:4])
print(train.iloc[[1,2,4],[2,5]])
print(train.iloc[3,3])
```
-	布尔选择
```python
print( train[ (train['Age']>40) & (train['Age']<50) ] )
print(train[train['Parch'].isin([1,2])])
print(train[pd.isnull(train['Age'])==True])
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222658806.JPG)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222706679.JPG)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222715106.JPG)

# 缺失值处理
-	利用reindex选择部分数据进行拷贝，并进行缺失值处理。一些函数会自动过滤掉缺失值，比如mean()
```python
train1=train.reindex(index=train.index[0:5],columns=['PassengerId']+['Age']+['Sex'])#选择前5行，只取选定的三列
print(train1)
print(train1.dropna(axis=0)) #删除存在nan值的行
print(train1.dropna(subset=['Age','Sex'])) #删除年龄性别列中存在nan值的行
print(pd.isnull(train1)) #nan值改为true，其余值改为false
print(train1.fillna(value=2333)) #缺失值替换为2333
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222723080.JPG)
# 应用函数
-	可以自己写函数并应用到数据的行或者列，通过axis参数选择行列
```python
#写函数统计包含nan值的行数
def null_count(column):
    column_null=pd.isnull(column)
    null=column[column_null == True]
    return len(null)
print(train.apply(null_count))
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222730409.JPG)
```python
#写函数对年龄列进行分类
def judge(row):
    if pd.isnull(row['Age']) ==True:
        return 'unknown'
    return 'youngth' if row['Age']<18 else 'adult'
print(train.apply(judge,axis=1))
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222741327.JPG)
# 数据透视表
-	自选分类和值进行数据透视，比如按照pclass和sex分类，统计age和fare的平均值
```python
print(train.pivot_table(index=["Pclass","Sex"], values=["Age", "Fare"], aggfunc=np.mean))
```
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170204/222751776.JPG)



# 数据合并

- 数据合并的一些操作，待补全

  ```Python
  import pandas as pd
  data1 = pd.DataFrame({'level':['a','b','c','d'],
                   'numeber':[1,3,5,7]})
   
  data2=pd.DataFrame({'level':['a','b','c','e'],
                   'numeber':[2,3,6,10]})

  print("merge:\n",pd.merge(data1,data2),"\n")

  data3 = pd.DataFrame({'level1':['a','b','c','d'],
                   'numeber1':[1,3,5,7]})
  data4 = pd.DataFrame({'level2':['a','b','c','e'],
                   'numeber2':[2,3,6,10]})
  print("merge with left_on,right_on: \n",pd.merge(data3,data4,left_on='level1',right_on='level2'),"\n")

  print("concat: \n",pd.concat([data1,data2]),"\n")

  data3 = pd.DataFrame({'level':['a','b','c','d'],
                   'numeber1':[1,3,5,np.nan]})
  data4=pd.DataFrame({'level':['a','b','c','e'],
                   'numeber2':[2,np.nan,6,10]})
  print("combine: \n",data3.combine_first(data4),"\n")
  ```
  ![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170212/214643504.JPG)
