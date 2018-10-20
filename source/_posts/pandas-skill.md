---
title: pandas 数据处理基础
date: 2017-02-04 21:02:39
tags: [math,machinelearning,python,code]
categories: Python
---
***
以泰坦尼克号的数据为例介绍一下前期对数据的基础操作。

<!--more-->

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
![i0Tn41.jpg](https://s1.ax1x.com/2018/10/20/i0Tn41.jpg)

# 数据概览
-	describe 显示整体数据常见属性
```python
print(train.describe())
```
![i0TP3V.jpg](https://s1.ax1x.com/2018/10/20/i0TP3V.jpg)
-	head tail 显示首尾一些数据
```python
print(train.head(5))
print(train.tail(3))
```
![i0TFjU.jpg](https://s1.ax1x.com/2018/10/20/i0TFjU.jpg)
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
![i0TAuF.jpg](https://s1.ax1x.com/2018/10/20/i0TAuF.jpg)
-	sort：可以按索引或者值进行排序，axis选择维度(行还是列),ascending选择升序或者降序,Nan永远排在最后，无论升序降序
```python
print(train.sort_index(axis=0,ascending=True))
print(train.sort_values(by="Age",ascending=False))
```
![i0TEB4.jpg](https://s1.ax1x.com/2018/10/20/i0TEB4.jpg)
![i0TmNR.jpg](https://s1.ax1x.com/2018/10/20/i0TmNR.jpg)

# 数据选择
-	按照标签选择，选择列，行切片
```python
print(train['Age'])
print(train[0:9])
```
![i0TVHJ.jpg](https://s1.ax1x.com/2018/10/20/i0TVHJ.jpg)
![i0TeE9.jpg](https://s1.ax1x.com/2018/10/20/i0TeE9.jpg)
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
![i0TK9x.jpg](https://s1.ax1x.com/2018/10/20/i0TK9x.jpg)
![i0TM36.jpg](https://s1.ax1x.com/2018/10/20/i0TM36.jpg)
![i0TljO.jpg](https://s1.ax1x.com/2018/10/20/i0TljO.jpg)

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
![i0TcUs.jpg](https://s1.ax1x.com/2018/10/20/i0TcUs.jpg)
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
![i0TQgK.jpg](https://s1.ax1x.com/2018/10/20/i0TQgK.jpg)
```python
#写函数对年龄列进行分类
def judge(row):
    if pd.isnull(row['Age']) ==True:
        return 'unknown'
    return 'youngth' if row['Age']<18 else 'adult'
print(train.apply(judge,axis=1))
```
![i07jFs.jpg](https://s1.ax1x.com/2018/10/20/i07jFs.jpg)
# 数据透视表
-	自选分类和值进行数据透视，比如按照pclass和sex分类，统计age和fare的平均值
```python
print(train.pivot_table(index=["Pclass","Sex"], values=["Age", "Fare"], aggfunc=np.mean))
```
![i0T3uD.jpg](https://s1.ax1x.com/2018/10/20/i0T3uD.jpg)



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
![i0T8De.jpg](https://s1.ax1x.com/2018/10/20/i0T8De.jpg)
