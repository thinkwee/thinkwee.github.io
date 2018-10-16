---
title: Logistic回归与最大熵
date: 2018-10-14 20:38:59
tags: [logistic regression,math,machinelearning]
categories: 机器学习
mathjax: true
html: true
---

翻译John Mount的*The equivalence of logistic regression and maximum entropy models* 一文 

此文从softmax回归推导出了平衡条件，并证明在广义线性回归模型的平衡条件约束下，满足最大熵模型的非线性激活函数就是sigmoid(softmax、logistic)。

<!--more-->  

# 明确符号
- n维特征，m个样本，$x(i)_j$表示第i个样本第j维特征，讨论多分类情况，输出分类$y(i)$有k类，映射概率函数$\pi$从$R^n$映射到$R^k$，我们希望$\pi(x(i))_{y(i)}$尽可能大。
- 指示函数$A(u,v)$，当$u==v$时为1，否则为0

# Logistic回归
$$
\pi(x)_1 = \frac{e^{\lambda x}}{1+e^{\lambda x}} \\
\pi(x)_2 = 1 - \pi(x)_1\\
$$
- 其中要学习到的参数$\lambda$为$R^n$

# Softmax回归
$$
\pi(x)_v = \frac{e^{\lambda _v x}} {\sum _{u=1}^k e^{\lambda _u x}}
$$
- $\lambda$为$R^{k \* n}$

# 求解softmax
- 当使用softmax或者logistic作为非线性函数时，它们存在一个很好的求导的性质，即导函数可以用原函数表示
$$
\frac {\partial \pi (x)_v}{\partial \lambda _{v,j}} = x_j  \pi (x)_v (1-\pi (x)_v) \\
\frac {\partial \pi (x)_v}{\partial \lambda _{u,j}} = -x_j \pi (x)_v \pi (x)_u \ where \  u \neq v \\
$$ 
- 现在我们可以定义目标函数，即希望$\pi$函数输出的正确类别概率最大（最大似然），并定义最优化得到的$\lambda$：
$$
\lambda = argmax \sum _{i=1}^m log (\pi (x(i))_{y(i)}) \\
= argmax f(\lambda) \\
$$

# 平衡等式
- 对上面的目标函数求导并令导函数为0：
$$
\frac {\partial f(\lambda)}{\partial \lambda _{u,j}} = \sum _{i=1，y(i)=u}^m x(i)_j - \sum _{i=1}^m x(i)_j \pi (x(i))_u =0 \\
$$
- 这样我们就得到一个重要的平衡等式(Balance Equation)：
$$
\ \  for \ all \ u,j \\
\sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
$$
- 分析这个等式：
	- 大白话：我们希望得到这么一个映射函数$\pi$，对某一维(j)特征，用所有样本被映射函数归为第u类的概率加权所有样本的特征值之和，等于第u类内所有样本的特征值之和。显然，最好的情况就是左右两个累加式内的元素完全一样，只有第u类的样本被累加，且第u类样本被映射函数归为第u类的概率为1，其他类样本被归为第u类样本的概率为0.
	- 但是，这个等式非常的宽松，它只要求两个和式相同，并不要求每一个元素相同，而且这个式子没有显示的写出映射函数的表达式，任何满足该式的非线性映射都有可能称为映射函数。
	- 用公式表达，就是
	$$
	\sum _{i=1}^m A(u,y(i)) x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
	\pi (x(i))_u \approx A(u,y(i)) \\
	$$

# 由最大熵推出softmax
- 上面说到了平衡等式并没有要求映射函数的格式，那么为什么我们选择了softmax？换句话，什么条件下能从平衡等式的约束推出非线性映射为softmax？
- 答案是最大熵。我们现在回顾一下$\pi$需要满足的条件：
	- 平衡等式（即这个$\pi$能拟合数据）：
	$$
	\ \  for \ all \ u,j \\
	\sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
	$$
	- $\pi$的输出得是一个概率：
	$$
	\pi (x)_v \geq 0 \\
	\sum _{v=1}^k \pi (x)_v = 1 \\
	$$
- 根据最大熵原理，我们希望满足上述约束条件的$\pi$能够具有最大的熵:
$$
\pi = argmax \ Ent(\pi) \\
Ent(\pi) = \sum_{v=1}^k \sum _{i=1}^m \pi (x(i))_v log (\pi (x(i))_v) \\
$$
- 最大熵可以从两个角度理解：
	- 最大熵也就是最小困惑度，在无监督模型中我们经常用困惑度衡量概率模型的效果，根据奥卡姆剃刀原则，在多个具有相同效果的模型中复杂程度小的模型具有更好的泛化能力，困惑度是一种衡量复杂程度的指标
	- 约束条件是我们的模型已知的需要满足、需要拟合的部分，剩下的部分是未知的部分，没有规则或者数据指导我们分配概率，那该怎么办？在未知的情况下就应该均匀分配概率给所有可能，这正是对应了最大熵的情况
- 现在问题已经形式化带约束条件的最优化问题，利用拉格朗日乘子法求解即可。这里有一个trick，原文中说如果直接考虑概率的不等条件就有点复杂，需要使用KTT条件，这里先不考虑，之后如果求出的$\pi$满足不等式条件的话就可以跳过了（事实也正是如此）。

$$
L = \sum _{j=1}^n \sum _{v=1}^k \lambda _{v,j} (\sum _{i=1}^m \pi (x(i))_v x(i)_j - A(v,y(i)) x(i)_j) +\sum _{v=1}^k \sum _{i=1}^m \beta _i (\pi (x(i))_v -1) -\sum _{v=1}^k \sum _{i=1}^m \pi(x(i))_v log(\pi (x(i))_v) \\
$$
- 这里又有一个trick，本来应该对所有参数求导，这里我们先对$\pi (x(i))_u$求导令其为0可得：
$$
\pi (x(i))_u = e^{\lambda _u x(i) + \beta _i -1}
$$
- 再考虑等式约束条件（概率之和为1），这样就不用再对$\beta$求导：
$$
\sum _{v=1}^k e^{\lambda _v x(i) + \beta _i -1} = 1 \\
e^{\beta} = \frac {1}{\sum _{v=1}^k e^{\lambda _v x(i) - 1}} \\
$$
- 回代可得：
$$
\pi (x)_u = \frac {e^{\lambda _u}x}{\sum _{v=1}^k e^{\lambda _v}x}
$$

# 求解参数
- 从推出平衡等式的时候可以看到，我们需要解$n \* k$个方程来得到$n \* k$个参数$\lambda$，或者在最大熵的拉格朗日方程里对$n \* k$个$\lambda$求偏导，因为$\pi$是$\lambda$的非线性函数，这两种求解方法比较困难，但是我们可以求导计算这些等式的雅各比方程（或者说是目标函数的Hessian矩阵），之后我们就可以用某种牛顿法、Fisher Scoring或者迭代的方法求解$\lambda$

