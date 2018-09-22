---
title: 行列式点过程学习笔记
date: 2018-09-14 20:17:29
categories: 数学
tags:
  - dpp
  - math
mathjax: true
photos: http://ojtdnrpmt.bkt.clouddn.com/blog/180914/hJk7lgJGJb.JPG
html: true
---
研究一下行列式点过程，这是一种广泛应用的确保diversity的一种数学方法
估计从此以后封面就用这种老E灵魂画风
DPP是结合了实分析、矩阵计算和概率计算的一种有效、优雅的算法，最广为流传的是Ben Taskar在2012作出的Determinantal point processes for machine learning，在videoslectures.net有一段半小时的视频，另外还有120页的pdf数学推导和一份250页的tutorial，可惜大神在2013年英年早逝，2017年在youtube上Wray Buntine教授根据这些材料有一段讲课视频，教授研究领域也是自动文摘，这份视频也值得推荐。
参考：
-	Determinantal point processes for machine learning ppt&pdf (Alex Kulesza , Ben Taskar)
-	Determinantal point processes (Laurent Decreusefond , Ian Flint , Nicolas Privault)
-	Determinantal Point Process and its Time-varying model (A/Prof Richard Yi Da Xu)
-	k-DPPs: Fixed-Size Determinantal Point Processes (Alex Kulesza , Ben Taskar)
-	On adding a list of numbers (and other one-dependent determinantal processes) (Alexei Borodin , Persi Diaconis , Jason Fulman)
-	Determinantal point processes (Alexei Borodin)

<!--more-->

# 一个例子：单依赖行列式点过程

## 例子
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180914/6k9chBkjFc.png?imageslim)
-	考虑上图所示一个简单的问题，52是左边一列数字相加之和，右边是对应累加到此行时和的个位数，例如(7+9)%10=6，所以9的右边是6，加点代表着从此行到下一行累加和会进位。
-	我们每次只加0到9这个范围内的数，那么可以得到一系列进位的序列，例如上图，在第1，2，4，7，8次加法时进位。考虑每次我们加的数是均匀分布在[0,9]的，那么这些进位是如何分布的？一般n次加法里会有几次进位？至少有两点我们从直觉上可以确定：
	-	一般n次加法里有一半的进位
	-	如果本次进位了，那么下一次就不太可能进位
-	我们进一步观察上图中的例子：
	-	如果某一行有点，也就是即将进位，那么此行右边的数到下一行会变少，也就是发生了下降（descent）
	-	如果左边一列的数字是相互独立且均匀分布的，那么右边一列也是独立且均匀分布
	-	综上，我们可以知道能从进位的分布推测出一个随机数字序列的下降模式（descent pattern）

## 从概率来描述进位和下降
-	定义集合$B={0,1,2,....,b-1}$，从中独立同分布抽取一个序列$B_1,B_2,...,B_n$，假如$B_i > B_{i+1}$，就说在位置i发生了下降，对应的$X_i$置为1，否则为0，所有下降的位置组成集合$D=\{i:X_i=1\}$
-	fact 1
$$
\forall i \in [n-1] \\
P(X_i=1) = \frac 12 - \frac{1}{2b} \\
= \frac {C_b^2}{b^2} \\
Var(X_i)=\frac 14 - \frac{1}{4b^2} \\
$$
-	fact 2
$$
\forall i,j \ with \ 1 \leq i < i+j \leq n \\
P(X_i=X_{i+1}=...=X_{i+j-1}=1) = \frac {C_b^{j+1}}{b^{j+1}} \\ 
Cov(X_i,X_{i+1})=-\frac {1}{12}(1-\frac{1}{b^2}) \\
$$
-	fact 3，${X_i}$的分布是稳定单依赖的
	-	稳定：是因为在有效范围内，$X_i$和$X_{i+j}$的分布一样，也就是与其位置无关。
	-	单依赖：只要不是相邻（距离大于1），则${X_i}$之间是独立的。
-	利用以上结论，我们可以发现进位的分布只和进位的基相关，且可以计算进位总个数的均值和方差
-	fact 4，k点相关性：在[n-1]里取一个大小为k的子集，则在这个子集内的$P(X_i=1)$为点过程$X_1,...,X_{n-1}$的k点相关性。k点相关性是描述一般点过程的基本单元,记作$\rho (A)$:
$$
\rho (A) = \prod _1 ^k [\frac{C_b^{a_{i+1}}}{b^{a_{i+1}}}] 
$$
-	fact 5，重要的来了，所有的$X_i$的联合分布可以用一个行列式乘一个系数表示，假如总共有k个$X_i$是1，那么行列式是k+1行k+1列的矩阵的行列式，假设这k个1的位置是$s_1,...,s_k$，且定义$s_0=0,s_{k+1}=n$，则矩阵中第i行第j列的值为：
$$
C_{s_{j+1}-s_i+b-1}^{b-1}
$$
-	系数是$\frac{1}{b^n}$，即
$$
p(X) = \frac{1}{b^n} det(C_{s_{j+1}-s_i+b-1}^{b-1})
$$
-	一个例子如下，进位的基为2，加8个数，在第一和第五个位置发生下降的情况的概率为0.03516：
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180915/IGF2j3CjAj.JPG)

## 来吧，行列式点过程
-	令X为一个有限集，在X上的一个点过程是指X上的$2^{|X|}$个子集的概率测度P，例如，$X={1,2,...,n-1}$，那么点过程就是记录在加上n个以b为基的数的过程中何时出现进位，也就是上一节中我们讨论的。P我们也可以通过相关函数$\rho$来明确：
$$
\rho (A) = P\{S:A \in S\}
$$
-	当一个点过程可以用核方法$K(x,y)$来表示时，就说这个点过程是行列式的：
$$
\rho (A) = det(K(x,y))_{x,y \in A}
$$
-	行列式是一个$|A|x|A|$大小的矩阵的行列式。
## 稳定单依赖过程-进位和下降模式，对称函数理论
## 混合例子
## 泛化：从加法到乘法
## 单依赖行列式家族