---
title: 凸优化概论笔记
date: 2018-07-03 15:17:27
tags: [convex optimization ,math]
categories: 数学
mathjax: true
html: true
photos: http://ojtdnrpmt.bkt.clouddn.com/blog/180703/15D61BF7mL.png?imageslim
---
***
凸优化笔记
Convex Optimization Overview
Zico Kolter (updated by Honglak Lee)

<!--more-->

# 引入
-	机器学习中经常需要解决最优化问题，找到全局最优解很难，但对于凸优化问题，我们通常可以有效找到全局最优解

# 凸集
-	定义
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/e78fGmcAgj.png?imageslim)
-	直观上理解及对于集合中两个元素，他们的特殊线性组合（凸组合)$\theta x+(1-\theta)y$依然属于该集合
-	常见的二维形式即图中两点连线依然在图中，不会越界
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/m18774d3Gk.png?imageslim)

## 例子
-	所有的实数n维空间
-	非负象限
-	范数域
-	仿射子空间和多面体
-	凸集的交集，注意并集一般不成立
-	半正定矩阵
-	以上这些例子，他们元素的凸组合依然符合原始集合的性质

# 凸函数
-	定义
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/De00K79cF6.png?imageslim)
-	直观上，凸函数即函数上两点连线，两点之间的函数曲线在直线下方
	-	如果严格在直线下方而不是会有相切，则为严格凸性
	-	如果在直线上方则为凹性
	-	严格凹性同理

## 凸性一阶条件

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/C4cAECJLhb.png?imageslim)
-	前提时函数可微
-	即在函数上任意一点做切线，切线在函数的下方

## 凸性二阶条件

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/elbdAFA9aH.png?imageslim)
-	前提函数二阶可微，即Hessian矩阵在所有定义域内存在

## Jensen不等式
-	由凸函数的定义，将凸组合从二维扩展到多维，进而扩展到连续的情况，可以得到3种不等式
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/AE34lI2FFj.png?imageslim)
-	从概率密度的角度改写为
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/c2GgbjigdA.png?imageslim)
-	即Jensen不等式

## 分段集
-	一种特别的凸集称为$\alpha$分段集，定义如下
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/L3fjcKl5if.png?imageslim)
-	可以证明该集合也是凸集
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/8ff5831B1d.png?imageslim)

## 凸函数例子
-	指数函数
-	负对数函数
-	线性函数。特别的时线性函数的Hessian矩阵为0，0矩阵机试正半定也是负半定，因此线性函数既是凸函数也是凹函数。
-	二次函数
-	范数
-	权值非负情况下，凸函数的加权和

# 凸优化问题
-	变量属于凸集，调整变量使得凸函数值最小。
-	变量术语凸集这一条件可以进一步明确为凸函数的不等式条件和线性函数的等式条件，等式条件可以理解为大于等于和小于等于的交集，即凸函数和凹函数的交集，这一交集只有线性函数满足。
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/hD8Ka52AcG.png?imageslim)
-	凸函数的最小值即最优值，最优值可以取正负无穷

## 凸问题中的全局最优性
-	可行点的局部最优条件和全局最优条件，略过
-	对于凸优化问题，所有的局部最优点都是全局最优点
	证明：反证，假如x是一个局部最优点而不是全局最优点，则存在点y函数值小于点x。根据局部最优条件的定义，x的邻域内不存在点z使得函数值小于点x。假设邻域范围为R，我们取z为x和y的凸组合：
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/DCd7l9lDbl.png?imageslim)
	则可以证明z在x的邻域内
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/j4D5hg8l4J.png?imageslim)
	并且z的函数值小于x，推出矛盾。且由于可行域为凸集，x和y为可行点则z一定为可行点。
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/C0l3BC192a.png?imageslim)


## 凸优化问题的特殊情况
-	对于一些特殊的凸优化问题，我们定制了特别的算法来解决大规模的问题
-	线性编程（LP）：f和g都是线性函数
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/Ci58g01eF5.png?imageslim)
-	二次编程（QP）：g均为线性函数，f为凸二次函数
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/HkiCClD4eh.png?imageslim)
-	二次约束的二次编程（QCQP）：f和所有的g都是凸二次函数
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/l45kgmJ8j1.png?imageslim)
-	半定编程（SDP）
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180703/Kce3mdA3FG.png?imageslim)
-	这四种类型依次越来越普遍，QCQP是SDP的特例，QP是QCQP的特例，LP是QP的特例

## 例子
-	SVM
-	约束最小二乘法
-	罗杰斯特回归的最大似然估计

## 实现：使用CVX的线性SVM
-	matlab代码解读
