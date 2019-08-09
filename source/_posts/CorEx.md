---
title: Correlation Explaination 学习笔记
date: 2019-07-29 11:17:11
categories: 机器学习
tags:
  - corex
  - machine learning
  -	topic model
mathjax: true
html: true
---
***
CorEx(Correlation Explaination)的相关笔记。

<!--more-->

![etsOld.gif](https://s2.ax1x.com/2019/07/31/etsOld.gif)

# 概述
-	Correlation Explaination是一类表示学习方法，可用于主题模型，与LDA具有相似的结果但其处理过程完全不同。Correlation Explaination不对数据的生成做任何结构上的先验假设，而是类似于信息增益，用Total Correlation之差来找出最能explain数据的Correlation的主题。 其中一种快速计算方法就简写为CorEx。
-	为了方便起见，下文都是用LDA中的概念来类比CorEx中的概念，包括背景是文档的主题建模，主题是一个离散随机变量，一篇文档会包含多个主题等等。

# Discovering Structure in High-Dimensional Data Through Correlation Explanation
## 定义Total Correlation
-	定义$X$为一离散随机变量，则其熵为
$$
H(X) \equiv \mathbb{E}_{X}[-\log p(x)]
$$
-	两个随机变量之间的互信息定义为
$$
I(X_1 : X_2) = H\left(X_{1}\right)+H\left(X_{2}\right)-H\left(X_{1}, X_{2}\right)
$$
-	我们定义Total Correlation（或者叫多元互信息multivariate mutual information）为
$$
T C\left(X_{G}\right)=\sum_{i \in G} H\left(X_{i}\right)-H\left(X_{G}\right)
$$
-	其中$G$是$X$的一个子集。只管来看就是子集中每一个随机变量熵之和减去子集的联合熵。当G中只有两个变量时，TC等价于两个变量的互信息。
-	为了更方便理解，TC还可以写成KL散度的形式
$$
T C\left(X_{G}\right)=D_{K L}\left(p\left(x_{G}\right) \| \prod_{i \in G} p\left(x_{i}\right)\right)
$$
-	也就是说TC可以看成联合分布和边缘分布累乘之间的KL散度，那么当TC为0时，KL散度为0，联合分布等于边缘分布累乘，也就意味着数据内部的相关性为0，变量之间相互独立，联合分布可以factorize为边缘分布之积。
-	接着我们定义conditional TC
$$
T C(X | Y)=\sum_{i} H\left(X_{i} | Y\right)-H(X | Y)
$$
-	那么我们就可以用TC与条件TC之差来衡量某一条件（变量）对于数据的correlation的贡献，原文写的是measure the extent to which $Y$ explains the correlations in $X$
$$
T C(X ; Y) \equiv T C(X)-T C(X | Y)=\sum_{i \in \mathbb{N}_{n}} I\left(X_{i} : Y\right)-I(X : Y)
$$
-	$T C(X ; Y)$最大时，$T C(X | Y)$为0，也就是已知$Y$时$X$的联合分布可分解，也就说明$Y$ explains all the correlation in $X$。我们认为好的主题应当是文档的一种表示，其解释的文档Total Correlation应该最大。
-	现在我们就可以把$Y$看成时解释$X$的一个隐变量，也就是主题，接下来我们就要求出主题。在LDA中，主题明确定义为词概率分布，然而在CorEx，我们通过$p(Y|X)$来定义主题，也就是说只将其定义为一个能够影响$X$的离散随机变量，取值范围有$k$种可能，而不像LDA定义为$|V|$种取值可能。
-	LDA通过迭代，不断更新为每个词分配的主题，从而间接得到文档的主题分布和主题的词分布。而CorEx则不一样，无论文档还是词都会计算一个主题分布。CorEx根据公式不断更新每个主题的概率$p(y_j)$，每个词的主题分布$p(y_j|x_i)$，词到主题子集合的分配矩阵$\alpha$，以及每篇文档的主题分布$p(y_j|x)$	
-	初始化时，我们随机设定$\alpha$以及文档的主题分布$p(y|x)$
-	LDA是生成式模型，而CorEX是判别式模型。

## 迭代
-	我们要找到的主题是
$$
\max _{p(y | x)} T C(X ; Y) \quad \text { s.t. } \quad|Y|=k
$$
-	我们可以找m个主题，并将$X$分为m个不相交的子集来建模
$$
\max _{G_{j}, p\left(y_{j} | x_{C_{j}}\right)} \sum_{j=1}^{m} T C\left(X_{G_{j}} ; Y_{j}\right) \quad \text { s.t. } \quad\left|Y_{j}\right|=k, G_{j} \cap G_{j^{\prime} \neq j}=\emptyset
$$
-	将上式用互信息改写为
$$
\max _{G, p\left(y_{j} | x\right)} \sum_{j=1}^{m} \sum_{i \in G_{j}} I\left(Y_{j} : X_{i}\right)-\sum_{j=1}^{m} I\left(Y_{j} : X_{G_{j}}\right)
$$
-	我们用指示函数进一步简化这个式子，去掉子集$G$的下标，统一用一个$\alpha$连通矩阵来代表子集的划分结果
$$
\alpha_{i, j}=\mathbb{I}\left[X_{i} \in G_{j}\right] \in\{0,1\}  \\
\max _{\alpha, p\left(y_{j} | x\right)} \sum_{j=1}^{m} \sum_{i=1}^{n} \alpha_{i, j} I\left(Y_{j} : X_{i}\right)-\sum_{j=1}^{m} I\left(Y_{j} : X\right) \\
$$
-	同时我们要加一个限制项保证子集不相交
$$
\sum_{\overline{j}} \alpha_{i, \overline{j}}=1
$$
-	这是一个带有限制项的最优化问题，通过拉格朗日乘子法可以解出
$$
\begin{aligned} p\left(y_{j} | x\right) &=\frac{1}{Z_{j}(x)} p\left(y_{j}\right) \prod_{i=1}^{n}\left(\frac{p\left(y_{j} | x_{i}\right)}{p\left(y_{j}\right)}\right)^{\alpha_{i, j}} \\ 
p\left(y_{j} | x_{i}\right) &=\sum_{\overline{x}} p\left(y_{j} | \overline{x}\right) p(\overline{x}) \delta_{\overline{x}_{i}, x_{i}} / p\left(x_{i}\right) \text { and } p\left(y_{j}\right)=\sum_{\overline{x}} p\left(y_{j} | \overline{x}\right) p(\overline{x}) \end{aligned} \\
$$

-	注意，这是在$\alpha$矩阵确认的情况下得到的主题最优解，通过放宽最优解条件，我们可以得到在主题更新之后$\alpha$的迭代公式
$$
\alpha_{i, j}^{t+1}=(1-\lambda) \alpha_{i, j}^{t}+\lambda \alpha_{i, j}^{* *} \\
\alpha_{i, j}^{* *}=\exp \left(\gamma\left(I\left(X_{i} : Y_{j}\right)-\max _{\overline{j}} I\left(X_{i} : Y_{\overline{j}}\right)\right)\right) \\
$$
## 伪算法
$$
\text { input : A matrix of size } n_{s} \times n \text { representing } n_{s} \text { samples of } n \text { discrete random variables } \\
\text { set } : \text { Set } m, \text { the number of latent variables, } Y_{j}, \text { and } k, \text { so that }\left|Y_{j}\right|=k \\
\text { output: Parameters } \alpha_{i, j}, p\left(y_{j} | x_{i}\right), p\left(y_{j}\right), p\left(y | x^{(l)}\right) \\
\text { for } i \in \mathbb{N}_{n}, j \in \mathbb{N}_{m}, l \in \mathbb{N}_{n_{s}}, y \in \mathbb{N}_{k}, x_{i} \in \mathcal{X}_{i} \\
\text { Randomly initialize } \alpha_{i, j}, p\left(y | x^{(l)}\right) \\
\text {repeat} \\
\text { Estimate marginals, } p\left(y_{j}\right), p\left(y_{j} | x_{i}\right) \text { using  } \\
p\left(y_{j} | x_{i}\right)=\sum_{\overline{x}} p\left(y_{j} | \overline{x}\right) p(\overline{x}) \delta_{\overline{x}_{i}, x_{i}} / p\left(x_{i}\right) \text { and } p\left(y_{j}\right)=\sum_{\overline{x}} p\left(y_{j} | \overline{x}\right) p(\overline{x}) \\
\text { Calculate } I\left(X_{i} : Y_{j}\right) \text { from marginals; } \\
\text { Update } \alpha \text { using  } \\
\alpha_{i, j}^{t+1}=(1-\lambda) \alpha_{i, j}^{t}+\lambda \alpha_{i, j}^{* *} \\
\text { Calculate } p\left(y | x^{(l)}\right), l=1, \ldots, n_{s} \text { using } \\
p\left(y_{j} | x\right)=\frac{1}{Z_{j}(x)} p\left(y_{j}\right) \prod_{i=1}^{n}\left(\frac{p\left(y_{j} | x_{i}\right)}{p\left(y_{j}\right)}\right)^{\alpha_{i, j}} \\
\text { until convergence; }
$$

# Maximally Informative Hierarchical Representations of High-Dimensional Data
-	本文分析了TC的上下界，有助于进一步理解TC的含义，并提出了一种最大化信息量的层次结构高维数据表示的优化方法，上文提到的CorEx可以看成这种优化方法的一种特例。
## 上界和下界
-	大部分定义与上文类似，更为一般性，我们将文档和主题扩展为数据$X$和表示$Y$，当联合概率可以分解时，我们称$Y$是$X$的一种表示
$$
p(x, y)=\prod_{j=1}^{m} p\left(y_{j} | x\right) p(x) \\
$$
-	这样，一种数据的表示完全由表示变量域和条件概率$p(y_j|x)$决定。
-	表示可以层次性堆叠，我们定义层次表示为：
$$
Y^{1 : r} \equiv Y^{1}, \ldots, Y^{r}
$$
-	![eYYvRK.png](https://s2.ax1x.com/2019/07/31/eYYvRK.png)
-	其中$Y^k$是$Y^{k-1}$的表示。我们主要关注量化层次表示对于数据的信息化程度的上下界。这种层次表示是一种一般性表示，包括了RBM和自编码器等等。
-	定义：
$$
T C_{L}(X ; Y) \equiv \sum_{i=1}^{n} I\left(Y : X_{i}\right)-\sum_{j=1}^{m} I\left(Y_{j} : X\right) \\
$$
-	则存在以下的边界和分解：
$$
T C(X) \geq T C(X ; Y)=T C(Y)+T C_{L}(X ; Y)
$$
-	同时得到$Y$关于$X$的TC值的一个下界：
$$
T C(X ; Y) \geq T C_{L}(X ; Y)
$$
-	当$TC(Y)$为0时取到下界，这时$Y$之间相互独立，不包含关于$X$的信息。将上面$TC(X)$的不等式扩展到层次表示，则可以得到
$$
T C(X) \geq \sum_{k=1}^{r} T C_{L}\left(Y^{k-1} ; Y^{k}\right)
$$
-	注意在这里我们定义第0层表示就是$X$，我们还能找到上界
$$
T C(X) \leq \sum_{k=1}^{r}\left(T C_{L}\left(Y^{k-1} ; Y^{k}\right)+\sum_{i=1}^{m_{k-1}} H\left(Y_{i}^{k-1} | Y^{k}\right)\right)
$$
-	可以看到上界与下界之间就差了一堆累加的条件熵。
-	TC的上下界可以帮助衡量表示对于数据的解释程度，
## 分析
-	先考虑最简单的情况，即第一层表示只有一个变量$Y^{1} \equiv Y_{1}^{1}$，这时
$$
TC(Y)+TC_L(X;Y)=TC(X;Y) \leq TC(X) \leq TC_L(X;Y)+\sum _{i=1}^{m_0} H(X_i|Y)
$$
-	待补充
## 优化
-	我们可以逐层优化，使得每一层最大化解释下一层的相关性（maximallly explain the correlations in the layer below)，这可以通过优化下界得到，以第一层为例
$$
\max _{\forall j, p\left(y_{j}^{1} | x\right)} T C_{L}\left(X ; Y^{1}\right)
$$
-	定义$\alpha$祖先信息为
$$
A I_{\alpha}(X ; Y) \equiv \sum_{i=1}^{n} \alpha_{i} I\left(Y : X_{i}\right)-I(Y : X) \\
\alpha_{i} \in[0,1] \\
$$
-	假如给定某个$\alpha$，其$AI_{\alpha}$为正，则 it implies the existence of common ancestors for some ($\alpha$-dependent) set of $X_i$ ’s in any DAG that describes $X$，这里不太懂，但可以看成是上文联通矩阵$\alpha$的泛化版本，从binarize泛化到01区间。最优化问题用$AI_{\alpha}$表示可以写成
$$
\max _{p(y | x)} \sum_{i=1}^{n} \alpha_{i} I\left(Y : X_{i}\right)-I(Y : X)
$$
-	化成了和上文一样的形式，之后的解法也一样
$$
p(y | x)=\frac{1}{Z(x)} p(y) \prod_{i=1}^{n}\left(\frac{p\left(y | x_{i}\right)}{p(y)}\right)^{\alpha_{i}}
$$
-	对归一化分母$Z(x)$取对数期望，可以得到自由能量，这正是我们的优化目标
$$
\begin{aligned} \mathbb{E}[\log Z(x)] &=\mathbb{E}\left[\log \frac{p(y)}{p(y | x)} \prod_{i=1}^{n}\left(\frac{p\left(y | x_{i}\right)}{p(y)}\right)^{\alpha_{i}}\right] \\ &=\sum_{i=1}^{n} \alpha_{i} I\left(Y : X_{i}\right)-I(Y : X) \end{aligned}
$$
-	对于多个隐变量，作者重构了下界，同样将$\alpha$扩展到01区间的连续值。具体过程比较复杂，最后的优化目标从最大化所有隐单元的$TC_L(X;Y)$变为优化$p(y_j|x)$和$\alpha$的下界：
$$
\max _{\alpha_{i, j}, p\left(y_{j} | x\right) \atop c_{i, j}\left(\alpha_{i, j}\right)=0}^{m} \sum_{j=1}^m \left(\sum_{i=1}^{n} \alpha_{i, j} I\left(Y_{j} : X_{i}\right)-I\left(Y_{j} : X\right)\right)
$$
-	$\alpha$定义了$X_i$和$Y_j$之间的关系，即结构。至于优化结构，理想的情况是
$$
\alpha _{i,j} = \mathbb{I} [j = argmax _{j} I(X_i : Y_j)]
$$
-	这样的结构是硬连接的，每一个节点只和下一层的某一个隐藏层节点相连接，基于$I(Y_j : X_i | Y_{1:j-1}) \geq \alpha _{i,j} I(Y_j : X_i)$,作者提出了一种启发式的算法来估计$\alpha$。我们检查$X_i$是否正确估计$Y_j$
$$
d_{i,j}^l \equiv \mathbb{I} [argmax_{y_j} \log p(Y_j = y_j|x^{(l)}) = argmax_{y_j} \log p(Y_j = y_j | x_i^{(l)}) / p(Y_j = y_j)]
$$
-	之后我们在所有样本上累加，统计正确估计数目，并根据比例设置$\alpha$值。

# Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge
## 概述
-	本文正式将CorEx应用于主题模型中，并强调了优于LDA的几点：
	-	不需要对数据做结构假设，相比LDA，CorEX具有更少的超参数
	-	不同于LDA，无需对模型做结构上的修改，即可泛化到层次模型和半监督没模型
-	模型的迭代依然是这几步：
$$
p_t(y_j) = \sum _{\overline{x}} p_t(y_j | \overline{x})p(\overline{x}) \\
p_t(x_i | y_j) = \sum _{\overline{x}} p_t(y_j|\overline{x})p(\overline{x}) \mathbb{I} [\overline{x}_i = x_i]/p_t(y_j) \\
\log p_{t+1} (y_j | x^l) = \log p_t(y_j) + \sum _{i=1}^n \alpha _{i,j}^t \log \frac{p_t(x_i^l | y_j)}{p(x_i^l)} - \log \mathbb{Z} _j (x^l) \\
$$
-	由于我们利用的是词袋信息，处理的是稀疏矩阵，因此边缘概率和条件概率计算都非常快，迭代中最慢的一步是第三个式子，即计算所有文档的主题分布。我们重写这个式子中累加的对数项：
$$
\log \frac{p_t(x_i^l | y_j)}{p(x_i^l)} = \log \frac{p_t(X_i=0|y_j)}{p(X_i=0)} + x_i^l \log (\frac{p_t(X_i^l=1|y_j)p(X_i=0)}{p_t(X_i=0|y_j)p(X_i^l=1)})
$$
-	这个累加是对每篇文档，在整个词典上计算似然，然而每篇文档只会出现一小部分词，当词典中的词没有出现在文档中的时候，上式中只有第一项不为0；当词出现时，上式中$\log P(X_i^l=1|y_j)/p(X_i^l=1)$为0，其余项保留，因此作者优先假设词不在文档中，之后再更新补充那些在文档中的词的概率项。经过这样的优化之后CorEx的计算速度和LDA差不多。
-	这个优化最大的好处是计算的复杂度只和文档数和主题数线性相关，因此计算大规模文档上的大规模主题成为可能。
## 半监督
-	半监督的思路非常简单，我们如何保证某一个主题一定出现某些词？只需要将连通矩阵的某些值固定即可。正常的$\alpha$在01区间之间，而将第i号词anchor在第j号主题可以将$\alpha_{i,j} = \beta _{i,j}$，其中$\beta$是anchor的强度。
-	这么做可以给每个主题anchor词，anchor一个或多个词，非常灵活。
-	在业务上来说，CorEx的优势在于：
	-	在训练超大规模主题数时非常快。
	-	可以方便的anchor词以适应领域。
	-	CorEx的主题之间词是不相交的，不会出现重复主题
-	层次主题就按照上一篇论文中的层次方法迭代，层次主题可以用于聚合概念，划分子话题。
