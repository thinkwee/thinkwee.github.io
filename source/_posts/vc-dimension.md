---
title: Note for VC Dimension
date: 2020-05-15 16:58:58
categories: ML
tags:
- vc dimension
- machine learning
- statistical learning
- math

mathjax: true
html: true 
---

简单的梳理VC维。所有讨论基于二分类这一简单情况出发。

<!--more-->

# Hoeffding不等式

- 机器学习的一个大假设就是，训练集上训练出的模型能够泛化到测试集，详细一点说即我们用算法A和训练集D，在假设空间H里中找到一个假设g，使得该假设g和需要学习的目标假设f近似。令g在训练集上的误差（经验误差）为$E_{in}(g)$，在除了训练集外所有可能样本上的误差（泛化误差，期望误差）为$E_{out}(g)$。
- 已知$E_{out}(f) = 0$，我们也希望得到的g满足$E_{out}(g) = 0$，这里包含了两点信息
  - 需要$E_{in}(g) \approx E_{out}(g)$
  - 需要$E_{in}(g) \approx 0$
- 第二点$E_{in}(g) \approx 0$就是我们训练模型所做的事，在训练集上减小误差，而第一点$E_{in}(g) \approx E_{out}(g)$，即如何保证模型的泛化能力，这是VC所衡量的事情。
- 训练集可以看成是样本空间的一个采样，从采样上计算一些量来估计整个样本空间的量，我们有Hoeffding不等式
  
  $$
  \mathbb{P}[|\nu-\mu|>\epsilon] \leq 2 \exp \left(-2 \epsilon^{2} N\right)
  $$
- 其中$\nu$是训练集上的计算量，$\mu$是整个样本空间上对应的量。我们自然可以把误差（损失）看成是一种计算量，得到：
  
  $$
  \mathbb{P}\left[\left|E_{i n}(h)-E_{o u t}(h)\right|>\epsilon\right] \leq 2 \exp \left(-2 \epsilon^{2} N\right)
  $$
- 其中h是某一假设（任意一个，不是我们通过训练挑出的最好假设g），左边即训练和真实损失差距大于某个相差阈值的概率，可以用来衡量泛化能力，而右侧发现只和相差阈值和训练集容量N相关。这符合我们的直觉，即训练集越大，泛化能力越强。
- 但问题是以上情况是针对固定的一个假设h，我们称经验误差和泛化误差相差很多的情况为bad case，那上述不等式说明了对于固定的一个h，bad case出现的概率很小。但是我们的算法A是在整个假设空间H里挑选假设，我们更关注对于所有h，其中任意一个h都不出现bad case出现的概率，即
  
  $$
  \begin{array}{c}
\mathbb{P}\left[\mathbf{E}\left(h_{1}\right)>\epsilon \cup \mathbf{E}\left(h_{2}\right)>\epsilon \ldots \cup \mathbf{E}\left(h_{M}\right)>\epsilon\right] \\
\leq \mathbb{P}\left[\mathbf{E}\left(h_{1}\right)>\epsilon\right]+\mathbb{P}\left[\mathbf{E}\left(h_{2}\right)>\epsilon\right] \ldots+\mathbb{P}\left[\mathbf{E}\left(h_{M}\right)>\epsilon\right] \\
\leq 2 M \exp \left(-2 \epsilon^{2} N\right)
\end{array}
  $$
- 其中$\mathbf{E}$即经验误差和泛化误差的差距，这里我们进一步放缩了这个上界，考虑最大的情况，也就是各个h经验误差和泛化误差的差距大于某个阈值的事件相互独立，事件的并的概率是各个事件概率之和，最终得到了针对所有h，bad case的上限是$2 M \exp (-2 \epsilon^{2} N)$
- 这下就出了问题，之前由于只有N存在于负指数项中，所以只要训练集容量足够大，这个上限是有限的，我们就对机器学习的泛化能力有信心；现在前面乘了一个M，即假设空间的容量，这下上限就不一定是有限（存在）的了。这也符合直觉，想象一下，假设空间越大，就需要越多的数据来训练，使得算法挑选出一个好的假设；假如数据量一定，假设空间越大，就越难挑出一个和真实假设f相近的g

# 有效的假设

- 接下来正式进入VC维的探讨。首先我们讨论有效假设数。考虑到上面的不等式，我们其实做了一个很大的放缩，即假定各个h经验误差和泛化误差的差距大于某个阈值的事件相互独立，事件的并的概率是各个事件概率之和。但实际上并不是这样的。例如对于二维线性可分的数据，存在分离间隔，那么分离间隔中的几个能正确分类训练集的几个相互平行且相差不远的分离面（直线），他们其实非常相近，对于绝大多数数据点而言在这两个分离面下的结果相同，同样泛化能力出现差距的概率分布也有很大重叠，将其视为相互独立是不合适的。
- 我们再回头看看不等式中的M，其字面意思就是假设空间的容量，实际上是假设空间表达能力的一种度量，在一定量的训练集下，假设空间表达能力越丰富，学习算法就越难找到一个好的假设。假设空间的容量，是所有假设的和，其中一个假设可以看成是确定模型下的一组参数。有没有其他的更为有效的衡量假设空间表达能力的量？
- 假如我们不看模型参数本身，而是看模型对数据的分类结果呢？例如，之前是相同的参数（相同的直线）视为相同的假设，现在用相同的分类结果（不同的直线，把点做了相同的分类）的参数视为相同的假设。这样看来似乎更加合理，因为表达能力最终落实在对数据的所有可能分类结果的处理，而无论经验误差还是泛化误差也都是通过误分类点来衡量。
- 这里引入了几个术语：
  - 对分(dichotomy)：记为$h\left(X_{1}, X_{2}, \ldots, X_{N}\right)$。N个点的对分即N个点的一种分类结果，显然对于二分类问题，N个点最多有$2^N$个dichotomy。
  - 假设空间H在训练集D里N个点上的对分，记为$\mathcal{H}\left(X_{1}, X_{2}, \ldots, X_{N}\right)$，这里对分的个数除了和数据集相关，还和假设空间H相关，因为假设空间并不一定能取到所有的dichotomy，例如四个点组成正方形，对角线的标签一样，那么一条直线的假设空间就取不到这种对分。
  - 增长函数(growth function)：假设空间在训练集上的对分个数不仅与训练集大小相关，还与具体的训练集相关，例如抽出来的训练集不包含（四个点组成正方形，对角线的标签一样）这种对分，那么可能直线在该训练集上取到的对分数量就和该训练集最多可能的对分数量一样。因此为了排除具体的训练集影响，我们设在所有大小为N的可能训练集上假设空间H能取到的最大对分数量为增长函数，定为：$m_{\mathcal{H}}(N)=\max _{X_{1}, X 2, \ldots, X_{N} \in \mathcal{X}}\left|\mathcal{H}\left(X_{1}, X_{2}, \ldots, X_{N}\right)\right|$
  - 打散(shatter)：即增长函数取到了理论上最大的对分数$2^N$，这里的含义就是这个假设空间对于大小为N的数据集（该任务设定下任意一个大小为N数据集）来说足够好了，能够考虑到所有情况，针对任何一种分类结果给出对应的假设。也即对于该任务，总容量为N的数据集，该模型有能力取到完美解。
  - break point：显然N越小，越容易打散；N越大，越不容易打散。那么从N=1开始逐渐增加，直到某个临界点N=k时，假设空间无法打散这k个点了，我们就称k为该假设空间的break point。可以证明N>k的情况都无法打散
- 可以看到其实到增长函数这里，这就是一个假设空间表达能力的一种衡量指标了。VC维考虑的就是用这种衡量指标替换Hoeffding不等式中的假设空间容量。当然不能直接替换，具体而言，是
  
  $$
  \forall g \in \mathcal{H}, \mathbb{P}\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 4 m_{\mathcal{H}}(2 N) \exp \left(-\frac{1}{8} \epsilon^{2} N\right)
  $$
- 上式右边即VC界，证明比较复杂略。那么这样我们再考虑上界是不是有限的。可以看到最重要的M被换成了$m_H$，即增长函数，假如增长函数有界或者说随N的增加速率低于指数部分的缩减速率，那么我们就可以说经验误差和泛化误差的差值上界是有限的。实际上，假如break point k存在，那么有
  
  $$
  m_{\mathcal{H}}(N) \leq B(N, k) \leq \sum_{i=0}^{k-1}\left(\begin{array}{c}
N \\
i
\end{array}\right) \leq N^{k-1}
  $$
- 其中B(N, k)就是break point为k时增长函数的上界，可以发现增长函数上界的上界是多项式级别的(k-1次方)，那么增长函数最多是多项式级别，而VC界中的$\exp \left(-2 \epsilon^{2} N\right)$是指数级别，显然VC界是存在的，有限的。
- 那么终于，对于break point存在的情况，我们可以说泛化能力是有保证的，机器学习是有保证的。

# VC维

- VC界保证了学习的可行性，而VC维考虑的是假设空间的表达能力。前面说到增长函数是对于假设空间表达能力的一种衡量，两者也存在密切关联。
- VC维定义为，给定假设空间H，其break point存在，那么其VC维是能够打散的最大数据集的大小，即
  
  $$
  V C(\mathcal{H})=\max \left\{N: m_{\mathcal{H}}(N)=2^{N}\right\}
  $$
- 回忆VC维定义中的两个概念：增长函数和打散。增长函数定义了假设空间解决对分的能力，打散代表着假设空间有能力解决一定量的数据集，对于每一种可能的对分都有对应假设，那么VC维就是从数据集的量来衡量假设空间的能力，假设空间越复杂，表达能力越强，就能打散更大的数据集，有足够多的假设解决更大数据集上的每一种上可能的对分——注意，不是每一种理论上可能的对分，因为VC维取的是能够达到的最大数据集容量，是存在而不是任意，即最大存在大小为VC维的数据集能够被该假设空间打散。
- 那么VC维具体是多少呢？其实之前我们定义过了，就是k-1，即容量小于break point的数据集都能被打散，那么能被打散的数据集容量最大就是break point - 1
- 注意到这个k-1不就是VC界中增长函数上界中的多项式次数吗，因此VC界又可以写成：
  
  $$
  \forall g \in \mathcal{H}, \mathbb{P}\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 4(2 N)^{V C(\mathcal{H})} \exp \left(-\frac{1}{8} \epsilon^{2} N\right)
  $$

# 衡量量化能力

- 由VC界的不等式可以看到，经验误差与泛化误差之间的差距（衡量泛化能力）和不等式右边即bad case发生的概率相关联，假如我们给定了bad case发生的概率，设为：
  
  $$
  4(2 N)^{V C(\mathcal{H})} \exp \left(-\frac{1}{8} \epsilon^{2} N\right)=\delta
  $$
  
  那么可以反过来算出衡量泛化能力的差距
  
  $$
  \epsilon=\sqrt{\frac{8}{N} \ln \left(\frac{4(2 N)^{V C(\mathcal{H})}}{\delta}\right)}
  $$
- 所以在机器学习基石和技法当中，老师经常写出一个VC bound用来估计泛化误差的公式：
  
  $$
  E_{\text {out }}(\mathbf{w}) \leq E_{\text {in }}(\mathbf{w})+\Omega(\mathcal{H})
  $$
- 其中的$\Omega(\mathcal{H})$即$\sqrt{\frac{8}{N} \ln \left(\frac{4(2 N)^{V C(\mathcal{H})}}{\delta}\right)}$

# 参考

- https://zhuanlan.zhihu.com/p/59113933
- https://www.coursera.org/learn/ntumlone-mathematicalfoundations