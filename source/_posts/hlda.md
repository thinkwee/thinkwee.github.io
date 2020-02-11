---
title: hLDA学习笔记
date: 2019-11-15 10:53:34
categories: 机器学习
tags:
  - lda
  - math
  - topic model
mathjax: true
html: true
---

记录 Hierarchical Latent Dirichlet Allocation，层次主题模型的学习笔记。
依然大量参考了徐亦达老师的教程。

<!--more-->

# hLDA改进了什么

-	改进了两点
	
	-	引入了Dirichlet Process
	
	-	引入了层次结构

# DP

-	Dirichlet Process将Dirichlet Distribution的概念扩展到随机过程，一般依概率采样会得到一个样本，一个值，而依据随机过程采样得到的是一个函数，是一个分布。给定DP的超参$\alpha$，给定度量空间$\theta$，以及该度量空间上的一个测度$H$（称为基分布, Base Distribution)，$DP(\alpha,H)$中采样得到的就是一个在$\theta$上的无限维离散分布$G$，假如对这个无限维（无限个离散点）做$\theta$上的任意一种划分$A_1,...,A_n$，那么划分之后的$G$分布依然满足对应Dirichlet Distribution在超参上的划分：
	$$
	(G(A_1,...,A_n))  \sim  Dir(\alpha H(A_1),...,\alpha H(A_n))
	$$
	$G$定义为Dirichlet Process的一个sample path/function/realization,即$G=DP(t,w_0) \sim \ DP(\alpha,H)$。Dirichelt Process的一个realization是一个概率测度，是一个函数，定义域在度量空间$\theta$上，函数输出即概率。注意因为是无限维，因此不能预先设置$\alpha$的维数，只能设置为一样的$\alpha$，对比LDA，可以看到DP的超参$\alpha$是一个concentration parameter，只能控制G分布趋于均匀分布的确定性，而不能控制G分布趋于怎样的分布，趋于怎样的分布由划分$A$决定。

-	这里可以看到和LDA使用Dir Distribution的区别：DP是直接采样生成了一个概率测度，可以进而生成离散的概率分布；而LDA中对Dir Distribution采样只能得到一个样本，但是这个样本作为了多项式分布的参数，确定了一个多项式分布（也是离散的）。

-	DP可以用于描述混合模型，在混合组件数量不确定的情况下，通过DP来构造一个组件分配。放在GMM的场景里，假如有n个样本，但我不知道有几个GM来生成这n个样本，那么对样本i，我将其分配给某一个GM，称这个样本i所在GM的参数为$\theta _i$，那么这个$\theta$服从一个基分布$H(\theta)$，假如$H$是连续分布，那么两个样本取到相同的$\theta$的概率趋于零，相当于n个样本对应n个GM，那么我们可以把这个$H$离散化为G，离散的方式为$G \sim DP(\alpha,H)$，$\alpha$越小越离散，越大则$G$越趋近于$H$。注意$H$也可以是离散的。

-	DP的两个参数，$H$和$\alpha$，前者决定了$G$的每一个离散点的位置，即$\theta _i$具体的值；后者决定了离散程度，或者理解为$\theta$有多分散，有多不重复，即概率分布是集中的还是分散的，这个Dirichlet Distribution里的$\alpha$是一致的。

-	由于G满足Dirichlet Distribution,因此有很多好的性质，包括对于多项式分布的conjugate，collapsing和splitting，以及renormalization。
	
	-	$E[G(A_i)]=H(A_i)$
	
	-	$Var[G(A_i)]=\frac {H(A_i)[1-H(A_i)]}{\alpha + 1}$
	
	-	可以看到$\alpha$取极端时，方差分别退化为0或者伯努利分布的方差，对应着之前我们说的G去离散化H的两种极端情况。


-	那么我们想用DP做什么，做一个生成式模型：我们想得到一个概率测度$G \sim \ DP(H,\alpha)$，根据$G$得到每一个样本点i所属的组对应的参数(Group Parameter)$x_i  \sim \  G$，之后根据这个参数和函数$F$生成样本点i：$p_i \sim \ F(x_i)$

-	接下来可以用中国餐馆过程(CRP)、折棒过程(Stick Breaking)和Polya Urm模型来细化这个$x_i$，即将和样本点i对应组的参数拆成样本点i对应的组和每组的参数，写成$x_i=\phi _{g_i}$，其中$g$是样本点的组分配，$\phi$是组参数。

-	接下来套用[echen大佬的描述](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process)来描述三个模型如何细化$x_i$的：

-	In the Chinese Restaurant Process:
	
	-	We generate table assignments $g_1, \ldots, g_n \sim CRP(\alpha)$ according to a Chinese Restaurant Process. ($g_i$ is the table assigned to datapoint $i$.)
	
	-	We generate table parameters $\phi_1, \ldots, \phi_m \sim G_0$ according to the base distribution $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
	
	-	Given table assignments and table parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$ from a distribution $F$ with the specified table parameters. (For example, $F$ could be a Gaussian, and $\phi_i$ could be a parameter vector specifying the mean and standard deviation).

-	In the Polya Urn Model:
	
	-	We generate colors $\phi_1, \ldots, \phi_n \sim Polya(G_0, \alpha)$ according to a Polya Urn Model. ($\phi_i$ is the color of the ith ball.)
	
	-	Given ball colors, we generate each datapoint $p_i \sim F(\phi_i)$.

-	In the Stick-Breaking Process:
	
	-	We generate group probabilities (stick lengths) $w_1, \ldots, w_{\infty} \sim Stick(\alpha)$ according to a Stick-Breaking process.
	
	-	We generate group parameters $\phi_1, \ldots, \phi_{\infty} \sim G_0$ from $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
	
	-	We generate group assignments $g_1, \ldots, g_n \sim Multinomial(w_1, \ldots, w_{\infty})$ for each datapoint.
	
	-	Given group assignments and group parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$.

-	In the Dirichlet Process:
	
	-	We generate a distribution $G \sim DP(G_0, \alpha)$ from a Dirichlet Process with base distribution $G_0$ and dispersion parameter $\alpha$.
	
	-	We generate group-level parameters $x_i \sim G$ from $G$, where $x_i$ is the group parameter for the ith datapoint. (Note: this is not the same as $\phi_i$. $x_i$ is the parameter associated to the group that the ith datapoint belongs to, whereas $\phi_k$ is the parameter of the kth distinct group.)
	
	-	Given group-level parameters $x_i$, we generate each datapoint $p_i \sim F(x_i)$.

# 折棒过程

-	折棒过程提供了一种在$\theta$上的无限划分，依然令DP的参数为$\alpha$，折棒过程如下：
	
	-	$\beta _1 \sim Beta(1,\alpha)$
	
	-	$A_1 = \beta _1$
	
	-	$\beta _2 \sim Beta(1,\alpha)$
	
	-	$A_2 = (1-\pi _1) * \beta _2$

-	这样每次从Beta分布中得到[0,1]上的一个划分，将整个$\theta$切成两部分，第一部分作为$\theta$上的第一个划分，剩下的部分看成下一次折棒的整体，接着从上面切两部分，第一部分作为$\theta$上的第二个划分，像一个棒不断被折断，每次从剩下的部分里折，最后折成的分段就是划分。

# DP2CRP

-	引入一个示性函数，假如两个样本点i,j他们被分配的组件相同，则他们的示性函数$z$相同，也就是表征每一个样本属于哪一个组件，$x_i \sim Component(\theta _{z_i})$

-	那么对于混合分布，比如GMM，我们希望得到的是predictive distribution，即已知数据的组件分配情况下，新来了一个未知数据，我想知道他属于哪个组件：
	$$
	p(z_i=m|z_{not \ i})
	$$

-	结合定义可以知道这个概率应该是和$H$无关的，因为我不在乎$\theta$具体的值，我只在乎是哪一个$\theta$，所以predictive distribution与$\alpha$密切相关。将其展开：
	$$
	p(z_i=m|z_{not \ i}) = \frac {p(z_i=m,z_{not \ i})}{p(z_{not \ i})} \\
	$$

-	由于在DP里是划分的类别数可以到无穷多个，因此这里采用了一个小技巧，我们先假设有k类，之后在把k趋于无穷
	$$
	= \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)p(p_1...p_k)}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)p(p_1...p_k)}
	$$

-	这里的k个类的概率是符合Dirichlet Distribution的，假设这里的Base Distribution是均匀分布，则
	$$
	= \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac {\alpha}{k})}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac{\alpha}{k})}
	$$

-	上面无论分子分母，积分内其实都是一个多项式分布乘以一个Dirichlet分布，根据共轭我们知道后验应该还是一个Dirichlet分布，我们推导一下多项式分布与Dirichlet分布相乘的积分：
	$$
	\int _{p_1...p_k} p(n_1...n_k|p_1...p_k) p(p_1...p_k|\alpha _1 ... \alpha _k) \\
	= \int _{p_1...p_k} Mul(n_1...n_k|p_1...p_k) Dir(p_1...p_k|\alpha _1 ... \alpha _k) \\
	= \int _{p_1...p_k} (\frac {n!}{n_1!...n_k!} \prod _{i=1}^k p_i ^{n_i}) \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \prod _{i=1}^k p_i^{\alpha _i -1} \\
	= \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \int _{p_1...p_k} \prod _{i=1}^k  p_i^{n_i+\alpha _i -1} \\
	$$

-	其中积分式内实际上是一个Dirichelt Distribution$Dir(\alpha _1 + n_1 ... \alpha _k + n_k)$排除了常数部分，因此积分的结果就是1/常数，即：
	$$
	= \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \frac { \prod \Gamma (\alpha _i + n_i)}{\Gamma (n + \sum \alpha _i)}
	$$

-	上式包括了三个部分，第一部分的一堆n，它是由多项式分布引入的，代表我们只看划分后每个集合的大小，而不看划分之后每个集合具体的内容，这和我们的需求是不一样的，因此不需要这个常数；第二个部分，是由Dir分布先验产生的，而在predictive distribution中，分布先验都相同，因此抵消了，所以我们主要关注第三部分，回代入predictive distribution那个分式当中。

-	首先定义一个辅助变量$n_{l , not \ i} = Count(z_{not \ i} == l)$，那么：
	$$
	n_1 = n_{1,not \ i} \\
	... \\
	n_k = n_{k,not \ i} \\
	$$

-	因为我们是是在求$p(z_i=m, z_{not \ i})$，那么肯定除了第m类，其余类的数量早已由除了第i个样本以外的样本确定，那么第m类呢？
	$$
	n_m = n_{m,not \ i} + 1
	$$

-	这样我们就完成了从指示函数表示的概率到多项式分布的转换，分子部分代入之前得到的第三部分有：
	$$
	\frac {\Gamma(n_{m,not \ i} + \frac {\alpha}{k} + 1) \prod _{l=1,l \neq m}^k Gamma(n_{l,not \ i})}{\Gamma (\alpha + n)}
	$$

-	同理计算分子，分子不用考虑第i个样本分给第m类，因此不用在累乘里单独拎出来第m项，形式要简单一些：
	$$
	\frac {\prod _{l=1}^k \Gamma(n_{l,not \ i})}{\Gamma(\alpha +n -1)}
	$$

-	将上面两式相除，再利用Gamma函数$\Gamma(x) = (x-1) \Gamma (x-1)$的性质简化，得到：
	$$
	= \frac {n_{m,not \ i} + \frac {\alpha}{k}}{n + \alpha - 1}
	$$

-	再令k趋于无穷，得到：
	$$
	= \frac {n_{m,not \ i}}{n + \alpha - 1}
	$$

-	但是上面这个式子对所有的类别从1到m求和并不为1，而是$\frac {n-1}{n + \alpha -1}$，剩下一部分概率就设为取一个新类别的概率，这样我们的predictive distribution就算完成了，而且可以发现，这个概率，实际上就对应着中国餐馆过程。

# CRP

-	中国餐馆过程的经典描述就是把n个人，一个一个人来，分到不确定张数目的桌子上，做一个整数集合上的划分。假设集合每个元素是一位顾客，第n位顾客走进了一家参观，则他按照以下概率去选择某一张已经有人的桌子坐下，或者找一张没人的新桌子坐下：
	$$
	\begin{aligned} p(\text { occupied table } i | \text { previous customers }) &=\frac{n_{i}}{\alpha +n-1} \\ p(\text { next unoccupied table } | \text { previous customers }) &=\frac{\alpha }{\alpha +n-1} \end{aligned}
	$$

-	其中$n_i$是第i张桌子上已经有的人数，$\alpha $是超参数。这样人到桌子的分配就对应了整数集合上的划分。

-	分析一下，若是选择已经有人的桌子，则顾客倾向于选择人多的桌子；若是在有人的桌子与新桌子之间纠结，则依赖于超参$\alpha $

-	那根据之前的推导，这个$\alpha$其实就是Dirichlet Distribution的超参数，且效果完全吻合。由于在CRP中我们base distribution选的是均匀分布，那对应的Dirichlet Distribution选择对称超参，各个$alpha _i$相同。那么$\alpha$越大，以Dirichlet Distritbuion为参数先验的多项式分布里，取得各个项等概率的可能就越大，在中国餐馆过程中对应着每个顾客进来都想选择一张新桌子，因此每个桌子都只有一个人，等量分配；反之$\alpha$越小则越不确定，在中国餐馆过程中桌子的分配也不确定

-	可以得到第m个人选择之后，桌子数量的期望是$E(K_m|\alpha ) = O(\alpha  \log m)$，具体而言是$E(K_m|\alpha ) = \alpha  (\Psi (\alpha  + n) - \Psi (\alpha )) \approx \alpha  \log (1 + \frac{n}{\alpha })$， 也就是聚类数的增加与样本数的对数成线性关系。我们可以根据数据量和想要聚类的数量来反估计超参$\alpha$的设置。

# nCRP

-	以上仅仅完成了一个利用了DP的不确定数目聚类，我们可以认为餐馆里每个桌子是一个主题，人就是单词，主题模型就是把词分配到主题，把人分配到桌子，但是这样的话和LDA一样，主题之间没有关联。为了建立主题之间的层次关系，Blei提出了嵌套餐馆过程。

-	在嵌套餐馆过程中，我们统一了餐馆和桌子的概念，餐馆就是桌子，桌子就是餐馆！为什么这么说？首先我们设置一个餐馆作为root餐馆（显然我们要建立一棵树了），然后根据中国餐馆过程选择root餐馆里的一个桌子，餐馆里的每个桌子上都有一张纸条指示顾客第二天去某一个餐馆，因此第二天顾客来到这个餐馆，接着根据CRP选个桌子，同时知晓了自己第三天该去哪个参观。因此桌子对应着餐馆，父节点餐馆的桌子对应着子节点餐馆，每一天就是树的每一层，这样就建立了一个层次结构的中国餐馆过程。

# hLDA

-	接下来我们可以在nCRP的框架上描述hLDA

-	定义符号
	
	-	$z$：主题，假设有$K$个
	
	-	$\beta$：主题到词分布的参数，Dir先验参数
	
	-	$w$：词
	
	-	$\theta$：文档到主题的分布
	
	-	$\alpha$：文档到主题分布的参数，Dir先验参数

-	那么可以简单定义LDA：
	$$
	p(w | \beta) \sim Dir(\beta) \\
	p(\theta | \alpha) \sim Dir(\alpha) \\
	\theta \sim p(\theta | \alpha) \\
	w \sim p(w | \theta , \beta) = \sum _{i=1}^K \theta _i p(w|z=i, \beta _i) \\
	$$

-	hLDA流程如下：
	
	-	根据nCRP获得一条从root到leaf的长为$L$的路径
	
	-	根据一个$L$维的Dirichlet采样一个在路径上的主题分布
	
	-	根据这L个主题混合生成一个词

-	详细描述如下：
	[![Mwfu34.md.jpg](https://s2.ax1x.com/2019/11/16/Mwfu34.md.jpg)](https://imgchr.com/i/Mwfu34)

-	概率图如下，其中$c$是餐馆，这里把nCRP单独拎出来了，实际上$c$决定了主题$z$，另外$\gamma$是nCRP中CRP对应DP的concentration paramter：
	[![Mwf3Hx.md.jpg](https://s2.ax1x.com/2019/11/16/Mwf3Hx.md.jpg)](https://imgchr.com/i/Mwf3Hx)

# Gibbs Sampling in hLDA

-	定义变量：
	
	-	$w_{m,n}$：第m篇文档里的第n个词
	
	-	$c_{m,l}$：第m篇文档里路径上第l层选择的主题对应的餐馆，需要采样计算
	
	-	$z_{m,n}$：第m篇文档里第n个词分配的主题，需要采样计算

-	从后验分布中采样的公式分为两部分，第一部分是得到路径，这一部分就会利用到之前的predictive distribution；第二部分是已知路径，剩下的部分就是普通的LDA，最终采样公式为：
	$$
	p\left(\mathbf{w}_{m} | \mathbf{c}, \mathbf{w}_{-m}, \mathbf{z}\right)=\prod_{\ell=1}^{L}\left(\frac{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+W \eta\right)}{\prod_{w} \Gamma\left(n_{c_{m, e},-m}^{(w)}+\eta\right)} \frac{\prod_{w} \Gamma\left(n_{c_{m, \ell},-m}^{(w)}+n_{c_{m, \ell}, m}^{(w)}+\eta\right)}{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+n_{c_{m, \ell}, m}^{(\cdot)}+W \eta\right)}\right)
	$$
