---
title: 推断算法笔记
date: 2018-08-28 09:55:10
categories: 机器学习
tags:
  - inference
  - math
  -	mcmc
  - variational inference
  - em
mathjax: true
photos: http://ojtdnrpmt.bkt.clouddn.com/blog/180828/BA94mibfCf.png?imageslim
html: true
---

记录Variational Inference、Expectation Maximization、Markov Chain Monte Carlo等用于概率机器学习中未知变量推断的算法的原理、推导。
很多内容和推导来自悉尼科技大学徐亦达教授的在线课程，徐老师讲非参贝叶斯的一系列视频非常好，可以直接在b站或者优酷搜索他的名字找到视频。
其他一些内容来自各种书或者tutorial，我会在文中说明。

<!--more-->

# Bayesian Inference
-	在贝叶斯推断中，需要区别可观察量（数据）和未知变量（可能是统计参数、缺失数据、隐变量）
-	统计参数在贝叶斯框架中被看成是随机变量，我们需要对模型的参数进行概率估计，而在频率学派的框架下，参数是确定的非随机的量，主要针对数据做概率估计
-	在频率学派框架中只关注似然$p(x|\theta)$，而贝叶斯学派认为应将参数$\theta$作为变量，在观察到数据之前，对参数做出先验假设$p(\theta)$
-	后验正比与似然乘以先验，代表观察到数据后我们对参数先验调整，得到的参数概率分布
-	在贝叶斯框架中我们更关注精确度，它是方差的倒数，例如在正态分布的后验中，精确度是先验和数据的精确度之和
-	后验实际上是在最大似然估计和先验之间权衡
-	当数据非常多时，后验渐渐不再依赖于先验
-	很多时候我们并没有先验知识，这时一般采用平坦的、分散的分布作为先验分布，例如范围很大的均匀分布，或者方差很大的正态分布
-	有时我们并不需要知道整个后验分布，而仅仅做点估计或者区间估计

# Markov Chain Monte Carlo
-	MCMC，前一个MC代表如何采样，使得采样点满足分布，后一个MC代表用随机采样来估计分布的参数
-	吉布斯采样的一个动机：对于多个参数的联合分布，很难直接采样，但是如果固定其他参数作为条件，仅仅对一个参数的条件分布做采样，这时采样会简单许多，且可以证明收敛之后这样采样出来的样本满足联合分布
-	明天再写

# Expectation Maximization
## 公式
-	对于简单的分布，我们想要做参数推断，只需要做最大似然估计，先求对数似然：

$$
\theta=\mathop{argmax}_{\theta} L(\theta | X) \\
=\mathop{argmax}_{\theta} \log \prod p(x_i | \theta) \\
=\mathop{argmax}_{\theta} \sum \log p(x_i | \theta) \\
$$

-	之后对这个对数似然求导计算极值即可，但是对于复杂的分布，可能并不方便求导
-	这时我们可以用EM算法迭代求解。EM算法为分布引入一个隐变量Z，之后迭代求解出一系列的$\theta$，可以证明，每一次迭代之后得到的$\theta$都会使对数似然增加。
-	每一次迭代分为两个部分，E和M，也就求期望和最大化
	-	求期望，是求$\log p(x,z|\theta)$在分布$p(z|x,\theta ^{(t)})$上的期望，其中$\theta ^{(t)}$是第t次迭代时计算出的参数
	-	最大化，也就是求使这个期望最大的$\theta$，作为本次参数迭代更新的结果
-	合起来就得到EM算法的公式：
$$
\theta ^{(t+1)} = \mathop{argmax} _{\theta} \int \log p(x,z|\theta)p(z|x,\theta ^{(t)}) dz
$$
## 为何有效
-	也就是证明，每次迭代后最大似然会增加
-	要证明：
$$
\log p(x|\theta ^{(t+1)}) \geq \log p(x|\theta ^{(t)})
$$
-	先改写对数似然
$$
\log p(x|\theta) = \log p(x,z|\theta) - \log p(z|x,\theta) \\
$$
-	两边对分布$p(z|x,\theta ^{(t)})$求期望，注意到等式左边与z无关，因此求期望之后不变：
$$
\log p(x|\theta) = \int _z \log p(x,z|\theta) p(z|x,\theta ^{(t)}) dz - \int _z \log p(z|x,\theta) p(z|x,\theta ^{(t)}) dz \\
=Q(\theta,\theta ^{(t)})-H(\theta,\theta ^{(t)}) \\
$$
-	其中Q部分就是EM算法中的E部分，注意在这里$\theta$是变量，$\theta ^{(t)}$是常量
-	迭代之后，由于EM算法中M部分作用，Q部分肯定变大了（大于等于），那么使Q部分变大的这个迭代之后新的$\theta$，代入H部分，H部分会怎么变化呢？
-	我们先计算，假如H部分的$\theta$不变，直接用上一次的$\theta ^{(t)}$带入，即$H(\theta ^{(t)},\theta ^{(t)})$
$$
H(\theta ^{(t)},\theta ^{(t)})-H(\theta,\theta ^{(t)})= \\
\int _z \log p(z|x,\theta ^{(t)}) p(z|x,\theta ^{(t)}) dz - \int _z \log p(z|x,\theta) p(z|x,\theta ^{(t)}) dz \\
= \int _z \log (\frac {p(z|x,\theta ^{(t)})} {p(z|x,\theta)} ) p(z|x,\theta ^{(t)}) dz \\
= - \int _z \log (\frac {p(z|x,\theta)} {p(z|x,\theta ^{(t)})} ) p(z|x,\theta ^{(t)}) dz \\
\geq - \log \int _z  (\frac {p(z|x,\theta)} {p(z|x,\theta ^{(t)})} ) p(z|x,\theta ^{(t)}) dz \\
= - \log 1 \\
= 0 \\
$$
-	其中那个不等式是利用了Jensen不等式。也就是说，直接用上一次的$\theta ^{(t)}$作为$\theta$代入H，就是H的最大值!那么无论新的由argmax Q部分得到的$\theta ^{(t+1)}$是多少，带入	H,H部分都会减小（小于等于）！被减数变大，减数变小，那么得到的结果就是对数似然肯定变大，也就证明了EM算法的有效性

## 从ELBO的角度理解
-	我们还可以从ELBO（Evidence Lower Bound）的角度推出EM算法的公式
-	在之前改写对数似然时我们得到了两个式子$p(x,z|\theta)$和$p(z|x,\theta)$，我们引入隐变量的一个分布$q(z)$，对这个两个式子做其与$q(z)$之间的KL散度，可以证明对数似然是这两个KL散度之差：
$$
KL(q(z)||p(z|x,\theta)) = \int q(z) [\log q(z) - \log p(z|x,\theta)] dz \\
= \int q(z) [\log q(z) - \log p(x|z,\theta) - \log (z|\theta) + \log p(x|\theta)] dz \\
= \int q(z) [\log q(z) - \log p(x|z,\theta) - \log (z|\theta)] dz + \log p(x|\theta) \\
= \int q(z) [\log q(z) - \log p(x,z|\theta)] dz + \log p(x|\theta) \\
= KL(q(z)||p(x,z|\theta)) + \log p(x|\theta) \\
$$
-	也就是
$$
\log p(x|\theta) = - KL(q(z)||p(x,z|\theta)) + KL(q(z)||p(z|x,\theta))
$$
-	其中$- KL(q(z)||p(x,z|\theta))$就是ELBO，因为$ KL(q(z)||p(z|x,\theta)) \geq 0 $，因此ELBO是对数似然的下界。我们可以通过最大化这个下界来最大化对数似然
-	可以看到，ELBO有两个参数，$q$和$\theta$，首先我们固定$\theta ^{(t-1)}$，找到使ELBO最大化的$q^{(t)}$，这一步实际上是EM算法的E步骤，接下来固定$q^{(t)}$，找到使ELBO最大化的$\theta ^{(t)}$，这一步对应的就是EM算法的M步骤
-	我们把$\theta = \theta ^{(t-1)}$带入ELBO的表达式：
$$
ELBO=\log p(x|\theta ^{(t-1)}) - KL(q(z)||p(z|x,\theta ^{(t-1)}))
$$
-	q取什么值时ELBO最大？显然当KL散度为0时，ELBO取到最大值，也就是下界达到对数似然本身，这时$q(z)=p(z|x,\theta ^{(t-1)})$，接下来我们固定$q$，求使ELBO最大的$\theta$，先把ELBO的定义式改写：
$$
ELBO = - KL(q(z)||p(x,z|\theta)) \\
= \int q^{(t)}(z) [ \log p(x,z|\theta) - \log q^{(t)}(z)] dz \\
= - \int q^{(t)}(z) \log p(x,z|\theta) - q^{(t)}(z) \log q^{(t)}(z) dz \\
$$
-	其中第二项与$\theta$无关，因此：
$$
\theta ^{(t)} = \mathop{argmax} _{\theta} \int q^{(t)}(z) \log p(x,z|\theta) dz \\
$$
-	代入上一步得到的$q(z)=p(z|x,\theta ^{(t-1)})$，得到
$$
\theta ^{(t)} = \mathop{argmax} _{\theta} \int \log p(x,z|\theta)p(z|x,\theta ^{(t-1)}) dz
$$
-	同样得到了EM算法的迭代公式
-	下面两张图截取自Christopher M. Bishop的Pattern Recognition and Machine Learning，说明了E步骤和M步骤实际在做什么：E步骤将下界ELBO提高到对数似然，但是这时只更新了隐变量，因此对数似然没有变化，而当利用更新的隐变量更新参数$\theta$，也就是M步骤执行后，我们继续获得了更高的ELBO，以及其对应的对数似然，此时q没有变化，但p发生改变，因此KL不为0，对数似然一定大于ELBO，也就是会提升。直观的来说，我们在E和M步骤都提高了ELBO，E步骤先一口气将ELBO提满到对数似然，之后M步骤依然可以提高ELBO，但对数似然肯定会大于等于（在M步骤时实际上是大于）ELBO，因此对数似然就被M步骤提升的ELBO给“顶上去了”。
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180830/dCj203m7jB.PNG)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180830/6k4kbJH3bc.PNG)
-	剩下的问题就是，如何选择z以及q，在混合模型中，可以将z作为示性函数引入，其他在设计时包含隐变量的概率模型里，可以直接将隐变量引入

## 从假设隐变量为可观察的角度理解
-	这种理解来自Chuong B Do & Serafim Batzoglou的tutorial:What is the expectation maximization algorithm?
-	EM用于包含不可观察隐变量的概率模型推断，事实上，如果我们将隐变量从不可观察变为可观察，针对隐变量每一种可能的取值做最大似然估计，一样可以得到结果，但其时间代价是相当高的。
-	EM则改进了这种朴素的算法。一种对EM算法的理解是：EM算法在每次迭代中先猜想一种隐变量的取值概率分布，创造一个考虑了所有隐变量取值可能的加权的训练集，然后在这上面做一个魔改版本的最大似然估计。
-	猜想一种隐变量的取值概率分布就是E步骤，但是我们不需要知道具体的概率分布，我们只需要求充分统计量在这个分布上的期望（Expectation）。
-	所以说EM算法是最大似然估计在包含隐变量的数据（或者说包含部分不可观察样本的数据）上的自然泛化。

## EM算法与K-means
-	K-means是一种Hard-EM算法，它一样对隐变量的各种可能做出假设（样本属于的类），但是他并不是在类上计算概率和期望，而是比较Hard，只指定一个类作为样本的类，只有这个类概率为1，其余均为0。

## 广义EM算法

## Wake-Sleep算法

## 广义EM算法与吉布斯采样

# Variational Inference
## ELBO
-	接下来介绍变分推断，可以看到，EM算法可以推广到变分推断
-	重新推出ELBO与对数似然的关系：
$$
\log p(x) = \log p(x,z) - \log p(z|x) \\
= \log \frac{p(x,z)}{q(z)} - \log \frac{p(z|x)}{q(z)} \\
= \log p(x,z) - \log q(z) - \log \frac{p(z|x)}{q(z)} \\
$$
-	两边对隐分布$q(z)$求期望
$$
\log p(x) = [ \int _z q(z) \log p(x,z)dz - \int _z q(z) \log q(z)dz ] + [- \int _z \log \frac{p(z|x)}{q(z)} q(z) dz ]\\
= ELBO+KL(q||p(z|x)) \\
$$
-	我们希望推断隐变量$z$的后验分布$p(z|x)$，为此我们引入一个分布$q(z)$来近似这个后验。当目前观测量也就是对数似然确定的前提下，近似后验等价于使得$q(z)$和$p(z|x)$的KL散度最小，由上式可以看出，当ELBO最大时，KL散度最小。
-	接下来就是讨论如何使得ELBO最大化

## 第一种方法
-	对任意分布使用，一次选取隐变量一个分量更新，比如第j个分量
-	我们自己选取的$q(z)$当然要比近似的分布简单，这里假设分布是独立的，隐变量是$M$维的：
$$
q(z)=\prod _{i=1}^M q_i(z_i)
$$
-	因此ELBO可以写成两部分
$$
ELBO=\int \prod q_i(z_i) \log p(x,z) dz - \int \prod q_j(z_j) \sum \log q_j(z_j) dz \\
=part1-part2 \\
$$
-	其中part1可以写成对隐变量各个维度求多重积分的形式，我们挑出第j个维度将其改写成
$$
part1=\int \prod q_i(z_i) \log p(x,z) dz \\
= \int _{z_1} \int _{z_2} ... \int _{z_M} \prod _{i=1}^M q_i(z_i) \log p(x,z) d z_1 , d z_2 , ... ,d z_M \\
= \int _{z_j} q_j(z_j) ( \int _{z_{i \neq j}} \log (p(x,z)) \prod _{z_{i \neq j}} q_i(z_i) d z_i) d z_j \\
= \int _{z_j}  q_j(z_j) [E_{i \neq j} [\log (p(x,z))]] d z_j \\
$$
-	在此我们定义一种伪分布的形式，一种分布的伪分布就是对其对数求积分再求指数：
$$
p_j(z_j) = \int _{i \neq j} p(z_1,...,z_i) d z_1 , d z_2 ,..., d z_i \\
p_j^{'}(z_j) = exp \int _{i \neq j} \log p(z_1,...,z_i) d z_1 , d z_2 ,..., d z_i \\
\log p_j^{'}(z_j)  = \int _{i \neq j} \log p(z_1,...,z_i) d z_1 , d z_2 ,..., d z_i \\
$$
-	这样part1用伪分布的形式可以改写成
$$
part1= \int _{z_j} \log \log p_j^{'}(x,z_j) \\
$$
-	part2中因为隐变量各个分量独立，可以把函数的和在联合分布上的期望改写成各个函数在边缘分布上的期望的和，在这些和中我们关注第j个变量，其余看成常量：
$$
part2=\int \prod q_j(z_j) \sum \log q_j(z_j) dz \\
= \sum ( \int q_i(z_i) \log (q_i(z_i)) d z_i ) \\
= \int q_j(z_j) \log (q_j(z_j)) d z_j + const \\
$$
-	再把part1和part2合起来，得到ELBO关于分量j的形式：
$$
ELBO = \int _{z_j} \log \log p_j^{'}(x,z_j) -  \int q_j(z_j) \log (q_j(z_j)) d z_j + const \\
= \int _{z_j} q_j(z_j) \log \frac{p_j^{'}(x,z_j)}{q_j(z_j)} + const \\
= - KL(p_j^{'}(x,z_j) || q_j(z_j)) \\
$$
-	也就是将ELBO写成了伪分布和近似分布之间的负KL散度，最大化ELBO就是最小化这个KL散度
-	何时这个KL散度最小？也就是：
$$
q_j(z_j) = p_j^{'}(x,z_j) \\
\log q_j(z_j) = E_{i \neq j} [\log (p(x,z))] \\
$$
-	到此我们就得到了变分推断下对于隐变量单一分量的近似分布迭代公式，在计算第j个分量的概率时，用到了$\log (p(x,z))$在其他所有分量$q_i(z_i)$上的期望，之后这个新的第j个分量的概率就参与下一次迭代，计算出其他分量的概率。

## 第二种方法
-	针对指数家族分布的变分推断
-	定义指数家族分布：
$$
p(x | \theta)=h(x) exp(\eta (\theta) \cdot T(x)-A(\theta)) \\
$$
-	其中
	-	$T(x)$:sufficient statistics
	-	$\theta$:parameter of the family
	-	$\eta$:natural parameter
	-	$h(x)$:underlying measure
	-	$A(\theta)$:log normalizer / partition function
-	注意parameter of the family和natural parameter都是向量，当指数家族分布处于标量化参数形式，即$\eta _i (\theta) = \theta _i$的时候，指数家族分布可以写成：
$$
p(x | \eta)=h(x) exp(\eta (T(x) ^T \eta - A(\eta))
$$
-	当我们把概率密度函数写成指数家族形式，求最大对数似然时，有：
$$
\eta = \mathop{argmax} _ {\eta} [\log p(X | \eta)] \\
= \mathop{argmax} _ {\eta} [\log \prod p(x_i | \eta)] \\
= \mathop{argmax} _ {\eta} [\log [\prod h(x_i) exp [(\sum T(x_i))^T \eta - n A(\eta)]]] \\
= \mathop{argmax} _ {\eta} (\sum T(x_i))^T \eta - n A(\eta)] \\
= \mathop{argmax} _ {\eta} L(\eta) \\
$$
-	继续求极值，我们就可以得到指数家族分布关于log normalizer和sufficient statistics的很重要的一个性质：
$$
\frac{\partial L (\eta)}{\partial \eta} = \sum T(x_i) - n A^{'}(\eta) =0 \\
A^{'}(\eta) = \sum \frac{T(x_i)}{n} \\
$$
-	举个例子，高斯分布写成指数家族分布形式：
$$
p(x) = exp[- \frac{1}{2 \sigma ^2}x^2 + \frac{\mu}{\sigma ^2}x - \frac{\mu ^2}{2 \sigma ^2} - \frac 12 \log(2 \pi \sigma ^2)] \\
=exp ( [x \ x^2] [\frac{\mu}{\sigma ^2} \ \frac{-1}{2 \sigma ^2}] ^T - \frac{\mu ^2}{2 \sigma ^2} - \frac 12 \log(2 \pi \sigma ^2) )
$$
-	用自然参数去替代方差和均值，写成指数家族分布形式：
$$
p(x) = exp( [x \ x^2] [ \eta _1 \ \eta _2] ^T + \frac{\eta _1 ^2}{4 \eta _2} + \frac 12 \log (-2 \eta _2 ) - \frac 12 \log (2 \pi))
$$
-	其中：
	-	$T(x)$:$[x \ x^2]$
	-	$\eta$:$[ \eta _1 \ \eta _2] ^T$
	-	$-A(\eta)$:$\frac{\eta _1 ^2}{4 \eta _2} + \frac 12 \log (-2 \eta _2 )$
-	接下来我们利用指数家族的性质来快速计算均值和方差
$$
A^{'}(\eta) = \sum \frac{T(x_i)}{n} \\
[\frac{\partial A}{\eta _1} \ \frac{\partial A}{\eta _2}] = [\frac{- \eta _1}{2 \eta _2} \ \frac{\eta _1 ^2 }{2 \eta _2}-\frac{1}{2 \eta _2}] \\
= [\frac{\sum x_i}{n} \ \frac{\sum x_i^2}{n}] \\
= [\mu \ \mu ^2 + \sigma ^2] \\
$$
