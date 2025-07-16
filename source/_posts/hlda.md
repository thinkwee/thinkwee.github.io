---
title: Note for Hierarchical Latent Dirichlet Allocation
date: 2019-11-15 10:53:34
categories: ML
tags:
  - lda
  - math
  - topic model
mathjax: true
html: true
refplus: true
---

<img src="https://i.mji.rip/2025/07/16/cbd846ecd88abb611db2c204930d896d.png" width="500"/>

Note for Hierarchical Latent Dirichlet Allocation


<!--more-->

{% language_switch %}

{% lang_content en %}
- Still mainly referred to Prof. Yida Xu’s tutorials{% ref yidaxu %}.

# Improvements of hLDA

- Improved two points
  
  - Introduced Dirichlet Process
  
  - Introduced Hierarchical Structure

# DP

- The Dirichlet Process extends the concept of Dirichlet Distribution to a random process. Typically, sampling by probability yields a sample, a value, while sampling by random process yields a function, a distribution. Given the DP's hyperparameter $\alpha$, a metric space $\theta$, and a measure $H$ on this metric space (called base distribution), sampling from $DP(\alpha,H)$ generates an infinite-dimensional discrete distribution $G$ on $\theta$. For any partition $A_1,...,A_n$ of $\theta$, the partitioned $G$ still follows the Dirichlet Distribution corresponding to the hyperparameters:
  
  $$
  (G(A_1,...,A_n))  \sim  Dir(\alpha H(A_1),...,\alpha H(A_n))
  $$
  
  $G$ is defined as a sample path/function/realization of the Dirichlet Process, i.e., $G=DP(t,w_0) \sim \ DP(\alpha,H)$. A realization of the Dirichlet Process is a probability measure, a function defined on the metric space $\theta$, with its output being a probability. Note that due to its infinite-dimensionality, $\alpha$ cannot be preset to a specific dimension, but only set to be the same $\alpha$. Compared to LDA, we can see that DP's hyperparameter $\alpha$ is a concentration parameter that can only control the certainty of G distribution trending towards uniformity, while the specific distribution trend is determined by the partition $A$.

- Here we can see the difference from LDA's use of Dir Distribution: DP directly samples to generate a probability measure, which can further generate a discrete probability distribution; while in LDA, sampling from Dir Distribution only yields a sample, which serves as a parameter for the multinomial distribution, determining a discrete distribution.

- DP can be used to describe mixture models in scenarios with an uncertain number of components. In a GMM scenario, if there are n samples but we don't know how many GMs generated these n samples, for sample i, we assign it to a certain GM, and let the parameters of this GM for sample i be $\theta _i$. This $\theta$ follows a base distribution $H(\theta)$. If $H$ is a continuous distribution, the probability of two samples taking the same $\theta$ approaches zero, equivalent to n samples corresponding to n GMs. We can discretize this $H$ into G, with the discretization method being $G \sim DP(\alpha,H)$. The smaller $\alpha$ is, the more discrete it becomes; the larger $\alpha$ is, the closer G is to H. Note that $H$ can also be discrete.

- The two parameters of DP, $H$ and $\alpha$, where the former determines the location of each discrete point of $G$, i.e., the specific value of $\theta _i$; the latter determines the degree of discreteness, or how dispersed $\theta$ is, whether the probability distribution is concentrated or dispersed, consistent with the $\alpha$ in Dirichlet Distribution.

- Since G satisfies Dirichlet Distribution, it has many good properties, including conjugacy to the multinomial distribution, collapsing and splitting, and renormalization.
  
  - $E[G(A_i)]=H(A_i)$
  
  - $Var[G(A_i)]=\frac {H(A_i)[1-H(A_i)]}{\alpha + 1}$
  
  - We can see that when $\alpha$ takes extreme values, the variance degenerates to 0 or the variance of a Bernoulli distribution, corresponding to the two extreme cases of G discretizing H that we mentioned earlier.

- So what do we want to do with DP? Create a generative model: we want to obtain a probability measure $G \sim \ DP(H,\alpha)$, obtain the group parameter for each sample point i based on $G$: $x_i  \sim \  G$, and then generate the sample point i based on this parameter and function $F$: $p_i \sim \ F(x_i)$

- Next, we can use the Chinese Restaurant Process (CRP), Stick Breaking Process, and Polya Urn Model to refine this $x_i$, that is, split the group parameter corresponding to the sample point i into the group assignment and group parameters, written as $x_i=\phi _{g_i}$, where $g$ is the group assignment of the sample point, and $\phi$ is the group parameter.

- Next, using [echen's description](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process) to describe how three models refine $x_i$:

- In the Chinese Restaurant Process:
  
  - We generate table assignments $g_1, \ldots, g_n \sim CRP(\alpha)$ according to a Chinese Restaurant Process. ($g_i$ is the table assigned to datapoint $i$.)
  
  - We generate table parameters $\phi_1, \ldots, \phi_m \sim G_0$ according to the base distribution $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
  
  - Given table assignments and table parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$ from a distribution $F$ with the specified table parameters. (For example, $F$ could be a Gaussian, and $\phi_i$ could be a parameter vector specifying the mean and standard deviation).

- In the Polya Urn Model:
  
  - We generate colors $\phi_1, \ldots, \phi_n \sim Polya(G_0, \alpha)$ according to a Polya Urn Model. ($\phi_i$ is the color of the ith ball.)
  
  - Given ball colors, we generate each datapoint $p_i \sim F(\phi_i)$.

- In the Stick-Breaking Process:
  
  - We generate group probabilities (stick lengths) $w_1, \ldots, w_{\infty} \sim Stick(\alpha)$ according to a Stick-Breaking process.
  
  - We generate group parameters $\phi_1, \ldots, \phi_{\infty} \sim G_0$ from $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
  
  - We generate group assignments $g_1, \ldots, g_n \sim Multinomial(w_1, \ldots, w_{\infty})$ for each datapoint.
  
  - Given group assignments and group parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$.

- In the Dirichlet Process:
  
  - We generate a distribution $G \sim DP(G_0, \alpha)$ from a Dirichlet Process with base distribution $G_0$ and dispersion parameter $\alpha$.
  
  - We generate group-level parameters $x_i \sim G$ from $G$, where $x_i$ is the group parameter for the ith datapoint. (Note: this is not the same as $\phi_i$. $x_i$ is the parameter associated to the group that the ith datapoint belongs to, whereas $\phi_k$ is the parameter of the kth distinct group.)
  
  - Given group-level parameters $x_i$, we generate each datapoint $p_i \sim F(x_i)$.

# Stick-Breaking Process

- The Stick-Breaking Process provides an infinite division on $\theta$. Let the DP parameters be $\alpha$, and the process is as follows:
  
  - $\beta _1 \sim Beta(1,\alpha)$
  
  - $A_1 = \beta _1$
  
  - $\beta _2 \sim Beta(1,\alpha)$
  
  - $A_2 = (1-\pi _1) * \beta _2$

- This way, each time a division on [0,1] is obtained from the Beta distribution, cutting the entire $\theta$ into two parts. The first part is taken as the first division on $\theta$, and the remaining part is seen as the whole for the next stick-breaking. Then, cut it into two parts again, with the first part taken as the second division on $\theta$. It's like a stick being continuously broken, each time breaking from the remaining part, and the final segments are the divisions.

# DP2CRP

- Introduce an indicator function. If two sample points i and j are assigned to the same component, their indicator function $z$ is the same, which represents which component each sample belongs to, $x_i \sim Component(\theta _{z_i})$

- For a mixture distribution, such as GMM, we want to obtain the predictive distribution, that is, given the component assignment of known data, for a new unknown data point, we want to know which component it belongs to:
  
  $$
  p(z_i=m|z_{not \ i})
  $$

- From the definition, we know this probability should be independent of $H$, because we don't care about the specific value of $\theta$, we only care which $\theta$ it is, so the predictive distribution is closely related to $\alpha$. Expanding it:
  
  $$
  p(z_i=m|z_{not \ i}) = \frac {p(z_i=m,z_{not \ i})}{p(z_{not \ i})} \\
  $$

- Since in DP, the number of categories can be infinite, we first assume k categories, and then let k approach infinity
  
  $$
  = \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)p(p_1...p_k)}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)p(p_1...p_k)}
  $$

- The probabilities of these k categories follow a Dirichlet Distribution. Assuming the Base Distribution is uniform, then
  
  $$
  = \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac {\alpha}{k})}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac{\alpha}{k})}
  $$

- In both numerator and denominator, the integral is essentially a multinomial distribution multiplied by a Dirichlet distribution. Due to conjugacy, the posterior should still be a Dirichlet distribution. We derive the integral of the multinomial distribution multiplied by the Dirichlet distribution:
  
  $$
  \int _{p_1...p_k} p(n_1...n_k|p_1...p_k) p(p_1...p_k|\alpha _1 ... \alpha _k) \\
  $$
  
  $$
  = \int _{p_1...p_k} Mul(n_1...n_k|p_1...p_k) Dir(p_1...p_k|\alpha _1 ... \alpha _k) \\
  $$
  
  $$
  = \int _{p_1...p_k} (\frac {n!}{n_1!...n_k!} \prod _{i=1}^k p_i ^{n_i}) \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \prod _{i=1}^k p_i^{\alpha _i -1} \\
  $$
  
  $$
  = \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \int _{p_1...p_k} \prod _{i=1}^k  p_i^{n_i+\alpha _i -1} \\
  $$

- The integral term is actually a Dirichlet Distribution $Dir(\alpha _1 + n_1 ... \alpha _k + n_k)$ excluding the constant part, so the integral result is 1/constant, i.e.:
  
  $$
  = \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \frac { \prod \Gamma (\alpha _i + n_i)}{\Gamma (n + \sum \alpha _i)}
  $$

- This expression includes three parts. The first part with n's is introduced by the multinomial distribution, representing that we only look at the size of each set after division, not the specific content of each set, which is different from our requirements, so we don't need this constant. The second part is generated by the Dir distribution prior, and in the predictive distribution, the distribution priors are all the same, so they cancel out. We mainly focus on the third part, substituting it back into the predictive distribution fraction.

- First, define an auxiliary variable $n_{l , not \ i} = Count(z_{not \ i} == l)$, then:
  
  $$
  n_1 = n_{1,not \ i} \\
  $$
  
  $$
  ... \\
  $$
  
  $$
  n_k = n_{k,not \ i} \\
  $$

- Because we are seeking $p(z_i=m, z_{not \ i})$, the number of other categories is already determined by samples other than the ith sample. What about the mth category?
  
  $$
  n_m = n_{m,not \ i} + 1
  $$

- This completes the transformation from indicator function representation to multinomial distribution. Substituting the third part of the previous derivation into the numerator gives:
  
  $$
  \frac {\Gamma(n_{m,not \ i} + \frac {\alpha}{k} + 1) \prod _{l=1,l \neq m}^k Gamma(n_{l,not \ i})}{\Gamma (\alpha + n)}
  $$

- Similarly calculating the numerator, the numerator doesn't need to consider the ith sample assigned to the mth category, so the form is simpler:
  
  $$
  \frac {\prod _{l=1}^k \Gamma(n_{l,not \ i})}{\Gamma(\alpha +n -1)}
  $$

- Dividing the two expressions and using the property of the Gamma function $\Gamma(x) = (x-1) \Gamma (x-1)$ to simplify, we get:
  
  $$
  = \frac {n_{m,not \ i} + \frac {\alpha}{k}}{n + \alpha - 1}
  $$

- Letting k approach infinity, we get:
  
  $$
  = \frac {n_{m,not \ i}}{n + \alpha - 1}
  $$

- However, the sum of this expression for all categories from 1 to m is not 1, but $\frac {n-1}{n + \alpha -1}$. The remaining probability is set as the probability of taking a new category, thus completing the predictive distribution. Interestingly, this probability corresponds exactly to the Chinese Restaurant Process.

# CRP

- The classic description of the Chinese Restaurant Process is to distribute n people to an uncertain number of tables, creating a partition on an integer set. Assuming each element in the set is a customer, when the nth customer enters a restaurant, they choose a table according to the following probabilities:
  
  $$
  \begin{aligned} p(\text { occupied table } i | \text { previous customers }) &=\frac{n_{i}}{\alpha +n-1} \\ p(\text { next unoccupied table } | \text { previous customers }) &=\frac{\alpha }{\alpha +n-1} \end{aligned}
  $$

- Where $n_i$ is the number of people already at table i, and $\alpha$ is the hyperparameter. This way, the assignment of people to tables corresponds to a partition on the integer set.

- Analyzing this, if choosing an occupied table, customers tend to choose tables with more people; if torn between occupied tables and a new table, it depends on the hyperparameter $\alpha$

- According to the previous derivation, this $\alpha$ is actually the hyperparameter of the Dirichlet Distribution, and the effect completely matches. Since we choose a uniform base distribution in CRP, the corresponding Dirichlet Distribution chooses symmetric hyperparameters with the same $alpha _i$. The larger $\alpha$ is, the more likely it is to obtain an equal probability for each item in the Dirichlet Distribution as a prior for the multinomial distribution. In the Chinese Restaurant Process, this corresponds to each customer wanting to choose a new table, so each table has only one person and is equally distributed. Conversely, the smaller $\alpha$ is, the less certain, and in the Chinese Restaurant Process, the table assignments are also less certain.

- We can obtain that the expected number of tables after the mth person chooses is $E(K_m|\alpha ) = O(\alpha  \log m)$, specifically $E(K_m|\alpha ) = \alpha  (\Psi (\alpha  + n) - \Psi (\alpha )) \approx \alpha  \log (1 + \frac{n}{\alpha })$, which means the increase in the number of clusters is linearly related to the logarithm of the sample size. We can estimate the hyperparameter $\alpha$ based on the amount of data and the desired number of clusters.

# nCRP

- The above only completes an uncertain number of clustering using DP. We can consider each table in the restaurant as a topic, people as words, and the topic model as assigning words to topics, or people to tables, but this is the same as LDA, with no correlation between topics. To establish a hierarchical relationship between topics, Blei proposed the Nested Chinese Restaurant Process.

- In the Nested Chinese Restaurant Process, we unify the concepts of restaurants and tables: restaurants are tables, tables are restaurants! Why do we say this? First, we set a root restaurant (obviously, we're building a tree), then choose a table in the root restaurant according to the Chinese Restaurant Process. Each table in the restaurant has a note indicating which restaurant the customer should go to the next day. So the next day, the customer arrives at this restaurant and chooses a table according to CRP, while also knowing which restaurant to go to on the third day. Thus, tables correspond to restaurants, and the tables of the parent restaurant correspond to child restaurants. Each day is a layer of the tree, establishing a hierarchical structure of the Chinese Restaurant Process.

# hLDA

- Now we can describe hLDA in the framework of nCRP

- Define symbols
  
  - $z$: topics, assuming $K$ topics
  
  - $\beta$: parameters from topics to word distribution, Dir prior parameters
  
  - $w$: words
  
  - $\theta$: document-to-topic distribution
  
  - $\alpha$: parameters of document-to-topic distribution, Dir prior parameters

- We can simply define LDA:
  
  $$
  p(w | \beta) \sim Dir(\beta) \\
p(\theta | \alpha) \sim Dir(\alpha) \\
\theta \sim p(\theta | \alpha) \\
w \sim p(w | \theta , \beta) = \sum _{i=1}^K \theta _i p(w|z=i, \beta _i) \\
  $$

- hLDA process:
  
  - Obtain a path from root to leaf of length $L$ according to nCRP
  
  - Sample a topic distribution on the path from a $L$-dimensional Dirichlet
  
  - Generate a word by mixing these L topics

- Detailed description:
  [![Mwfu34.md.jpg](https://s2.ax1x.com/2019/11/16/Mwfu34.md.jpg)](https://imgchr.com/i/Mwfu34)

- Probability graph, where $c$ is the restaurant, nCRP is separately drawn out here. Actually, $c$ determines the topic $z$, and $\gamma$ is the concentration parameter of CRP corresponding to DP:
  [![Mwf3Hx.md.jpg](https://s2.ax1x.com/2019/11/16/Mwf3Hx.md.jpg)](https://imgchr.com/i/Mwf3Hx)

# Gibbs Sampling in hLDA

- Define variables:
  
  - $w_{m,n}$: the nth word in the mth document
  
  - $c_{m,l}$: the restaurant corresponding to the topic at the lth layer in the path of the mth document, needs to be sampled and calculated
  
  - $z_{m,n}$: the topic assigned to the nth word in the mth document, needs to be sampled and calculated

- The sampling formula from the posterior distribution is divided into two parts. The first part is obtaining the path, which will use the previous predictive distribution; the second part is known the path, which is similar to ordinary LDA. The final sampling formula is:
  
  $$
  p\left(\mathbf{w}_{m} | \mathbf{c}, \mathbf{w}_{-m}, \mathbf{z}\right)=\prod_{\ell=1}^{L}\left(\frac{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+W \eta\right)}{\prod_{w} \Gamma\left(n_{c_{m, e},-m}^{(w)}+\eta\right)} \frac{\prod_{w} \Gamma\left(n_{c_{m, \ell},-m}^{(w)}+n_{c_{m, \ell}, m}^{(w)}+\eta\right)}{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+n_{c_{m, \ell}, m}^{(\cdot)}+W \eta\right)}\right)
  $$
{% endlang_content %}

{% lang_content zh %}

- 依然主要参考了徐亦达老师的教程{% ref yidaxu %}。

# hLDA改进了什么

- 改进了两点
  
  - 引入了Dirichlet Process
  
  - 引入了层次结构

# DP

- Dirichlet Process将Dirichlet Distribution的概念扩展到随机过程，一般依概率采样会得到一个样本，一个值，而依据随机过程采样得到的是一个函数，是一个分布。给定DP的超参$\alpha$，给定度量空间$\theta$，以及该度量空间上的一个测度$H$（称为基分布, Base Distribution)，$DP(\alpha,H)$中采样得到的就是一个在$\theta$上的无限维离散分布$G$，假如对这个无限维（无限个离散点）做$\theta$上的任意一种划分$A_1,...,A_n$，那么划分之后的$G$分布依然满足对应Dirichlet Distribution在超参上的划分：
  
  $$
  (G(A_1,...,A_n))  \sim  Dir(\alpha H(A_1),...,\alpha H(A_n))
  $$
  
  $G$定义为Dirichlet Process的一个sample path/function/realization,即$G=DP(t,w_0) \sim \ DP(\alpha,H)$。Dirichelt Process的一个realization是一个概率测度，是一个函数，定义域在度量空间$\theta$上，函数输出即概率。注意因为是无限维，因此不能预先设置$\alpha$的维数，只能设置为一样的$\alpha$，对比LDA，可以看到DP的超参$\alpha$是一个concentration parameter，只能控制G分布趋于均匀分布的确定性，而不能控制G分布趋于怎样的分布，趋于怎样的分布由划分$A$决定。

- 这里可以看到和LDA使用Dir Distribution的区别：DP是直接采样生成了一个概率测度，可以进而生成离散的概率分布；而LDA中对Dir Distribution采样只能得到一个样本，但是这个样本作为了多项式分布的参数，确定了一个多项式分布（也是离散的）。

- DP可以用于描述混合模型，在混合组件数量不确定的情况下，通过DP来构造一个组件分配。放在GMM的场景里，假如有n个样本，但我不知道有几个GM来生成这n个样本，那么对样本i，我将其分配给某一个GM，称这个样本i所在GM的参数为$\theta _i$，那么这个$\theta$服从一个基分布$H(\theta)$，假如$H$是连续分布，那么两个样本取到相同的$\theta$的概率趋于零，相当于n个样本对应n个GM，那么我们可以把这个$H$离散化为G，离散的方式为$G \sim DP(\alpha,H)$，$\alpha$越小越离散，越大则$G$越趋近于$H$。注意$H$也可以是离散的。

- DP的两个参数，$H$和$\alpha$，前者决定了$G$的每一个离散点的位置，即$\theta _i$具体的值；后者决定了离散程度，或者理解为$\theta$有多分散，有多不重复，即概率分布是集中的还是分散的，这个Dirichlet Distribution里的$\alpha$是一致的。

- 由于G满足Dirichlet Distribution,因此有很多好的性质，包括对于多项式分布的conjugate，collapsing和splitting，以及renormalization。
  
  - $E[G(A_i)]=H(A_i)$
  
  - $Var[G(A_i)]=\frac {H(A_i)[1-H(A_i)]}{\alpha + 1}$
  
  - 可以看到$\alpha$取极端时，方差分别退化为0或者伯努利分布的方差，对应着之前我们说的G去离散化H的两种极端情况。

- 那么我们想用DP做什么，做一个生成式模型：我们想得到一个概率测度$G \sim \ DP(H,\alpha)$，根据$G$得到每一个样本点i所属的组对应的参数(Group Parameter)$x_i  \sim \  G$，之后根据这个参数和函数$F$生成样本点i：$p_i \sim \ F(x_i)$

- 接下来可以用中国餐馆过程(CRP)、折棒过程(Stick Breaking)和Polya Urm模型来细化这个$x_i$，即将和样本点i对应组的参数拆成样本点i对应的组和每组的参数，写成$x_i=\phi _{g_i}$，其中$g$是样本点的组分配，$\phi$是组参数。

- 接下来套用[echen大佬的描述](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process)来描述三个模型如何细化$x_i$的：

- In the Chinese Restaurant Process:
  
  - We generate table assignments $g_1, \ldots, g_n \sim CRP(\alpha)$ according to a Chinese Restaurant Process. ($g_i$ is the table assigned to datapoint $i$.)
  
  - We generate table parameters $\phi_1, \ldots, \phi_m \sim G_0$ according to the base distribution $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
  
  - Given table assignments and table parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$ from a distribution $F$ with the specified table parameters. (For example, $F$ could be a Gaussian, and $\phi_i$ could be a parameter vector specifying the mean and standard deviation).

- In the Polya Urn Model:
  
  - We generate colors $\phi_1, \ldots, \phi_n \sim Polya(G_0, \alpha)$ according to a Polya Urn Model. ($\phi_i$ is the color of the ith ball.)
  
  - Given ball colors, we generate each datapoint $p_i \sim F(\phi_i)$.

- In the Stick-Breaking Process:
  
  - We generate group probabilities (stick lengths) $w_1, \ldots, w_{\infty} \sim Stick(\alpha)$ according to a Stick-Breaking process.
  
  - We generate group parameters $\phi_1, \ldots, \phi_{\infty} \sim G_0$ from $G_0$, where $\phi_k$ is the parameter for the kth distinct group.
  
  - We generate group assignments $g_1, \ldots, g_n \sim Multinomial(w_1, \ldots, w_{\infty})$ for each datapoint.
  
  - Given group assignments and group parameters, we generate each datapoint $p_i \sim F(\phi_{g_i})$.

- In the Dirichlet Process:
  
  - We generate a distribution $G \sim DP(G_0, \alpha)$ from a Dirichlet Process with base distribution $G_0$ and dispersion parameter $\alpha$.
  
  - We generate group-level parameters $x_i \sim G$ from $G$, where $x_i$ is the group parameter for the ith datapoint. (Note: this is not the same as $\phi_i$. $x_i$ is the parameter associated to the group that the ith datapoint belongs to, whereas $\phi_k$ is the parameter of the kth distinct group.)
  
  - Given group-level parameters $x_i$, we generate each datapoint $p_i \sim F(x_i)$.

# 折棒过程

- 折棒过程提供了一种在$\theta$上的无限划分，依然令DP的参数为$\alpha$，折棒过程如下：
  
  - $\beta _1 \sim Beta(1,\alpha)$
  
  - $A_1 = \beta _1$
  
  - $\beta _2 \sim Beta(1,\alpha)$
  
  - $A_2 = (1-\pi _1) * \beta _2$

- 这样每次从Beta分布中得到[0,1]上的一个划分，将整个$\theta$切成两部分，第一部分作为$\theta$上的第一个划分，剩下的部分看成下一次折棒的整体，接着从上面切两部分，第一部分作为$\theta$上的第二个划分，像一个棒不断被折断，每次从剩下的部分里折，最后折成的分段就是划分。

# DP2CRP

- 引入一个示性函数，假如两个样本点i,j他们被分配的组件相同，则他们的示性函数$z$相同，也就是表征每一个样本属于哪一个组件，$x_i \sim Component(\theta _{z_i})$

- 那么对于混合分布，比如GMM，我们希望得到的是predictive distribution，即已知数据的组件分配情况下，新来了一个未知数据，我想知道他属于哪个组件：
  
  $$
  p(z_i=m|z_{not \ i})
  $$

- 结合定义可以知道这个概率应该是和$H$无关的，因为我不在乎$\theta$具体的值，我只在乎是哪一个$\theta$，所以predictive distribution与$\alpha$密切相关。将其展开：
  
  $$
  p(z_i=m|z_{not \ i}) = \frac {p(z_i=m,z_{not \ i})}{p(z_{not \ i})} \\
  $$

- 由于在DP里是划分的类别数可以到无穷多个，因此这里采用了一个小技巧，我们先假设有k类，之后在把k趋于无穷
  
  $$
  = \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)p(p_1...p_k)}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)p(p_1...p_k)}
  $$

- 这里的k个类的概率是符合Dirichlet Distribution的，假设这里的Base Distribution是均匀分布，则
  
  $$
  = \frac {\int _{p_1...p_k} p(z_i=m, z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac {\alpha}{k})}{\int _{p_1...p_k} p(z_{not \ i}|p_1...p_k)Dir(\frac {\alpha}{k} ... \frac{\alpha}{k})}
  $$

- 上面无论分子分母，积分内其实都是一个多项式分布乘以一个Dirichlet分布，根据共轭我们知道后验应该还是一个Dirichlet分布，我们推导一下多项式分布与Dirichlet分布相乘的积分：
  
  $$
  \int _{p_1...p_k} p(n_1...n_k|p_1...p_k) p(p_1...p_k|\alpha _1 ... \alpha _k) \\
  $$
  
  $$
  = \int _{p_1...p_k} Mul(n_1...n_k|p_1...p_k) Dir(p_1...p_k|\alpha _1 ... \alpha _k) \\
  $$
  
  $$
  = \int _{p_1...p_k} (\frac {n!}{n_1!...n_k!} \prod _{i=1}^k p_i ^{n_i}) \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \prod _{i=1}^k p_i^{\alpha _i -1} \\
  $$
  
  $$
  = \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \int _{p_1...p_k} \prod _{i=1}^k  p_i^{n_i+\alpha _i -1} \\
  $$

- 其中积分式内实际上是一个Dirichelt Distribution$Dir(\alpha _1 + n_1 ... \alpha _k + n_k)$排除了常数部分，因此积分的结果就是1/常数，即：
  
  $$
  = \frac {n!}{n_1!...n_k!} \frac {\Gamma(\sum \alpha _i)}{\prod \Gamma (\alpha _i)} \frac { \prod \Gamma (\alpha _i + n_i)}{\Gamma (n + \sum \alpha _i)}
  $$

- 上式包括了三个部分，第一部分的一堆n，它是由多项式分布引入的，代表我们只看划分后每个集合的大小，而不看划分之后每个集合具体的内容，这和我们的需求是不一样的，因此不需要这个常数；第二个部分，是由Dir分布先验产生的，而在predictive distribution中，分布先验都相同，因此抵消了，所以我们主要关注第三部分，回代入predictive distribution那个分式当中。

- 首先定义一个辅助变量$n_{l , not \ i} = Count(z_{not \ i} == l)$，那么：
  
  $$
  n_1 = n_{1,not \ i} \\
  $$
  
  $$
  ... \\
  $$
  
  $$
  n_k = n_{k,not \ i} \\
  $$

- 因为我们是是在求$p(z_i=m, z_{not \ i})$，那么肯定除了第m类，其余类的数量早已由除了第i个样本以外的样本确定，那么第m类呢？
  
  $$
  n_m = n_{m,not \ i} + 1
  $$

- 这样我们就完成了从指示函数表示的概率到多项式分布的转换，分子部分代入之前得到的第三部分有：
  
  $$
  \frac {\Gamma(n_{m,not \ i} + \frac {\alpha}{k} + 1) \prod _{l=1,l \neq m}^k Gamma(n_{l,not \ i})}{\Gamma (\alpha + n)}
  $$

- 同理计算分子，分子不用考虑第i个样本分给第m类，因此不用在累乘里单独拎出来第m项，形式要简单一些：
  
  $$
  \frac {\prod _{l=1}^k \Gamma(n_{l,not \ i})}{\Gamma(\alpha +n -1)}
  $$

- 将上面两式相除，再利用Gamma函数$\Gamma(x) = (x-1) \Gamma (x-1)$的性质简化，得到：
  
  $$
  = \frac {n_{m,not \ i} + \frac {\alpha}{k}}{n + \alpha - 1}
  $$

- 再令k趋于无穷，得到：
  
  $$
  = \frac {n_{m,not \ i}}{n + \alpha - 1}
  $$

- 但是上面这个式子对所有的类别从1到m求和并不为1，而是$\frac {n-1}{n + \alpha -1}$，剩下一部分概率就设为取一个新类别的概率，这样我们的predictive distribution就算完成了，而且可以发现，这个概率，实际上就对应着中国餐馆过程。

# CRP

- 中国餐馆过程的经典描述就是把n个人，一个一个人来，分到不确定张数目的桌子上，做一个整数集合上的划分。假设集合每个元素是一位顾客，第n位顾客走进了一家参观，则他按照以下概率去选择某一张已经有人的桌子坐下，或者找一张没人的新桌子坐下：
  
  $$
  \begin{aligned} p(\text { occupied table } i | \text { previous customers }) &=\frac{n_{i}}{\alpha +n-1} \\ p(\text { next unoccupied table } | \text { previous customers }) &=\frac{\alpha }{\alpha +n-1} \end{aligned}
  $$

- 其中$n_i$是第i张桌子上已经有的人数，$\alpha $是超参数。这样人到桌子的分配就对应了整数集合上的划分。

- 分析一下，若是选择已经有人的桌子，则顾客倾向于选择人多的桌子；若是在有人的桌子与新桌子之间纠结，则依赖于超参$\alpha $

- 那根据之前的推导，这个$\alpha$其实就是Dirichlet Distribution的超参数，且效果完全吻合。由于在CRP中我们base distribution选的是均匀分布，那对应的Dirichlet Distribution选择对称超参，各个$alpha _i$相同。那么$\alpha$越大，以Dirichlet Distritbuion为参数先验的多项式分布里，取得各个项等概率的可能就越大，在中国餐馆过程中对应着每个顾客进来都想选择一张新桌子，因此每个桌子都只有一个人，等量分配；反之$\alpha$越小则越不确定，在中国餐馆过程中桌子的分配也不确定

- 可以得到第m个人选择之后，桌子数量的期望是$E(K_m|\alpha ) = O(\alpha  \log m)$，具体而言是$E(K_m|\alpha ) = \alpha  (\Psi (\alpha  + n) - \Psi (\alpha )) \approx \alpha  \log (1 + \frac{n}{\alpha })$， 也就是聚类数的增加与样本数的对数成线性关系。我们可以根据数据量和想要聚类的数量来反估计超参$\alpha$的设置。

# nCRP

- 以上仅仅完成了一个利用了DP的不确定数目聚类，我们可以认为餐馆里每个桌子是一个主题，人就是单词，主题模型就是把词分配到主题，把人分配到桌子，但是这样的话和LDA一样，主题之间没有关联。为了建立主题之间的层次关系，Blei提出了嵌套餐馆过程。

- 在嵌套餐馆过程中，我们统一了餐馆和桌子的概念，餐馆就是桌子，桌子就是餐馆！为什么这么说？首先我们设置一个餐馆作为root餐馆（显然我们要建立一棵树了），然后根据中国餐馆过程选择root餐馆里的一个桌子，餐馆里的每个桌子上都有一张纸条指示顾客第二天去某一个餐馆，因此第二天顾客来到这个餐馆，接着根据CRP选个桌子，同时知晓了自己第三天该去哪个参观。因此桌子对应着餐馆，父节点餐馆的桌子对应着子节点餐馆，每一天就是树的每一层，这样就建立了一个层次结构的中国餐馆过程。

# hLDA

- 接下来我们可以在nCRP的框架上描述hLDA

- 定义符号
  
  - $z$：主题，假设有$K$个
  
  - $\beta$：主题到词分布的参数，Dir先验参数
  
  - $w$：词
  
  - $\theta$：文档到主题的分布
  
  - $\alpha$：文档到主题分布的参数，Dir先验参数

- 那么可以简单定义LDA：
  
  $$
  p(w | \beta) \sim Dir(\beta) \\
p(\theta | \alpha) \sim Dir(\alpha) \\
\theta \sim p(\theta | \alpha) \\
w \sim p(w | \theta , \beta) = \sum _{i=1}^K \theta _i p(w|z=i, \beta _i) \\
  $$

- hLDA流程如下：
  
  - 根据nCRP获得一条从root到leaf的长为$L$的路径
  
  - 根据一个$L$维的Dirichlet采样一个在路径上的主题分布
  
  - 根据这L个主题混合生成一个词

- 详细描述如下：
  [![Mwfu34.md.jpg](https://s2.ax1x.com/2019/11/16/Mwfu34.md.jpg)](https://imgchr.com/i/Mwfu34)

- 概率图如下，其中$c$是餐馆，这里把nCRP单独拎出来了，实际上$c$决定了主题$z$，另外$\gamma$是nCRP中CRP对应DP的concentration paramter：
  [![Mwf3Hx.md.jpg](https://s2.ax1x.com/2019/11/16/Mwf3Hx.md.jpg)](https://imgchr.com/i/Mwf3Hx)

# Gibbs Sampling in hLDA

- 定义变量：
  
  - $w_{m,n}$：第m篇文档里的第n个词
  
  - $c_{m,l}$：第m篇文档里路径上第l层选择的主题对应的餐馆，需要采样计算
  
  - $z_{m,n}$：第m篇文档里第n个词分配的主题，需要采样计算

- 从后验分布中采样的公式分为两部分，第一部分是得到路径，这一部分就会利用到之前的predictive distribution；第二部分是已知路径，剩下的部分就是普通的LDA，最终采样公式为：
  
  $$
  p\left(\mathbf{w}_{m} | \mathbf{c}, \mathbf{w}_{-m}, \mathbf{z}\right)=\prod_{\ell=1}^{L}\left(\frac{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+W \eta\right)}{\prod_{w} \Gamma\left(n_{c_{m, e},-m}^{(w)}+\eta\right)} \frac{\prod_{w} \Gamma\left(n_{c_{m, \ell},-m}^{(w)}+n_{c_{m, \ell}, m}^{(w)}+\eta\right)}{\Gamma\left(n_{c_{m, \ell},-m}^{(\cdot)}+n_{c_{m, \ell}, m}^{(\cdot)}+W \eta\right)}\right)
  $$

{% endlang_content %}


{% references %}

[yidaxu] 徐亦达机器学习课程 Variational Inference for LDA https://www.youtube.com/watch?v=e1wr0xHbfYk

{% endreferences %}
<script src="https://giscus.app/client.js"
        data-repo="thinkwee/thinkwee.github.io"
        data-repo-id="MDEwOlJlcG9zaXRvcnk3OTYxNjMwOA=="
        data-category="Announcements"
        data-category-id="DIC_kwDOBL7ZNM4CkozI"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="light"
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>
