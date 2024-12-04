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

A brief review of the VC dimension. All discussions are based on the simple case of binary classification.

<!--more-->

{% language_switch %}

{% lang_content en %}
# Hoeffding's Inequality

- A major assumption in machine learning is that the model trained on the training set can generalize to the test set. More specifically, using algorithm A and training set D, we find a hypothesis g in the hypothesis space H that approximates the target hypothesis f. Let $E_{in}(g)$ represent the error (empirical error) on the training set, and $E_{out}(g)$ represent the error (generalization error, expected error) on all possible samples outside the training set.
- Given $E_{out}(f) = 0$, we also hope that the obtained g satisfies $E_{out}(g) = 0$, which contains two pieces of information:
  - We need $E_{in}(g) \approx E_{out}(g)$
  - We need $E_{in}(g) \approx 0$
- The second point $E_{in}(g) \approx 0$ is what we do when training the model, reducing the error on the training set, while the first point $E_{in}(g) \approx E_{out}(g)$, i.e., how to ensure the model's generalization ability, is what VC dimension measures.
- The training set can be seen as a sampling of the sample space, and we use Hoeffding's inequality to estimate quantities of the entire sample space from the sampling:
  
  $$
  \mathbb{P}[|\nu-\mu|>\epsilon] \leq 2 \exp \left(-2 \epsilon^{2} N\right)
  $$
- Where $\nu$ is the calculated quantity on the training set, $\mu$ is the corresponding quantity on the entire sample space. Naturally, we can view error (loss) as a kind of calculation, giving:
  
  $$
  \mathbb{P}\left[\left|E_{i n}(h)-E_{o u t}(h)\right|>\epsilon\right] \leq 2 \exp \left(-2 \epsilon^{2} N\right)
  $$
- Here h is a certain hypothesis (any one, not the best hypothesis g we selected through training). The left side is the probability of the difference between training and true loss exceeding a certain threshold, which can be used to measure generalization ability. The right side depends only on the difference threshold and training set size N. This aligns with our intuition that the larger the training set, the stronger the generalization ability.
- However, the above situation is for a fixed hypothesis h. We call the case where the empirical error and generalization error differ greatly a "bad case". The above inequality shows that for a fixed h, the probability of a bad case occurring is very small. But our algorithm A selects hypotheses from the entire hypothesis space H. We are more concerned with the probability that no h among all h experiences a bad case, i.e.:
  
  $$
  \begin{array}{c}
\mathbb{P}\left[\mathbf{E}\left(h_{1}\right)>\epsilon \cup \mathbf{E}\left(h_{2}\right)>\epsilon \ldots \cup \mathbf{E}\left(h_{M}\right)>\epsilon\right] \\
\leq \mathbb{P}\left[\mathbf{E}\left(h_{1}\right)>\epsilon\right]+\mathbb{P}\left[\mathbf{E}\left(h_{2}\right)>\epsilon\right] \ldots+\mathbb{P}\left[\mathbf{E}\left(h_{M}\right)>\epsilon\right] \\
\leq 2 M \exp \left(-2 \epsilon^{2} N\right)
\end{array}
  $$
- Where $\mathbf{E}$ represents the difference between empirical error and generalization error. Here we further bound this upper limit, considering the maximum case, assuming the events of each h's empirical and generalization errors differing beyond a certain threshold are independent. The probability of the union of events is the sum of individual event probabilities. We ultimately obtain an upper limit of bad cases for all h as $2 M \exp (-2 \epsilon^{2} N)$
- Now we've encountered a problem. Previously, since only N existed in the negative exponential term, as long as the training set size was large enough, this upper limit was finite, giving us confidence in machine learning's generalization ability. Now, with an M (the capacity of the hypothesis space) multiplied in front, the upper limit may no longer be finite. This also aligns with intuition: imagine that the larger the hypothesis space, the more data is needed to train and select a good hypothesis. If the data volume is fixed, the larger the hypothesis space, the harder it is to select a g close to the true hypothesis f.

# Effective Hypotheses

- Next, we formally enter the discussion of VC dimension. First, we discuss the number of effective hypotheses. Considering the above inequality, we made a large approximation by assuming that the events of each h's empirical and generalization errors differing beyond a threshold are independent, and the probability of the union of events is the sum of individual event probabilities. But this is not actually the case. For example, for two-dimensionally linearly separable data with separation intervals, several parallel separation planes (lines) that correctly classify the training set and are close to each other have very similar characteristics. For most data points, the results under these two separation planes are the same, and the probability distribution of generalization ability differences also has significant overlap. Treating them as independent is inappropriate.
- Let's look again at M in the inequality, which literally means the capacity of the hypothesis space, actually a measure of the hypothesis space's expression capability. Under a certain amount of training data, the richer the hypothesis space's expression ability, the harder it is for the learning algorithm to find a good hypothesis. The capacity of the hypothesis space is the sum of all hypotheses, where one hypothesis can be seen as a set of parameters under a specific model. Is there a more effective measure of the hypothesis space's expression ability?
- What if we look at the model's classification results instead of the model parameters themselves? For example, previously we considered hypotheses with the same parameters (same line) as the same hypothesis. Now, we consider hypotheses with the same classification results (different lines that classify points the same way) as the same hypothesis. This seems more reasonable because expression capability ultimately falls on handling all possible classification results of the data, and both empirical and generalization errors are measured through misclassified points.
- Here we introduce several terms:
  - Dichotomy: Denoted as $h\left(X_{1}, X_{2}, \ldots, X_{N}\right)$. The dichotomy of N points is a classification result of N points. For binary classification, there can be at most $2^N$ dichotomies.
  - Dichotomies of hypothesis space H on N points in training set D, denoted as $\mathcal{H}\left(X_{1}, X_{2}, \ldots, X_{N}\right)$. The number of dichotomies depends not only on the dataset but also on the hypothesis space H, because the hypothesis space may not achieve all dichotomies. For example, for four points forming a square with diagonal labels the same, a linear hypothesis space cannot achieve this dichotomy.
  - Growth function: The number of dichotomies in the hypothesis space on the training set depends not only on the training set size but also on the specific training set. For example, if the extracted training set does not include (four points forming a square with diagonal labels the same), the number of dichotomies for lines on this training set might be the same as the maximum possible dichotomies for this training set. Therefore, to exclude the influence of specific training sets, we define the maximum number of dichotomies the hypothesis space H can achieve on all possible training sets of size N as the growth function: $m_{\mathcal{H}}(N)=\max _{X_{1}, X 2, \ldots, X_{N} \in \mathcal{X}}\left|\mathcal{H}\left(X_{1}, X_{2}, \ldots, X_{N}\right)\right|$
  - Shatter: When the growth function reaches the theoretically maximum dichotomy of $2^N$. The meaning here is that this hypothesis space is good enough for a dataset of size N (for any dataset of size N in this task setting), capable of considering all situations and providing corresponding hypotheses for any classification result. That is, for a dataset with total capacity N, the model can achieve a perfect solution.
  - Break point: Obviously, the smaller N is, the easier it is to shatter; the larger N is, the harder it is to shatter. Starting from N=1 and gradually increasing until a critical point N=k where the hypothesis space cannot shatter these k points, we call k the break point of this hypothesis space. It can be proven that for N>k, shattering is impossible.
- We can see that the growth function is actually a measure of the hypothesis space's expression capability. VC dimension considers using this measure to replace the hypothesis space capacity in the Hoeffding inequality. Of course, it cannot be directly substituted. Specifically:
  
  $$
  \forall g \in \mathcal{H}, \mathbb{P}\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 4 m_{\mathcal{H}}(2 N) \exp \left(-\frac{1}{8} \epsilon^{2} N\right)
  $$
- The right side is the VC bound, which is complex to prove and omitted. Now let's consider whether the upper bound is finite. We can see that the important M has been replaced by $m_H$, the growth function. If the growth function is bounded or its growth rate with N is lower than the reduction rate of the exponential part, we can say that the upper bound of the difference between empirical and generalization errors is finite. In fact, if a break point k exists, then:
  
  $$
  m_{\mathcal{H}}(N) \leq B(N, k) \leq \sum_{i=0}^{k-1}\left(\begin{array}{c}
N \\
i
\end{array}\right) \leq N^{k-1}
  $$
- Where B(N, k) is the upper bound of the growth function when the break point is k. We can see that the upper bound of the growth function is polynomial-level (k-1 power), so the growth function is at most polynomial-level, while the VC bound's $\exp \left(-2 \epsilon^{2} N\right)$ is exponential-level. Clearly, the VC bound exists and is finite.
- Finally, for cases where a break point exists, we can say that generalization ability is guaranteed, and machine learning is viable.

# VC Dimension

- The VC bound ensures the feasibility of learning, while VC dimension considers the expression capability of the hypothesis space. We previously discussed the growth function as a measure of the hypothesis space's expression capability, and the two are closely related.
- The VC dimension is defined as follows: Given a hypothesis space H with an existing break point, the VC dimension is the size of the largest dataset that can be shattered, i.e.:
  
  $$
  V C(\mathcal{H})=\max \left\{N: m_{\mathcal{H}}(N)=2^{N}\right\}
  $$
- Recall the two concepts in the VC dimension definition: growth function and shattering. The growth function defines the hypothesis space's ability to solve dichotomies, and shattering represents the hypothesis space's ability to handle a certain amount of dataset, with a corresponding hypothesis for each possible dichotomy. The VC dimension measures the hypothesis space's capability from the perspective of dataset size. The more complex the hypothesis space and the stronger its expression capability, the larger the dataset it can shatter, with sufficient hypotheses to solve every possible dichotomy on a larger dataset - note that this is not every theoretically possible dichotomy, because the VC dimension takes the maximum dataset size that can actually be shattered, meaning the largest existing dataset size that can be shattered by this hypothesis space.
- So what exactly is the VC dimension? We actually defined it earlier: it's k-1, meaning datasets with capacity less than the break point can be shattered, so the maximum dataset size that can be shattered is the break point - 1.
- Notice that this k-1 is the polynomial degree in the growth function's upper bound of the VC bound, so the VC bound can also be written as:
  
  $$
  \forall g \in \mathcal{H}, \mathbb{P}\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 4(2 N)^{V C(\mathcal{H})} \exp \left(-\frac{1}{8} \epsilon^{2} N\right)
  $$

# Measuring Generalization Ability

- From the VC bound inequality, we can see that the difference between empirical error and generalization error (measuring generalization ability) is associated with the probability of bad cases on the right side of the inequality. If we specify the probability of bad cases occurring as:
  
  $$
  4(2 N)^{V C(\mathcal{H})} \exp \left(-\frac{1}{8} \epsilon^{2} N\right)=\delta
  $$
  
  We can conversely calculate the difference measuring generalization ability:
  
  $$
  \epsilon=\sqrt{\frac{8}{N} \ln \left(\frac{4(2 N)^{V C(\mathcal{H})}}{\delta}\right)}
  $$
- Therefore, in machine learning foundations and techniques, teachers often write a VC bound to estimate the generalization error formula:
  
  $$
  E_{\text {out }}(\mathbf{w}) \leq E_{\text {in }}(\mathbf{w})+\Omega(\mathcal{H})
  $$
- Where $\Omega(\mathcal{H})$ is $\sqrt{\frac{8}{N} \ln \left(\frac{4(2 N)^{V C(\mathcal{H})}}{\delta}\right)}$

# References

- https://zhuanlan.zhihu.com/p/59113933
- https://www.coursera.org/learn/ntumlone-mathematicalfoundations
{% endlang_content %}

{% lang_content zh %}
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

{% endlang_content %}


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
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>
