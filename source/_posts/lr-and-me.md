---
title: Logistic Regression and Maximum Entropy
date: 2018-10-14 20:38:59
tags: [logistic regression,math,machine learning]
categories: ML
mathjax: true
html: true
---

Note for John Mount's "The Equivalence of Logistic Regression and Maximum Entropy Models" and explains that this proof is a special case of the general derivation proof of the maximum entropy model introduced in statistical learning methods

Conclusion

*   Maximum entropy model is softmax classification
*   Under the balanced conditions of the general linear model, the model mapping function that satisfies the maximum entropy condition is the softmax function
*   In the book on Statistical Machine Learning methods, a maximum entropy model defined under the feature function is presented, which, along with softmax regression, belongs to the class of log-linear models
*   When the feature function extends from a binary function to the feature value itself, the maximum entropy model becomes a softmax regression model
*   The maximum entropy maximizes conditional entropy, not the entropy of conditional probabilities, nor the entropy of joint probabilities.

<!--more-->

{% language_switch %}

{% lang_content en %}

Define symbols
=============

*   n-dimensional features, m samples, $x(i)_j$ denotes the j-th feature of the i-th sample, discuss the multi-class scenario, the output classification $y(i)$ has k classes, the mapping probability function $\pi$ maps from $R^n$ to $R^k$ , we hope $\pi(x(i))_{y(i)}$ to be as large as possible.
*   Indicator function $A(u,v)$ , equals 1 when $u==v$ and 0 otherwise

Logistic regression
===================

$$
\pi(x)_1 = \frac{e^{\lambda x}}{1+e^{\lambda x}} \\
\pi(x)_2 = 1 - \pi(x)_1\\
$$

*   The parameter to be learned $\lambda$ is $R^n$

Softmax regression
==================

$$
\pi(x)_v = \frac{e^{\lambda _v x}} {\sum _{u=1}^k e^{\lambda _u x}}
$$

*   For $R^{k * n}$

Solving softmax
===============

*   When using softmax or logistic as nonlinear functions, they possess a good property of differentiation, that is, the derivative function can be expressed in terms of the original function
    
*   We can now define the objective function, which is to maximize the correct category probability output by the $\pi$ function (maximum likelihood), and define the optimization obtained by $\lambda$ :
    
    $$
    \lambda = argmax \sum _{i=1}^m log (\pi (x(i))_{y(i)}) \\
    = argmax f(\lambda) \\
    $$
    

Balanced Equation
=================

*   Derive the objective function above and set the derivative to 0:
    
    $$
    \frac {\partial f(\lambda)}{\partial \lambda _{u,j}} = \sum _{i=1，y(i)=u}^m x(i)_j - \sum _{i=1}^m x(i)_j \pi (x(i))_u =0 \\
    $$
    
*   Thus, we obtain an important balance equation:
    
    $$
    \ \  for \ all \ u,j \\
    \sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
    $$
    
*   Analyze this equation:
    
    *   Plain Language: We hope to obtain a mapping function $\pi$ , such that for a certain dimension (j) feature, the sum of the weighted feature values of all samples mapped to the u class by the mapping function is equal to the sum of the feature values of all samples within the u class. It is obvious that the best case is that the elements within both summation expressions are completely identical, only the samples of the u class are summed, and the probability that the mapping function maps the u class samples to the u class is 1, while the probability that samples of other classes are mapped to the u class is 0.
        
    *   However, this equation is very lenient, requiring only that the two sums be equal, without demanding that each element be the same, and the expression of the mapping function is not explicitly written out. Any nonlinear mapping that satisfies this equation could be called a mapping function.
        
    *   In formulaic terms, it is expressed as
        
        $$
        \sum _{i=1}^m A(u,y(i)) x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
        \pi (x(i))_u \approx A(u,y(i)) \\
        $$
        

From Maximum Entropy to Softmax
===============================

*   What was mentioned above is that the balanced equation does not require the format of the mapping function, then why did we choose softmax? In other words, under what conditions can the constraint of the balanced equation lead to the conclusion that the nonlinear mapping is softmax?
    
*   The answer is maximum entropy. Now let's review the conditions that need to be met in $\pi$ .
    
    *   Balance equation (i.e., this $\pi$ can fit the data):
        
        $$
        \ \  for \ all \ u,j \\
        \sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
        $$
        
    *   The output of $\pi$ should be a probability:
        
        $$
        \pi (x)_v \geq 0 \\
        \sum _{v=1}^k \pi (x)_v = 1 \\
        $$
        
*   According to the maximum entropy principle, we hope that $\pi$ can have the maximum entropy while satisfying the aforementioned constraints:
    
    $$
    \pi = argmax \ Ent(\pi) \\
    Ent(\pi) = - \sum_{v=1}^k \sum _{i=1}^m \pi (x(i))_v log (\pi (x(i))_v) \\
    $$
    
*   The maximum entropy can be understood from two perspectives:
    
    *   Maximum entropy, also known as maximum perplexity, refers to the low risk of overfitting in the model, with low model complexity. According to Ockham's Razor principle, among multiple models with the same effect, the one with lower complexity has better generalization ability. Under the satisfaction of constraint conditions, of course, we would hope for a model with lower complexity, which is equivalent to regularization.
    *   The constraints are the parts of our model that are known to need to be satisfied and need to be fitted; the remaining parts are the unknown parts, with no rules or data to guide us in assigning probabilities. What should we do in this unknown situation? In the case of the unknown, probabilities should be uniformly distributed among all possibilities, which corresponds to the maximum entropy situation.
*   The problem has now been formulated as a constrained optimization problem, which can be solved using the Lagrange multiplier method. There is a trick; the original text states that it would be somewhat complex to directly consider the probabilistic inequality conditions, and the KKT conditions would need to be used, which we will not consider here. If the $\pi$ obtained satisfies the inequality conditions, we can skip it (which is indeed the case).
    

$$
L = \sum _{j=1}^n \sum _{v=1}^k \lambda _{v,j} (\sum _{i=1}^m \pi (x(i))_v x(i)_j - A(v,y(i)) x(i)_j) \\
+ \sum _{v=1}^k \sum _{i=1}^m \beta _i (\pi (x(i))_v -1) \\
- \sum _{v=1}^k \sum _{i=1}^m \pi(x(i))_v log(\pi (x(i))_v) \\
$$

*   Here is another trick, where we should differentiate all parameters. Here, we first differentiate $\pi (x(i))_u$ and set it to 0 to obtain:
    
    $$
    \pi (x(i))_u = e^{\lambda _u x(i) + \beta _i -1}
    $$
    
*   Considering the equality constraint condition (the sum of probabilities equals 1), it is not necessary to differentiate with respect to $\beta$
    
    $$
    \sum _{v=1}^k e^{\lambda _v x(i) + \beta _i -1} = 1 \\
    e^{\beta} = \frac {1}{\sum _{v=1}^k e^{\lambda _v x(i) - 1}} \\
    $$
    
*   Re-substitution yields:
    
    $$
    \pi (x)_u = \frac {e^{\lambda _u}x}{\sum _{v=1}^k e^{\lambda _v}x}
    $$
    

Solving Parameters
==================

*   From the time of introducing the balanced equation, it can be seen that we need to solve $n \* k$ equations to obtain $n \* k$ parameters $\lambda$ , or take partial derivatives of $n \* k$ $\lambda$ in the Lagrange equation of maximum entropy, because $\pi$ is a non-linear function of $\lambda$ . Both of these methods are relatively difficult, but we can calculate the Jacobian equations (or the Hessian matrix of the objective function) of these equations by differentiation, and then we can solve $\lambda$ using some Newton method, Fisher Scoring, or iterative method

Connection with the Maximum Entropy Model Defined by Characteristic Functions
=============================================================================

*   In this paper, the constraint is (omitted the constraint $\pi$ must be a probability):
    
    $$
    \sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
    $$
    
*   The maximum entropy is:
    
    $$
    Ent(\pi) = - \sum_{v=1}^k \sum _{i=1}^m \pi (x(i))_v log (\pi (x(i))_v) \\
    $$
    
*   The results obtained are:
    
    $$
    \pi (x)_u = \frac {e^{\lambda _u}x}{\sum _{v=1}^k e^{\lambda _v}x}
    $$
    
*   In statistical learning methods, the constraints are (with the probability constraints similarly omitted), where $P^{*}$ represents the empirical distribution:
    
    $$
    \sum _{x,y} P^{*} (x,y)f(x,y) = \sum _{x,y} P^{*} (x)P(y|x)f(x,y)
    $$
    
*   The maximum entropy is:
    
    $$
    Ent(P) = - \sum _{x,y} P^{*}(x) P(y|x) log P(y|x)
    $$
    
*   The results obtained are:
    
    $$
    P(y|x) = \frac{e^{\sum _i w_i f_i(x,y)}}{\sum _y e^{\sum _i w_i f_i(x,y)}}
    $$
    
*   It can be seen that there is a distinction in the representation of the two; the former directly obtains the form of the softmax function, but does not maximize the conditional entropy, whereas the latter is the opposite
    
*   In fact, both are unified. Firstly, the parameters of the model are all Lagrange multipliers, the former being $\lambda$ , and the latter being $w$ , with the relationship:
    
    $$
    \lambda = \{w_0,...,w_i,...\}
    $$
    
*   When the characteristic function extends to the eigenvalue, the model obtained by both is the same (softmax function):
    
    $$
    f_i(x_j,y) = x(j)_i
    $$
    
*   The balance conditions of both are also consistent. Noticing that $P^{*}$ is an empirical distribution, which is statistically obtained through classical probability type on the training set, in general, repeated data is not considered (with a total sample size of N and a number of categories K), then:
    
    $$
    P^{*} (x) = \frac 1N \\
    \sum _{x,y} P^{*} (x,y) = 1 \\
    P^{*} (x,y) \in \{0,\frac 1N \} \\
    $$
    
*   After substitution, it will be found that the balance conditions of the two are consistent, while the calculation in the paper seems to be entropy, but it is actually conditional entropy; it merely ignores the constant condition $P^{*} (x) = \frac 1N$ from the argmax expression and writes it in the form of entropy.
    



{% endlang_content %}

{% lang_content zh %}

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

- $\lambda$为$R^{k * n}$

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
Ent(\pi) = - \sum_{v=1}^k \sum _{i=1}^m \pi (x(i))_v log (\pi (x(i))_v) \\
  $$
- 最大熵可以从两个角度理解：
  - 最大熵也就是最大困惑度，即模型过拟合的风险低，模型复杂程度低，根据奥卡姆剃刀原则，在多个具有相同效果的模型中复杂程度小的模型具有更好的泛化能力，在满足了约束条件的情况下，当然我们希望要一个复杂程度小的模型，相当于正则化。
  - 约束条件是我们的模型已知的需要满足、需要拟合的部分，剩下的部分是未知的部分，没有规则或者数据指导我们分配概率，那该怎么办？在未知的情况下就应该均匀分配概率给所有可能，这正是对应了最大熵的情况。
- 现在问题已经形式化带约束条件的最优化问题，利用拉格朗日乘子法求解即可。这里有一个trick，原文中说如果直接考虑概率的不等条件就有点复杂，需要使用KTT条件，这里先不考虑，之后如果求出的$\pi$满足不等式条件的话就可以跳过了（事实也正是如此）。

$$
L = \sum _{j=1}^n \sum _{v=1}^k \lambda _{v,j} (\sum _{i=1}^m \pi (x(i))_v x(i)_j - A(v,y(i)) x(i)_j) \\
+ \sum _{v=1}^k \sum _{i=1}^m \beta _i (\pi (x(i))_v -1) \\
- \sum _{v=1}^k \sum _{i=1}^m \pi(x(i))_v log(\pi (x(i))_v) \\
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

# 与特征函数定义的最大熵模型的联系

- 在本文中，约束为（省略了$\pi$必须为概率的约束）：
  
  $$
  \sum _{i=1，y(i)=u}^m x(i)_j = \sum _{i=1}^m x(i)_j \pi (x(i))_u \\
  $$
- 最大化的熵为：
  
  $$
  Ent(\pi) = - \sum_{v=1}^k \sum _{i=1}^m \pi (x(i))_v log (\pi (x(i))_v) \\
  $$
- 得到的结果为：
  
  $$
  \pi (x)_u = \frac {e^{\lambda _u}x}{\sum _{v=1}^k e^{\lambda _v}x}
  $$
- 而在统计学习方法中，约束为（同样省略了概率约束），其中$P^{*}$代表经验分布：
  
  $$
  \sum _{x,y} P^{*} (x,y)f(x,y) = \sum _{x,y} P^{*} (x)P(y|x)f(x,y)
  $$
- 最大化的熵为：
  
  $$
  Ent(P) = - \sum _{x,y} P^{*}(x) P(y|x) log P(y|x)
  $$
- 得到的结果为：
  
  $$
  P(y|x) = \frac{e^{\sum _i w_i f_i(x,y)}}{\sum _y e^{\sum _i w_i f_i(x,y)}}
  $$
- 可以看到两者的表示有区别，前者直接得到了softmax函数的形式，但是最大化的不是条件熵，后者则相反
- 实际上两者是统一的。首先，模型的参数都是拉格朗日乘子，前者是$\lambda$，后者是$w$，两者的关系：
  
  $$
  \lambda = \{w_0,...,w_i,...\}
  $$
- 当特征函数扩展到特征值时，两者得到的模型就是一样的（softmax函数）：
  
  $$
  f_i(x_j,y) = x(j)_i
  $$
- 两者的平衡条件也是一致的，注意到$P^{*}$是经验分布，是在训练集上通过古典概型统计出来的，一般情况下不考虑重复数据（样本总数为N，类别数为K），则有：
  
  $$
  P^{*} (x) = \frac 1N \\
\sum _{x,y} P^{*} (x,y) = 1 \\
P^{*} (x,y) \in \{0,\frac 1N \} \\
  $$
- 代入之后会发现两者的平衡条件一致，而论文中计算的貌似是熵，实际上是条件熵，只不过把$P^{*} (x) = \frac 1N $这一常量条件从argmax表达式中忽略了，写成了熵的形式。

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
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>