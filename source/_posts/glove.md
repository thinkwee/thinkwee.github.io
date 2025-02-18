---

title: Glove Embedding - Mathematical Derivation 
date: 2019-01-13 09:42:37
categories: ML
tags:

- glove
- math
- word embedding
mathjax: true
html: true

---

***

*   Record the mathematical derivation of GloVe word vectors, as the original paper does not derive the model graphically but rather calculates the objective function through pure mathematical operations. This design approach is very interesting, and it also writes out and compares the mathematical essence of word2vec.
*   GloVe: Global Vectors for Word Representation

<!--more-->

{% language_switch %}

{% lang_content en %}
Word vectors
============

*   Whether based on global matrix factorization or local window-based word vectors, the method of extracting semantics is to mine meaning from the co-occurrence statistical information between words.
*   Clearly, the global approach does not make use of the advantages of the local one: for example, global techniques such as LSA are insensitive to local contextual information, making it difficult to mine synonyms based on context; the local approach does not make use of the advantages of the global one, as it only relies on independent local contexts, and if the window is too small, it cannot effectively utilize the information of the entire document or corpus.
*   The GloVe approach is to utilize the global word co-occurrence matrix while also calculating relevance using local contextual relationships.
*   The result of word vectors is a mapping that generates meaningful semantic relationships based on distance relationships. To achieve this goal, GloVe designed a log-bilinear regression model and specifically adopted a weighted least mean square regression model to train word vectors.

Discovery
=========

*   Definition:
    *   For a single word.
    *   The number of occurrences of $x_j$ in the context of $x_i$ .
    *   The number of times all words appear in the context of $x_i$ .
    *   The probability of $x_j$ appearing in the context of $x_i$ , which is the probabilization of the frequency count of the context occurrence, referred to as "co-occurrence probabilities" in the paper.
    *   $r = \frac {P_{ik}}{P_{jk}}$ : Introduce an intermediate word $x_k$ , referred to as "probe word" in the paper, by introducing this $x_k$ , it can indirectly measure the relationship between $x_i$ and $x_j$ , represented by $r$ , i.e., the ratio.
*   The role of introduction is reflected in two aspects:
    *   For the $x_i$ and $x_j$ to be compared, filter out the $x_k$ without discriminative power, which is noise. When $r \approx 1$ , $x_k$ is considered noise.
    *   Given $x_k$ , such that those $r >> 1$ of $x_i$ have similar meanings, and those $r << 1$ of $x_j$ have similar meanings.
*   Therefore, we can filter out noise and only mine word sense relationships from co-occurrence data where $r$ is very large or very small.

Design
======

*   Next, the author directly applies the target design function.
    
*   The goal is: the distance calculation results between the word vectors designed should reflect the ratio previously discovered from the word co-occurrence matrix, specifically for the triplet, words i, j, and the probe word k, the word vectors of these three words should embody r
    
*   Then, directly designing, defining $w_i$ as the word vector corresponding to $x_i$ , we assume $F$ to be the function for calculating distance:
    
    $$
    F(w_i,w_j,w^{*}_k) = r \\
    = \frac {P_{ik}}{P_{jk}} \\
    $$
    
*   The word vectors of above $w_k$ are distinguished by asterisks from the word vectors of $w_i$ and $w_j$ , because $w_k$ is an independent context word vector, parallel to the required word vectors, similar to the forward and backward word embedding matrices in word2vec.
    
*   Next, a natural thought is to reduce the parameters, i.e., only the word vectors and the context word vectors are needed, because it is a distance calculation function and the vector space is a linear space; we use the vector difference between $w_i$ and $w_j$ as the parameters:
    
    $$
    F(w_i - w_j,w^{*}_k) = \frac {P_{ik}}{P_{jk}} \\
    $$
    
*   The current function takes a vector as its parameter, and outputs a tensor. The simplest structure is to perform a dot product:
    
    $$
    F((w_i-w_j)^T w^{*}_k) = \frac {P_{ik}}{P_{jk}} \\
    $$
    
*   The next key point is symmetry. Noticing that although context and non-context word vectors are distinguished, since the co-occurrence matrix $X$ is symmetric, the two sets of word vectors $w$ and $w^{\*}$ should have the same effect. This is because the values of the two sets of word vectors are different due to different random initialization, but they should be the same in terms of measuring similarity, that is, $w_i^T w^{\*}_j$ and $w_j^T w^{\*}_i$ should be the same.
    
*   Due to symmetry, $x_i,x_j,x_k$ can be any word in the corpus, so the two parameters of the $F$ function should be interchangeable ( $w$ and $w^{\*}$ , $X$ and $X^T$ ), and here a bit of mathematical technique is further applied to symmetrize the function:
    
    *   Design:
        
        $$
        F((w_i-w_j)^T w^{*}_k) = \frac {F(w_i w^{*}_k)} {F(w_j w^{*}_k)} \\
        $$
        
    *   Then both the numerator and denominator are of the same form, that is
        
        $$
        F(w_i w^{*}_k) = P_{ik} = \frac {X_{ik}} {X_i} \\
        $$
        
    *   To satisfy the above $F$ , it can be decomposed into two sub- $F$ , and then $F$ can be the $exp$ function, i.e
        
        $$
        w_i^T w_k^{*} = log(X_{ik}) - log {X_i} \\
        $$
        
    *   The indices k, i, j can be interchanged without changing the meaning. Since the numerator and denominator have the same form, we only need to ensure that this form is satisfied; the fraction will naturally satisfy the mapping from the triplet to the ratio thereafter.
        
    *   Noticing that in the above formula, the inner product of the two vectors on the left remains unchanged when the i,k symbols are interchanged, while the subtraction of the two log expressions on the right does not satisfy this symmetry. Therefore, we add an $log{x_k}$ to make it symmetric and simplify it to the bias $b^{*}$ . Similarly, after interchanging the i,k symbols, we add an $Log{x_i}$ to make it symmetric, i.e., the bias $b_i$ . The bias, like word vectors, also consists of two sets:
        
        $$
        w_i^Tw_k^{*} + b_i + b_k^{*} = log(X_{ik}) \\
        $$
        
    *   Finally, add smoothing to prevent the log parameter from being 0:
        
        $$
        w_i^Tw_k^{*} + b_i + b_k^{*} = log(1 + X_{ik}) \\
        $$
        
*   Here we have preliminarily completed the design of the $F$ function, but there is still an issue that it averages the weights of each co-occurrence, while in general corpora, most co-occurrences have very low frequencies
    
*   The solution for Glove is to use weighted functions. After weighting, the training of word vectors is regarded as a least mean square error regression of the F function, and the loss function is designed:
    
    $$
    J = \sum _{i,j}^V f(X_{ij}) (w_i^T w_j^{*} + b_i + b_j^{*} - log (1 + X_{ij}))^2 \\
    $$
    
*   Among which, f is the weighted function, with its parameters being the co-occurrence frequency; the author points out that this function must satisfy three properties:
    
    *   Clearly, if no co-occurrence occurs, the weight is 0.
    *   Non-decreasing: The higher the co-occurrence frequency, the greater the weight.
    *   relatively small for large X: To prevent over-weighting for certain common co-occurrences with high frequencies, which may affect the results.
*   Based on the above three properties, the author designed a truncated weighted function within the threshold $X_{max}$
    
    $$
    f(x) = (\frac {x}{X_{max}}) ^ {\alpha} \\
    $$
    
    If exceeding the threshold, the function value is 1.
    

Comparing with Word2vec
=======================

*   For the skip-gram model in Word2vec, the goal is to maximize the probability of predicting the correct central word given the context, which is generally probabilized through the softmax function, i.e.:
    
    $$
    Q_{ij} = \frac {exp (w_i^T w_j^{*})} { \sum _{k=1}^V exp(w_i^T w_k^{*})} \\
    $$
    
*   Through gradient descent, the overall loss function can be written as:
    
    $$
    J = - \sum _{i \in corpus , j \in context(i)} log Q_{ij} \\
    $$
    
*   Group the same $Q_{ij}$ first and then sum up to get:
    
    $$
    J = - \sum _{i=1}^V \sum _{j=1}^V X_{ij} log Q_{ij} \\
    $$
    
*   Next, further transformations are made using the previously defined symbols:
    
    $$
    J = - \sum _{i=1^V} X_i \sum _{j=1}^V P_{ij} log Q_{ij} \\
    = \sum _{i=1}^V X_i H(P_i,Q_i) \\
    $$
    
*   That is to say, the loss function of Word2vec is actually weighted cross-entropy, however, cross-entropy is only one possible measure and has many drawbacks:
    
    *   Probability requiring normalization as a parameter
    *   Softmax computation is computationally intensive, referred to as the model's computational bottleneck
    *   For long-tailed distributions, cross-entropy often assigns too much weight to less likely items
*   Solution to the above problems: Simply do not normalize, directly use co-occurrence counts, do not use cross-entropy and softmax, directly use mean squared error, let $Q_{ij} = exp(w_i^T w_j^{*})$ , $P_{ij} = X_{ij}$ , then:
    
    $$
    J = \sum _{i,j} X_i (P_{ij} - Q_{ij})^2 \\
    $$
    
*   However, non-normalization can cause numerical overflow, so take the logarithm again:
    
    $$
    J = \sum _{i,j} X_i (log P_{ij} - log Q_{ij})^2 \\
    =  \sum _{i,j} X_i (w_i^T w_j^{*} - log X_{ij})^2 \\
    $$
    
*   Thus, the simplest objective function of GloVe is obtained.
    
*   The authors of Word2vec found that filtering out some common words could improve the effectiveness of word vectors, and the weighted function in Word2vec is denoted as $f(X_i)=X_i$ , thus filtering out common words is equivalent to designing a non-decreasing weighted function. GloVe designed a more sophisticated weighted function.
    
*   Therefore, from the perspective of mathematical derivation, GloVe simplifies the objective function of Word2vec, replacing cross-entropy with mean squared error and redesigning the weighting function.
    

Concept
=======

*   The paper provides a good idea for designing a model, namely, designing the objective function based on evaluation indicators, and then training the model to obtain the parameters (by-products) as the desired results.


{% endlang_content %}

{% lang_content zh %}

# 词向量

- 无论是基于全局矩阵分解的还是基于局部窗口的词向量，其提取semantic的方式都是从词与词的共现统计信息中挖掘意义。
- 显然，全局的方式没有利用到局部的优点：全局例如LSA等技术对于局部上下文信息不敏感，难以根据上下文挖掘近义词；局部的方式没有利用到全局的优点，它只依赖于独立的局部上下文，窗口太小的话不能有效利用整个文档乃至语料的信息。
- Glove的思路是利用全局的词与词共现矩阵，同时利用局部上下文关系计算相关性。
- 词向量的结果是能产生有意义的语义关系到距离关系的映射，针对这个目标，Glove设计了一个log-bilinear回归模型，并具体采用一个加权最小均方回归模型来训练词向量。

# 发现

- 定义：
  - $x$：为单个词。
  - $X_{ij}$：$x_j$ 出现在$x_i$的上下文中的次数。
  - $X_i = \sum _k x_{ik}$：所有词出现在$x_i$的上下文中的次数。
  - $P_{ij} = P(j|i) = \frac {x_{ij}} {X_i}$：$x_j$出现在$x_i$的上下文中的概率，即上下文出现频次计数概率化，论文中称之为"co-occurrence probabilities"。
  - $r = \frac {P_{ik}}{P_{jk}}$：引入中间词$x_k$，论文中叫"probe word"，通过引入这个$x_k$可以间接的衡量$x_i$和$x_j$的关系，通过$r$即ratio表示。
- $r$引入的作用体现在两个方面：
  - 对于要比较的$x_i$和$x_j$，筛除对于没有区分度的$x_k$，也就是噪音。当$r \approx 1$时，$x_k$即为噪音。
  - 给定$x_k$，使得$r >> 1$的那些$x_i$具有相近的词义，使得$r << 1$的那些$x_j$具有相近的词义。
- 因此，我们可以过滤噪音，仅仅在$r$很大或很小的词共现数据中挖掘词义关系。

# 设计

- 接下来，作者直接根据目标设计函数。
- 目标是：设计出来的词向量之间的距离计算结果应该能够反映之前我们从词共现矩阵中发现的ratio，具体而言是对于三元组，词i,j和probe word k，这三个词的词向量能够体现r
- 那么直接设计,定义$w_i$为$x_i$对应的词向量，则假设$F$为计算距离的函数：
  
  $$
  F(w_i,w_j,w^{*}_k) = r \\
= \frac {P_{ik}}{P_{jk}} \\
  $$
- 上面$w_k$的词向量加了星号区别于$w_i$和$w_j$的词向量，因为$w_k$是独立的上下文词向量，与我们需要的词向量是平行的两套，类似于word2vec里面的前后词嵌入矩阵。
- 接下来，一个自然的想法是，减少参数，即只需要词向量和上下文词向量，因为是距离计算函数且向量空间是线性空间，我们使用$w_i$和$w_j$的向量差作为参数：
  
  $$
  F(w_i - w_j,w^{*}_k) = \frac {P_{ik}}{P_{jk}} \\
  $$
- 现在函数的参数是向量，输出是张量，最简单的一个结构就是做点乘：
  
  $$
  F((w_i-w_j)^T w^{*}_k) = \frac {P_{ik}}{P_{jk}} \\
  $$
- 接下来的一个关键点：对称。注意到虽然区分了上下文和非上下文词向量，但是由于共现矩阵$X$是对称的，因此两套词向量$w$和$w^{\*}$应该具有相同的效果，只是由于随机初始化不同，两套词向量的值不一样，在衡量相似度时应该是一样的目标，即$w_i^T w^{\*}_j$和$w_j^T w^{\*}_i$一样。
- 由于对称性，$x_i,x_j,x_k$可以是语料中任意词，因此$F$函数的两个参数应该是可以交换位置（$w$和$w^{\*}$，$X$和$X^T$），那这里进一步运用了一点数学技巧将函数对称化：
  - 设计：
    
    $$
    F((w_i-w_j)^T w^{*}_k) = \frac {F(w_i w^{*}_k)} {F(w_j w^{*}_k)} \\
    $$
  - 那么分子分母都是一样的形式，即
    
    $$
    F(w_i w^{*}_k) = P_{ik} = \frac {X_{ik}} {X_i} \\
    $$
  - 要满足上面$F$可以拆分为两个子$F$的比，则$F$可以为$exp$函数，即
    
    $$
    w_i^T w_k^{*} = log(X_{ik}) - log {X_i} \\
    $$
  - 这样k,i,j下标可互换位置且表达意思一致。由于分子分母形式一致，因此我们只要关注这个形式能够满足就行了，之后求分数自然会满足从三元组到ratio的映射。
  - 注意到上面式子当中，左边的两个向量内积，i,k符号互换值不变，而右边的两个log式子相减并不满足这种对称，因此我们补上一个$log{x_k}$使之对称，并将其简化为偏置$b^{*}$，同样的道理，i,k符号互换后，补上一个$Log{x_i}$使之对称，即偏置$b_i$，偏置和词向量一样，也是两套：
    
    $$
    w_i^Tw_k^{*} + b_i + b_k^{*} = log(X_{ik}) \\
    $$
  - 最后加上平滑，防止log的参数取0：
    
    $$
    w_i^Tw_k^{*} + b_i + b_k^{*} = log(1 + X_{ik}) \\
    $$
- 到这里我们已经初步完成了$F$函数的设计，但这个还存在的一个问题是，它是平均加权每一个共现的，而一般语料中大部分共现都频次很低
- Glove的解决办法是使用加权函数。加权之后将词向量的训练看成是F函数的最小均方误差回归，设计损失函数：
  
  $$
  J = \sum _{i,j}^V f(X_{ij}) (w_i^T w_j^{*} + b_i + b_j^{*} - log (1 + X_{ij}))^2 \\
  $$
- 其中f为加权函数，其参数是共现频次，作者指出该函数必须满足三条性质：
  - $f(0)=0$：显然，没有出现共现则权重为0。
  - Non-decreasing：共现频次越大则权重越大。
  - relatively small for large X：防止对于某些频次很高的常见共现加权过大，影响结果。
- 基于以上三种性质，作者设计了截尾的加权函数，在阈值$X_{max}$以内：
  
  $$
  f(x) = (\frac {x}{X_{max}}) ^ {\alpha} \\
  $$
  
  超过阈值则函数值为1.

# 与Word2vec比较

- 对于Word2vec中的skip-gram模型，其目标是最大化给定上下文之后预测正确中心词的概率，一般通过softmax函数将其概率化，即：
  
  $$
  Q_{ij} = \frac {exp (w_i^T w_j^{*})} { \sum _{k=1}^V exp(w_i^T w_k^{*})} \\
  $$
- 通过梯度下降求解，则整体损失函数可以写成：
  
  $$
  J = - \sum _{i \in corpus , j \in context(i)} log Q_{ij} \\
  $$
- 将相同的$Q_{ij}$先分组再累加，得到：
  
  $$
  J = - \sum _{i=1}^V \sum _{j=1}^V X_{ij} log Q_{ij} \\
  $$
- 接下来用之前定义的符号进一步变换：
  
  $$
  J = - \sum _{i=1^V} X_i \sum _{j=1}^V P_{ij} log Q_{ij} \\
= \sum _{i=1}^V X_i H(P_i,Q_i) \\
  $$
- 也就是说，Word2vec的损失函数实际上是加权的交叉熵，然而交叉熵只是一种可能的度量，且具有很多缺点：
  - 需要归一化的概率作为参数
  - softmax计算量大，称为模型的计算瓶颈
  - 对于长尾分布，交叉熵常常分配给不太可能的项太多权重
- 解决以上问题的方法：干脆不归一化，直接用共现计数，不用交叉熵和softmax，直接用均方误差，令$Q_{ij} = exp(w_i^T w_j^{*})$，$P_{ij} = X_{ij}$，则：
  
  $$
  J = \sum _{i,j} X_i (P_{ij} - Q_{ij})^2 \\
  $$
- 但是不归一化会造成数值上溢，那就再取个对数：
  
  $$
  J = \sum _{i,j} X_i (log P_{ij} - log Q_{ij})^2 \\
=  \sum _{i,j} X_i (w_i^T w_j^{*} - log X_{ij})^2 \\
  $$
- 这样就得到了Glove最朴素的目标函数。
- Word2vec的作者发现筛除一些常见词能够提高词向量效果，而Word2vec中的加权函数即$f(X_i)=X_i$，因此筛除常见词等价于设计一个非降的加权函数。Glove则设计了更为精巧的加权函数。
- 因此从数学公式推导上看，Glove简化了Word2vec的目标函数，用均方误差替换交叉熵，并重新设计了加权函数。

# 思路

- 该文提供了一个很好的设计模型的思路，即根据评测指标设计目标函数，反过来训练模型，得到函数的参数（副产品）作为所需的结果。

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