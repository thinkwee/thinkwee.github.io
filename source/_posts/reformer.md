---
title: Reformer论文解读
date: 2020-02-07 21:18:11
categories: 自然语言处理
tags:
  - local sensitive hashing
  - deep learning
  - transformer
  -	natural language processing
mathjax: true
html: true
---

Reformer论文解读，未完待续

<!--more-->

# 多快好省
-	作者主要提出了两点操作来降低Transformer，尤其是在处理超长序列时的内存占用，减少了大量运算，提升了速度。

# LSH Attention
-	这一部分最原始的想法就是，Transformer当中的self attention，每一个token作为query时，要把序列中所有token当成key去计算注意力，再在所有token上加权得到当前token的一个表示，但我们知道注意力一般是非常稀疏的，权重就集中于少数几个token上，那不如只在这几个token上计算权重并加权，这样就大大减少了self attention里$O(N^2)$的计算量和内存占用量。
-	那么怎么才知道那少数几个token是哪几个？假如要完全靠注意力计算出来才能得到的话，怎么可能在计算注意力之前就知道哪几个token权重大？是不可能，但是在self attention里，query和key计算权重，就是简单的内积，和query相似的key权重大。模型学习到注意力，是指学习到生成正确的query以及key的表示，在计算注意力时只需要比对query和key就可以了。
-	所以问题转换成，对每一个query，我先找到相近的几个key计算注意力就好了。怎么找？当然不是全部算一遍取top k，那就与我们减少计算量的初衷相悖，在这里作者用到了Local Sensitive Hashing(LSH)，局部敏感哈希，大意就是相近的向量，映射到同一哈希值的概率较大，多个相近的、映射到同一哈希值的向量相当于装进了同一个桶里(bucket)，那么我们只需要对每个桶里的向量计算self attention。详细一点的描述是，两个向量$q_1,q_2$，满足LSH的哈希函数$h$能做到
	$$
	for \ dis(q_1,q_2) <= d_1 , \ p(h(q_1)==h(q_2)) >= p_1 \\
	for \ dis(q_1,q_2) >= d_2 , \ p(h(q_1)==h(q_2)) <= p_2 \\
	$$
-	相关领域已经有很多研究，对于不懂的距离度量$dis$，有不同的$h$满足LSH。显然在这里我们的距离度量是cosine距离，对应的LSH哈希是球形投影，即将向量投影到一个b维超球面上，该球面被分成了$n_{buckets}$个象限，投影到同一象限的向量即在同一个桶中，该投影哈希具体写出来是：
	$$
	h(x) = argmax[xR;-xR] \\
	$$
	$R$是一个$[d_k,b/2]$的随机投影矩阵
-	接下来的一个问题是，一个桶里面，query和key的数量不一定相等，而且有可能一个桶里许多query，没有key。于是作者干脆share QK，即令query和key相同，都是embedding从同一个线性变换出来的，只不过key做了归一化操作$k_{j}=\frac{q_{j}}{\left\|q_{j}\right\|}$
-	chunk操作
-	Multi-round lsh
-	casual masking

# Reversible Transformer

# Results
