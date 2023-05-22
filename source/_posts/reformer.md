---
title: Reformer - Paper Reading
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

Reformer论文解读

<!--more-->

# 多快好省
-	作者主要提出了两点操作来降低Transformer，尤其是在处理超长序列时的内存占用，减少了大量运算，提升了速度。

# LSH Attention
[![39Y8pt.png](https://s2.ax1x.com/2020/02/16/39Y8pt.png)](https://imgchr.com/i/39Y8pt)
-	这一部分最原始的想法就是，Transformer当中的self attention，每一个token作为query时，要把序列中所有token当成key去计算注意力，再在所有token上加权得到当前token的一个表示，但我们知道注意力一般是非常稀疏的，权重就集中于少数几个token上，那不如只在这几个token上计算权重并加权，这样就大大减少了self attention里$O(N^2)$的计算量和内存占用量。
-	那么怎么才知道那少数几个token是哪几个？假如要完全靠注意力计算出来才能得到的话，怎么可能在计算注意力之前就知道哪几个token权重大？是不可能，但是在self attention里，query和key计算权重，就是简单的内积，和query相似的key权重大。模型学习到注意力，是指学习到生成正确的query以及key的表示，在计算注意力时只需要比对query和key就可以了。
-	所以问题转换成，对每一个query，我先找到相近的几个key计算注意力就好了。怎么找？当然不是全部算一遍取top k，那就与我们减少计算量的初衷相悖，在这里作者用到了Local Sensitive Hashing(LSH)，局部敏感哈希，大意就是相近的向量，映射到同一哈希值的概率较大，多个相近的、映射到同一哈希值的向量相当于装进了同一个桶里(bucket)，那么我们只需要对每个桶里的向量计算self attention。详细一点的描述是，两个向量$q_1,q_2$，满足LSH的哈希函数$h$能做到
	$$
	for \ dis(q_1,q_2) <= d_1 , \ p(h(q_1)==h(q_2)) >= p_1 \\
	for \ dis(q_1,q_2) >= d_2 , \ p(h(q_1)==h(q_2)) <= p_2 \\
	$$
-	相关领域已经有很多研究，对于不同的距离度量$dis$，有不同的$h$满足LSH。显然在这里我们的距离度量是cosine距离，对应的LSH哈希是球形投影，即将向量投影到一个b维超球面上，该球面被分成了$n_{buckets}$个象限，投影到同一象限的向量即在同一个桶中，该投影哈希具体写出来是：
	$$
	h(x) = argmax[xR;-xR] \\
	$$
	$R$是一个$[d_k,b/2]$的随机投影矩阵
-	接下来的一个问题是，一个桶里面，query和key的数量不一定相等，而且有可能一个桶里许多query，没有key。于是作者干脆share QK，即令query和key相同，都是embedding从同一个线性变换出来的，只不过key做了归一化操作$k_{j}=\frac{q_{j}}{\left\|q_{j}\right\|}$
-	chunk操作：接下来作者并不是让每个桶里分别做self attention，而是做了分段，即把同一个桶里的放在一起，重新排成一个序列，然后等长切成若干个段，段内做self attention，相邻的段也做一次attention。这里其实有点疑问，论文的图画的非常理想，每个桶的大小差不多，可能差了一两个可以通过相邻段做attention来弥补，但是实际情况并不知道每个桶的大小。也许是因为attention本身也是学习出来的，作者这么人为设置，是不是相当于给了一个每个桶大小都趋于相同且等于段长的先验限制了attention的学习。
-	Multi-round lsh：lsh是讲概率的，有概率就有误差，作者构造了一个巧妙的实验来验证lsh对原始attention的还原度，发现单轮的效果并不好。因此就多次hash来保证概率，取多轮hash的并集来保证相似的向量能落到同一个桶里。这里取并集而不是交集，个人理解是桶一多，hash其实很稀疏，不相似的向量落在同一个桶的概率远小于相似的向量落在不同桶的概率。这里还有一些细节待补充
-	casual masking：正常的transformer在decoder端是要做时序掩码的，这里lsh把序列顺序打乱了，因此也要做对应的处理，保证时序掩码的正确性。
-	值得一提的是大部分self attention的实现，value包括了自身，但是在lsh里不能包含自身，因为key和value共享值，自身永远是最相似的。

# Reversible Transformer
-	这一部分的想法参照了论文：The Reversible Residual Network: Backpropagation Without Storing Activations。
-	基本思想就是，将残差结构改为可逆残差结构，从而节省了显存。网络在做方向传播的时候，需要存储每一层的激活值，带入自动微分计算每一层的导数，再链式求导，其中存储每一层的激活值占了很大的显存。可逆残差的思想就是，通过将channel一分为二，做成两路，互相残差，更改计算图的拓扑结构，使得两路的激活值能够通过上一层的激活值计算出来，如图：	
	![39l0tP.png](https://s2.ax1x.com/2020/02/16/39l0tP.png)
-	前向传播过程为：
	$$
	\begin{array}{l}{y_{1}=x_{1}+\mathcal{F}\left(x_{2}\right)} \\ {y_{2}=x_{2}+\mathcal{G}\left(y_{1}\right)}\end{array}
	$$
-	反向传播为：
	$$
	\begin{array}{l}{x_{2}=y_{2}-\mathcal{G}\left(y_{1}\right)} \\ {x_{1}=y_{1}-\mathcal{F}\left(x_{2}\right)}\end{array}
	$$
-	可以看到计算$x_2$时只用了上一层的激活值$y_1,y_2$，计算$x_1$时用了上一步计算出来的$x_1$，因此不需要存储这两个激活值。虽然节省了空间，但是激活函数需要重新算一遍，相当于用时间换空间。
-	原始论文用在resnet里，节约显存可以换得更大的batch_size，在transformer中就可以用来训练更长的sequence
-	reformer中把两个函数$\mathcal{F}$和$\mathcal{G}$分别改成了自注意力层和全连接层，这样就对应了transformer的可逆结构
-	可逆结构虽然消除了层数对于空间复杂度的影响，但是transformer里占显存最大的FFN，其依然受序列长度影响，为了减少这一部分显存占用，作者有一次采用了chunking，因为FFN这里是不存在序列依赖的，完全可以拆成几段计算，相应的，可逆结构的输入输出也拆成几段计算，又一次用时间换空间。此外，对于词典较大的应用场景，作者在计算损失log-probabilities时也是分段的。
-	作者还额外提到，这样节省的是反向传播计算梯度时用到的中间临时变量，并不会节省参数量，节省参数量在GPU的消耗可以通过将其转到CPU内存来解决，通常这样的操作得不偿失，因为在CPU和GPU之间传输数据非常耗时，但是由于reformer在每次转换时可以处理更多的数据，就“得能尝失”了。