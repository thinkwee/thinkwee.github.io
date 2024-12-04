---
title: Paper Reading 3
date: 2019-01-03 16:21:42
tags:
  - abstractive summarization
  - math
  - machine learning
  -    theory
  -    nlp
categories:
  - ML
author: Thinkwee
mathjax: true
html: true
---

*   Convolutional Sequence to Sequence
    
*   Robust Unsupervised Cross-Lingual Word Embedding Mapping
  
<!--more-->

{% language_switch %}

{% lang_content en %}

Convolutional Sequence to Sequence Learning
===========================================

*   Extremely straightforward, both the encoder and decoder use sequence-to-sequence learning with convolutional neural networks
*   Whether using a transformer or a CNN as an encoder, it is necessary to capture the semantic information of the entire sentence. Given the current situation where both are significantly ahead of RNNs, tree structures are more suitable as prior structures for natural language data.
*   transformer directly models the so-called self-attention in the first layer, personally, I believe this self-attention is modeling the parse relationships of sentences, modeling all parse pairs (a word to all other words in the sentence), all dimensions (it may be syntactic parse, entity relationship, coreference resolution, or dependency parse), and using the mechanism of attention to filter out unnecessary relationships, and then reorganizes them through a layer of fully connected layers. By iterating this parse + reorganization block multiple layers, abstracting features step by step, and adding structures commonly used in deep network design such as batch normalization and residual, the transformer is formed. Therefore, the transformer constructs the global parse relationships all at once and then gradually reorganizes, abstracts, and filters.
*   The structure of CNN conforms more to the conventional 套路 of syntactic parsing, still modeling across various dimensions, but it does not construct the global relationships all at once. Instead, it first analyzes local relationships (n-gram of kernel size) at the bottom level, and then summarizes and abstracts these local relationships through stacked layers.
*   Facebook uses a CNN block with a standard one-dimensional convolution in this paper, but employs a gated linear unit, i.e., multi-convolves to double the channel as the input for the gate structure, utilizing the gate structure to filter information and construct non-linear relationships, similar to the gating design of LSTM, and also achieving a similar effect of self-attention, while giving up the pooling design. On the decoder side, CNN is also used (I feel it's not very necessary), and the decoder still follows a process of generating from left to right in word order. To ensure this order relationship, the input to the decoder is masked, but I still haven't understood how the specific code implements it... Doing the masking and generating step by step in this way does not fully utilize the acceleration of CNN.
*   In this paper, attention is also introduced, which is the traditional encoder-decoder attention; the difference lies in
    *   Employed multi-layer attention; although the key remains the output of the encoder's last layer, attention is individually introduced to each layer of the decoder. However, the authors themselves also say that the decoder does not require many layers, two are sufficient, so this multi-layer attention may not be fully utilized. Moreover, multi-layers represent more context needed to decode each word, it seems that CNN as a decoder does not need much context, or has not fully utilized the longer context.
    *   The value of attention is not the same as the key; instead, it is the output of the last layer of the encoder plus the embedding of the encoder input. The authors believe that this approach can comprehensively consider both specific and abstract representations, and the actual effect is indeed better.
*   The author mentioned bytenet as a reference, but for some reason, did not adopt the dilation convolution design from bytenet.

A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings
==============================================================================================

*   Completely unsupervised cross-lingual word embedding mapping
*   Cross-lingual word embeddings, which refer to the use of the same word embedding matrix across multiple languages, allowing for cross-lingual model transfer of large-scale pre-trained word embeddings and/or language models
*   The general approach is to use word embedding matrices of two languages, map them to the same cross-lingual word embedding space, and establish word correspondence between the two languages
*   This type of research has been popular recently, and the most well-known downstream application it has spawned should be Facebook's unsupervised machine translation from 2018
*   Previous methods are divided into three categories:
    *   Supervised, using bilingual dictionaries, constructing thousands of supervised word pairs, treating learning mapping as a regression problem: modeling with the minimum mean square objective function, which subsequently gave rise to various methods: canonical correlation analysis; orthogonal methods; maximum margin methods. These methods can all be categorized as linear transformation mappings of word embedding matrices of the two languages into the same space.
    *   Semi-supervised, achieved through seed dictionary and bootstrap, such methods depend on good seeds and are prone to falling into local optima
    *   Another category is unsupervised generative methods, but the existing methods are too dependent on specific tasks, have poor generalization ability, and it is difficult to achieve good results for two different languages of the language system.
*   The text provided does not contain any source text to translate. Please provide the source text you wish to have translated into English.
*   ![APCWNT.png](https://s2.ax1x.com/2019/03/11/APCWNT.png)

Model
-----

*   Let $X$ and $Z$ be the word embedding matrices of two languages, the goal is to learn linear transformation matrices $W_x$ and $W_z$ such that the mapped matrices of the two languages are in the same cross-lingual space, forming a new cross-lingual word embedding matrix
*   The iterative update of the model depends on an alignment matrix $D$ , $D_{ij}=1$ , which is aligned when and only when the $i$ word of Language A corresponds to the $j$ word of Language B. The alignment relationship reflected by this matrix is unidirectional
*   Model is divided into four steps: preprocessing, fully unsupervised initialization, a robust self-learning process, and further improvement of results through symmetric weight reallocation

Preprocessing
-------------

*   Normalize the length of word embeddings
*   Perform mean removal for each dimension again
*   The first two preprocessing steps mentioned in the author's previous paper "Learning principled bilingual mappings of word embeddings while preserving monolingual invariance" aim to simplify the problem to seeking cosine similarity and maximum covariance. This paper discusses supervised methods and is to be read.
*   Perform another length normalization to ensure that each word embedding has a unit length, making the inner product of two word embeddings equivalent to the cosine distance

Completely unsupervised initialization
--------------------------------------

*   Initialization is difficult to perform because the word embedding matrices of the two languages are not aligned in two dimensions (each word, each dimension of the embedding)
*   The approach in this paper is to first construct two matrices $X^{'}$ and $Z^{'}$ , with the word embeddings in each dimension of these matrices aligned
*   $X^{'}$ and $Z^{'}$ are obtained by calculating the square root of the similarity matrix of the original word embedding matrix, i.e., $X^{'} = \sqrt sorted{XX^T}$ and $Z^{'} = \sqrt sorted{ZZ^T}$ . The product of a matrix with its transpose results in a similarity matrix under the same language (because preprocessing was done previously). Based on previous observations, two words representing the same meaning in two languages should have a similar single-language similarity distribution. Therefore, we sort each row of the similarity matrices of the two languages separately, from large to small. If two words have the same meaning, the corresponding rows in their sorted similarity matrices within their own language should have a similar distribution.
*   ![Ak0cYd.png](https://s2.ax1x.com/2019/03/13/Ak0cYd.png)
*   This skips the direct alignment of each dimension of word embeddings, converting it to alignment based on dictionary similarity. Afterward, only word alignment is needed, i.e., sorting each row of the similarity matrix individually and establishing the correspondence between words with similar row distributions.
*   Established word alignment, i.e., established the initial $D$ matrix

Robust self-learning process
----------------------------

*   Compute orthogonal mappings to maximize the similarity of the current $D$ matrix
    
    $$
    argmax_{W_x,W_z} \sum _i \sum _j D_{ij}((X_{i^*}W_X) \cdot (Z_{j^*}W_Z)) \\
    $$
    
    The optimal solution can be directly calculated as: $W_X=U,W_Z=V$ , where $U,V$ comes from $USV^T$ , which is the singular value decomposition of $X^TDZ$
    
*   After mapping the word embeddings of the two languages into a cross-lingual word embedding space (still two word embedding matrices, but within the same cross-lingual space), for each word in Language A, find its nearest word in Language B within the cross-lingual word embedding space, establish a mapping relationship, and update the $D$ matrix.
    
    $$
    D_{ij} = 1 \ \ \ if  \ \ j = argmax _k (X_{i^*}W_X) \cdot (Z_{j^*}W_Z) \\
    else \ \ D_{ij} = 0 \\
    $$
    
*   Repetitive iteration, $W_X,W_Z \rightarrow D \rightarrow W_X,W_Z \rightarrow D \rightarrow W_X,W_Z \rightarrow D \rightarrow W_X,W_Z$
    
*   Using a completely unsupervised initialization $D$ matrix results in better performance than random initialization, but it still falls into local optima. Therefore, the authors proposed several small tricks for the second step of the iteration, i.e., updating the $D$ matrix, to make the learning more robust
    
    *   Random dictionary induction: In each iteration, set elements of the D matrix model to 0 with a certain probability, forcing the model to explore more possibilities
    *   Based on word frequency dictionary truncation: Only update the top k most frequent words during each dictionary induction, to avoid noise from low-frequency words, with a truncation limit of 20,000
    *   CSLS retrieval: Previous methods find the nearest j for each i by mapping it to the cross-lingual word embedding space, updating $D_{ij}$ to 1. This nearest neighbor method is affected by the dimensionality disaster and does not perform well (the specific phenomenon caused is called hubs, where words cluster together, and hubs words are the nearest neighbors of many words with little difference). CSLS, which stands for cross-domain similarity local scaling, penalizes these hub words.
    *   Bidirectional dictionary induction, not only for finding j from i, but also for finding i from j
    *   These tricks differ in the initialization of the constructed matrices, do not perform random inductive, and the dictionary truncation limit is set to 4000

Further improving results through reweighting with symmetric weights
--------------------------------------------------------------------

*   After the iterative process is completed
    
    $$
    W_X = US^{\frac 12} \\
    W_Z = UV^{\frac 12} \\
    $$
    
*   Compared to previous papers, this method encourages the model to explore a wider search space by performing whitening and de-whitening before and after each iteration, and it is insensitive to direction
    
*   The reasons for reallocating weights were mentioned in the previous paper, to be read
    
{% endlang_content %}

{% lang_content zh %}

# Convolutional Sequence to Sequence Learning

- 非常直白，编码器和解码器都使用卷积神经网络的序列到序列学习
- 无论是transformer还是CNN作为encoder，都需要捕获整个句子的语义信息。就目前两者大幅领先RNN的现状开来，相比序列结构，树型结构更适合作为自然语言数据的先验结构。
- transformer直接在第一层建模所谓的自注意力，我个人觉得这个自注意力是在建模句子的parse关系，针对所有剖析对（一个词到本句所有其他词）、所有维度（可能是成分剖析、可能是实体关系、可能是共指消解、可能是依存剖析）进行建模，利用注意力的机制对无用的关系筛选，之后再过一层全连接层进行重组。这样剖析+重组的block迭代多层，逐步抽象特征，再加上batch normalization和residual这些深层网络设计常用的结构，就构成了transformer。因此transformer是一次性构建全局剖析关系，再逐步重组、抽象、筛选。
- CNN的结构则更符合常规句法剖析的套路，依然是针对了各个维度建模，但是不是一次性构建全局关系，而是在底层先剖析局部关系（kernal size大小的ngram)，然后通过叠加层对局部关系进行汇总、抽象。
- Facebook在本论文中采用的CNN block采用了普通的一维卷积，但是使用了gated linear unit，即多卷积出一倍的channel来作为门结构输入，利用门结构过滤信息、构建非线性关系，类似LSTM的门控设计，同时也起到了类似自注意力的效果，放弃了pooling的设计。而在decoder端，也使用了CNN（我感觉其实没有很大必要），decoder依然是一个从左往右逐字顺序生成的过程。为了保证这种顺序关系，对decoder的输入做了mask，而我现在还没弄懂具体代码是怎么实现的......这样做mask然后一步一步生成其实并没有充分利用CNN的加速。
- 在本文中也引入了注意力，是传统的encoder-decoder间注意力，不同的地方在于
  - 采用了多层注意力，虽然key依然是encoder最后一层的输出，但是对于decoder每一层都单独引入了注意力。然而作者自己也说decoder不需要太多层，两层足以，因此这个多层注意力可能也没充分利用。况且多层代表decode出每一个词所需要的上下文更多，看来CNN作为decoder并不需要太多上下文，或者说没有充分利用上比较长的上下文。
  - 注意力的value不是和key一样，而是encoder最后一层的输出加上encoder输入的embedding，作者认为这样可以综合考虑具体和抽象的表示，实际效果也确实要好一些。
- 作者提到了bytenet作为参考，但是不知道为啥并没有采用bytenet中的dilation convolution设计。

# A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings

- 完全无监督的跨语言词嵌入映射
- 跨语言词嵌入，即多种语言共用相同的词嵌入矩阵，这样可以将大规模预训练词嵌入和或者语言模型进行跨语言的模型迁移
- 一般的做法是利用两个语言的词嵌入矩阵，映射到同一个跨语言的词嵌入空间，并且建立两种语言的词对应关系
- 这类研究最近很火，其催生的最知名的一个下游应用应该就是Facebook18年的无监督机器翻译
- 之前的方法分三种：
  - 有监督的，使用双语词典，构建几千个监督词对，将学习映射看成回归问题：用最小均方目标函数建模，之后催生了各种方法：canonical correlation analysis；正交方法；最大间隔方法。这些方法都可以归为将两类语言的词嵌入矩阵做线性变化映射到同一个空间。
  - 半监督的，通过seed dictionary和bootstrap来做，这类方法依赖好的seed且容易陷入局部最优；
  - 另一类是无监督的生成式方法，但是已有的方法太过依赖特定任务，泛化能力不佳，针对语言系统不同的两种语言很难达到很好效果。
- 本文采用无监督的生成式方法，基于一个观察：在一种语言中，每个词都有一个在这种语言词典上的相似度分布，不同语言中等价的词应该具有相似的相似度分布。基于这种观察，本文建立了初始的seed dictionary，并采用一种更鲁棒的self learning方式来改进学习到的映射。
- ![APCWNT.png](https://s2.ax1x.com/2019/03/11/APCWNT.png)

## 模型

- 令$X$和$Z$分别为两种语言的词嵌入矩阵，目标是学习到线性变换矩阵$W_x$和$W_z$，使得映射后两种语言的矩阵在同一个跨语言空间,形成新的跨语言词嵌入矩阵
- 模型的迭代更新依赖一个对齐矩阵$D$，$D_{ij}=1$当且仅当A语言的第$i$个词对应着B语言的第$j$个词，该矩阵反映的对齐关系是单向的
- 模型分四步：预处理、完全无监督的初始化、一种鲁棒的自学习过程、通过对称权重重分配进一步改善结果

## 预处理

- 对词嵌入做长度归一化
- 再针对每一维做去均值
- 前两个预处理在作者之前的论文Learning principled bilingual mappings of word embeddings while preserving monolingual invariance中提到过，目的分别是简化问题为求余弦相似度和求最大协方差。这篇论文讲的是有监督方法，待阅读
- 再做一次长度归一化，保证每一个词嵌入都拥有单位长度，使得两个词嵌入的内积等价于cos距离

## 完全无监督的初始化

- 初始化很难做，因为两种语言的词嵌入矩阵，两个维度（每个词、嵌入的每一维）都不对齐
- 本文的做法是，先构造两个矩阵$X^{'}$和$Z^{'}$，这两个矩阵的词嵌入每一维是对齐的
- $X^{'}$和$Z^{'}$分别通过计算原词嵌入矩阵的相似矩阵开方得到，即$X^{'} = \sqrt sorted{XX^T}$，$Z^{'} = \sqrt sorted{ZZ^T}$。自己乘自己的转置即同一语言下的相似度矩阵（因为之前做了预处理）。根据之前的观察，表示同一词义的两种语言下的两个词，应该有相似的单语言相似度分布，那么我们将两种语言的相似度矩阵的每一行单独排序，从大到小排，则假如两个词有相同词义，他们在自己语言中的已排序相似度矩阵中的对应行，应该有相似的分布
- ![Ak0cYd.png](https://s2.ax1x.com/2019/03/13/Ak0cYd.png)
- 这样就跳过了词嵌入每个维度上的直接对齐，转换为词典相似度的对齐，之后只要做词的对齐，即对相似度矩阵每一行单独排序，建立行分布相似的词之间的对应关系即可。
- 建立了词语的对齐，即建立了初始化的$D$矩阵

## 鲁棒的自学习过程

- 计算正交映射以最大化当前$D$矩阵的相似度
  
  $$
  argmax_{W_x,W_z} \sum _i \sum _j D_{ij}((X_{i^*}W_X) \cdot (Z_{j^*}W_Z)) \\
  $$
  
  最优解可以直接计算得到：$W_X=U,W_Z=V$，其中$U,V$来自$USV^T$，是$X^TDZ$的奇异值分解
- 将两个语言的词嵌入映射到跨语言词嵌入空间（分别映射，依然是两个词嵌入矩阵，只不过在同一个跨语言空间内）后，对A语言的每一个词，在跨语言词嵌入空间内找其最近的B语言的词，建立映射关系，更新$D$矩阵。
  
  $$
  D_{ij} = 1 \ \ \ if  \ \ j = argmax _k (X_{i^*}W_X) \cdot (Z_{j^*}W_Z) \\
else \ \ D_{ij} = 0 \\
  $$
- 反复迭代，$W_X,W_Z \rightarrow D \rightarrow W_X,W_Z \rightarrow D \rightarrow W_X,W_Z \rightarrow D \rightarrow W_X,W_Z$
- 使用完全无监督的初始化$D$矩阵比随机初始化效果要好，但是依然会陷入局部最优，因此作者针对迭代的第二步，即更新$D$矩阵时提了几个小trick使得学习更为鲁棒
  - 随机词典归纳：每次迭代以一定概率将D矩阵模型元素设为0，迫使模型探索更多可能
  - 基于词频的词典截断：每次词典归纳时只更新前k个最频繁的词，避免低频词带来的噪音，截断上限为20000
  - CSLS检索：之前的方法是对每一个i,找一个映射到跨语言词嵌入空间后距离最相近的j，更新$D_{ij}$为1，这种最近邻方法受维度灾难影响，效果并不好（具体引起的现象叫hubs，即词语发生聚类，hubs词是许多词的最近邻，差异度不大）。CSLS即跨领域相似度局部放缩，惩罚了这些hubs词语
  - 双向词典归纳，不仅针对i找j，也针对j找i
  - 这些trick对初始化构建的矩阵有所差别，不做随机归纳，词典截断上限为4000

## 通过对称权重重分配进一步改善结果

- 即迭代完成之后计算
  
  $$
  W_X = US^{\frac 12} \\
W_Z = UV^{\frac 12} \\
  $$
- 比起之前的论文，在每一次迭代前后做白化和去白化的方法，这种方法鼓励模型探索更多搜索空间，且对方向不敏感
- 重新分配权重的原因在之前的论文中提到，待阅读

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