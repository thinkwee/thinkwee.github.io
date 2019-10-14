---
title: 基于图的摘要论文选读
date: 2019-10-03 17:18:15
categories: 自然语言处理
tags:
  - graph neural network
  - deep learning
  - summarization
  -	natural language processing
mathjax: true
html: true
---

基于图的自动摘要相关论文选读
<!--more-->


# Toward Abstractive Summarization Using Semantic Representations
-	探讨了如何从原文的AMR图构建摘要的AMR图，即graph summarization
-	三步走，source graph construction, subgraph prediction, 	text generation

## Source Graph Construction
-	这一步是将多句graph合并为source graph，一些concept node需要合并
-	首先AMR不会重复建模提到的concept，而是将提到的频次作为特征补充进node embedding当中
-	节点合并包含两步：把一些节点的子树合并，接着把相同的概念节点合并
-	子树合并之后的节点名称里包含了所有子树节点的信息，之后只有完全相同的节点才能合并，因此经过一次子树合并之后的节点很难再次合并（很难完全相同），这里需要做一些共指消解的工作(future work)
-	一些节点之间会有多条边，取出现次数最多的两个边的label，合并，抛弃其他边
-	相同的概念节点直接合并
-	加一个总的root节点连接各个句子的root节点
-	这样连接出来的source graph对于gold summary graph的边覆盖度不高，因此对于source graph还后处理一下，将所有节点之间加入null 边，提高覆盖率

## Subgraph Prediction
-	子图预测问题是一个structured prediction problem
-	作者构建子图打分函数为模型参数线性加权边和节点的特征：
$$
\operatorname{score}\left(V^{\prime}, E^{\prime} ; \boldsymbol{\theta}, \boldsymbol{\psi}\right)=\sum_{v \in V^{\prime}} \boldsymbol{\theta}^{\top} \mathbf{f}(v)+\sum_{e \in E^{\prime}} \boldsymbol{\psi}^{\top} \mathbf{g}(e)
$$
-	decoding：基于ILP选出得分最大的子图。这里的约束条件是选出的子图必须合法且是联通的，可以通过指示函数v,e和流量f来描述。$v_i=1$即第i个节点被选中，$e_{i,j}=1$即i,j两个节点之间的边被选中，$f_{i,j}$代表从i流向j的流量，那么合法即：
$$
v_{i}-e_{i, j} \geq 0, \quad v_{j}-e_{i, j} \geq 0, \quad \forall i, j \leq N
$$
-	联通即：从根流出的流量到达选中的每一个概念节点，每一个概念节点消耗一个流量，只有边被选中时流量才可能通过，这三个约束用数学描述为：
$$
\begin{array}{r}{\sum_{i} f_{0, i}-\sum_{i} v_{i}=0} \\ {\sum_{i} f_{i, j}-\sum_{k} f_{j, k}-v_{j}=0, \quad \forall j \leq N} \\ {N \cdot e_{i, j}-f_{i, j} \geq 0, \quad \forall i, j \leq N}\end{array}
$$
-	另外作者只假设了每一个概念只有一个父节点，即构建为树的形式
$$
\sum _j e_{i,j} \leq 1, \quad \forall i, j \leq N
$$
-	这种形式的ILP在sentence compression和dependency parsing中都出现过，作者使用gurobi的ILP算法完成最优化
-	可以附加一个约束来限制摘要的长度，例如选中的边总数不大于L
-	以上是decoding，即选子图，但选图基于分数，而分数由参数加权，因此还包含了一个参数的优化。我们需要一个损失函数来衡量decoded graph和gold summary graph之间的差距，然而gold summary graph可能不在source graph当中，作者借鉴了机器翻译中的ramp loss，作者对比了感知机所用的perceptron loss, structured SVM中的hinge loss以及ramp loss，其中$G$是source graph，$G^{*}$ 是gold summary graph：
$$
\begin{array}{ll}{\text {perceptron loss: }} & {-\text { score }\left(G^{*}\right)+\max _{G} \text { score }(G)} \\ {\text {hinge loss: }} & {-\text { score(G^{*} ) }+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)} \\ {\text {ramp loss: }} & {-\max _{G}\left(\text {score}(G)-\operatorname{cost}\left(G ; G^{*}\right)\right)+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)}\end{array}
$$
-	cost对多余的边惩罚
-	perceptron loss很简单，就是希望缩小gold graph与decoded graph之间的分数差距
-	hinge loss在ILP中加入对多余边的惩罚，使得decoded graph的分数尽可能大，而不仅仅是和gold graph接近，这里decoded的graph会比直接计算分数得到的graph分值上差一点
-	ramp loss相比hinge loss就是在前面一项加了一个反向的惩罚，实际ramp loss依然是在缩小两个图的分数差距，只不过一个图比best decoded graph分值高一点，另一个比best decoded graph低一点，放宽松了条件

## Generation
-	作者目前只统计了decoded graph中概念节点对应的text span，并没有生成可读的摘要，因此只计算了ROUGE-1

# Abstract Meaning Representation for Multi-Document Summarization
-	这是上一篇的扩展
-	用AMR构建有根有向无环图，节点是概念，边是语义关系:
	-	节点：可能是PropBank里的一个frameset（命题），一个普通英语单词，一个特殊类别词，一个字符串，
	-	边：可以是PropBank里的命题关系，或者魔改之后的关系
-	整个系统三个部分
	-	source sentence selection：输入一系列文章，然后挑出关于某一主题不同方面的句子
	-	content planning：输入一系列句子，输出摘要图
	-	surface realization：将图转换为可读的摘要句
-	三个组件可分别用领域内小语料优化

## Source Sentence Selection
-	因为是多文档摘要，因此对每一个输入样例（多篇文档），做谱聚类，每个簇再挑若干句子
-	这样就有多个句子组，之后和更改了输入的摘要模型一样，需要重新构造训练对，这里是要构造接下来提供给content planning的训练对，即句子组和对应的gold summary的AMR graph。就对gold summary里的每一句，和句子组算一个平均相似度，选相似度大的作为训练对里的句子组。平均相似度有：
	-	LCS
	-	VSM
	-	Smatch方法，参考了论文Smatch: an evaluation metric for semantic feature structures
	-	Concept Coverage，即最大覆盖gold summary AMR graph里的concept
-	四种相似度也做了ablation

## Content Planning
-	训练对是句子组和summary的AMR graph，自然这个部分就是学习这个转换过程
-	首先要把句子组里的summary转成AMR graph，作者试用了两种AMR Parser，JAMR和CAMR
-	之后把句子组里的每一句也转成AMR graph，并且做合并（这一部分论文描述并不清楚）
	-	相同概念节点合并？
	-	做共指消解，把相同指代概念节点合并
	-	一些特殊节点需要把子树整合进节点信息里，叫mega-node，其实就是取消不必要的展开，将展开的具体信息直接写进节点里，例如date entity :year 2002 :month 1 :day 5。这些mega-node只有完全相同时才能合并
	-	最后生成一个root节点，把各个子图的root节点连起来
	-	通过最后一个操作貌似相同概念节点合并是同一子图内相同节点合并？
-	接下来设计算法，从源AMR graph中识别出摘要的AMR graph，包含两部分
	-	graph decoding：通过整数线性规划(ILP)识别出一个最优摘要图：首先构造一个参数化的图打分函数，将每一个节点特征和边特征通过参数加权并累加得到分数，这里的特征是手工构造，参考Fei Liu他的一系列AMR summarization的论文；接下来做一个ILP，要求找一个子图，使得得分最大，限制为L个节点而且子图是连接的。
	-	parameter update：最小化系统解码出的摘要图和gold summary图之间的差距。这一步优化的是上一步打分函数中的特征加权参数。构造损失函数来衡量decoded graph和gold graph之间的差距。有时gold graph不能从source graph中解码出来，这时就采用structed ramp loss，不仅仅考虑score，还考虑cost，即gold graph和decoded graph就是否将某个节点或者边加入摘要达成一致的程度
	$$
	L_{ramp}(\theta, \phi) = max_G (score(G)+cost(G;G_{gold})) - max_G(score(G) - cost(G;G_{gold}))
	$$

## Surface Realization
-	将图转成句子
-	AMR图并不好转成句子，因为图并不包含语法信息，一个图可能生成多句不合法的句子，作者两步走，先将AMR图转成PENMAN形式，然后用现有的AMR-to-text来将PENMAN转成句子

# Abstractive Document Summarization with a Graph-Based Attentional Neural Model
-	万老师团队的一篇论文，想法非常的好，重要的部分在两点：
	-	hierarchical encoder and decoder：由于需要在句子级别上做编解码以适应图打分的操作，所以采用了分层的seq2seq，无论编码解码都是word-level加sentence-level
	-	graph-attention：这里用的图是其实是pagerank里的全连接图，相似度直接用enc-dec的隐层向量内积来衡量，然后利用topic-aware pagerank来重新计算句子级别注意力权重。
-	在编解码阶段，我们利用隐层来计算相似度，这和原始的attention是一样的，只不过原始的attention加了一个参数矩阵（现代的attention连参数矩阵都懒得加了）使得这个相似度能够体现出注意力权重（分数），那么graph-attention就是在这个相似度上直接计算pagerank的markov链迭代，认为马氏链的稳定分布$f$就是重新rank之后的句子分数，这里有一点论文里没讲，作者做了一个假设，即编解码时拿到的已经是稳定状态，而不是从头迭代，因此可以令$f(t+1)=f(t)=f$，直接算出稳定分布：
$$
\mathbf{f}(t+1)=\lambda W D^{-1} \mathbf{f}(t)+(1-\lambda) \mathbf{y} \\
\mathbf{f}=(1-\lambda)\left(I-\lambda W D^{-1}\right)^{-1} \mathbf{y} \\
$$
-	基本形式与pagerank一致，一部分是基于相似矩阵的salience分配，另一部分补上一个均匀分布$y$保证马氏链收敛(这里感觉应该是简略了了，把均匀转移矩阵乘以f直接写成了均匀分布)，值得注意的是这是在sentence-level的编解码隐层状态做的计算，因此是计算给定某解码句下，各个编码句的graph attention score，如何体现这个给定某解码句？那就是用topic-aware pagerank，将解码句看成topic，把这个topic句加入pagerank的图里，并且y从均匀分布改成one-hot分布，即保证了解码句在graph中的影响力，并借此影响其他句子。
-	之后借鉴了distraction attention使得注意力不重复：
$$
\alpha_{i}^{j}=\frac{\max \left(f_{i}^{j}-f_{i}^{j-1}, 0\right)}{\sum_{l}\left(\max \left(f_{l}^{j}-f_{l}^{j-1}, 0\right)\right)}
$$
-	在解码端也做了一些小技巧，包括：
	-	OOV的处理，用@entity+单词长度来作为标签替换所有容易成为OOV的实体，并尝试把解码句中生成的实体标签还原，根据单词长度在原文中查找
	-	hierarchical beam search：word-level的beam search打分考虑了attend to的原文句子和当前生成部分的bigram overlap，希望这个overlap越大越好；sentence-level的beam search则希望生成每一句时attend to的原文句子不相同，这一段描述不是很清楚，应该是生成每一句时会attend N个不同的原文句产生N个不同的decoded sentence
-	本文的层次编解码其实起到了很关键的作用，作者并没有一股脑用单词级别的注意力，还是根据句子关系构件图并重排序，在beam search也充分利用了两个层次的信息
-	从ablation来看，graph attention和sentence beam的效果其实不大，影响ROUGE分数最大的是考虑了bigram overlap的word-level beam search，这也暴露了ROUGE的问题，即我们之前工作中提到的OTR问题

# Towards a Neural Network Approach to Abstractive Multi-Document Summarization
-	这篇论文是上篇论文的扩展，从单文档摘要扩展到多文档摘要，主要是如何将大规模单文档摘要数据集上预训练好的模型迁移到多文档摘要任务上
-	相比单文档模型，编码端又加了一层文档级别的编码，文档之间并没有依存或者顺序关系，因此没必要用RNN，作者直接用了线性加权,值得注意的是这个加权的权重不应该是固定或者直接学习出来的，而应该根据文档本身决定，因此作者给权重加了一个依赖关系学习出来，依赖文档本身和文档集的关系：
$$
w_{m}=\frac{\mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}{\sum_{m} \mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}
$$
-	注意力的机制基本不变，decoder的初始状态从单文档变成多文档编码，注意力加权从单篇文档句子数量到多篇文档句子数量。这里带来的一个问题是多文档的句子数量太大了，很多注意力被分散的很均匀，加权之后包含的信息量太大。因此作者将global soft attention给截断了一下，只有top k个句子可以用权重加权，其余的句子直接在编码中被抛弃
-	单文档到多文档的迁移其实并不是论文的重点，作者在CNN/DM上训练单文档的模型部分，之后在少量DUC数据集上训练多文档的部分，但是这两个数据集挺一致的，很多工作在CNNDM上训练在DUC上测试也能取得不错的效果。
-	论文的ablation做的非常详细，对比了多种功能图模型方法下的效果，包括Textrank,Lexrank,Centroid
-	值得注意的是作者使用了编辑距离来衡量文摘的抽象程度

# Topical Coherence for Graph-based Extractive Summarization
-	基于主题建模构建图，使用ILP做抽取式摘要
-	作者使用了二分图，一边是句子节点，一边是主题节点，两组节点之间用边连接，边的权值是句子中所有单词在某一主题下概率的对数和，除以句子长度做归一化
-	使用HITS算法在二分图上计算句子的重要程度

# Graph-Based Keyword Extraction for Single-Document Summarization

# Multi-Document Abstractive Summarization Using ILP Based Multi-Sentence Compression

# Integrating Importance, Non-Redundancy and Coherence in Graph-Based Extractive Summarization

# Graph-based Neural Multi-Document Summarization
-	用GCN做抽取式摘要，在这里GCN起到了一个特征补充的作用，原始的做法就是一个two-level GRU，documents cluster做一个embedding，其中每一个sentence有一个embedding，然后类似IR，拿sentence embedding和documents embedding做一个比较算出salience score，之后再用一个贪心的方法根据分数抽句子，大框架依然是打分-抽取的思路
-	GCN加进了两层GRU之间，即句子的embedding在一个句子关系图下做了三层GCN，之后再由documents层次的GRU生成documents embedding
-	这里就关注两点：句子关系图如何构建
-	句子关系图作者试了三种：
	-	最naive的，tfidf的cosine距离
	-	Towards Coherent Multi-Document Summarization一文中的ADG
	-	作者在ADG上改进的PDG
-	之后直接套GCN传播就行了
