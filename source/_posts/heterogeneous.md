---
title: 异构信息网络的一些处理
date: 2019-10-30 18:42:00
categories: 自然语言处理
tags:
  - graph neural network
  - heterogeneous information network
  - machine learning
  - deep learning
  -	natural language processing
mathjax: true
html: true
---

记录近年来对于异构信息网络的一些处理

<!--more-->

# PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks
-	较早的一篇论文（作者都是大神），定义清楚了meta path中的很多概念，提出了衡量异构信息网络中节点相似度的一种方法。
-	传统的相似度衡量方法存在偏差，基于路径数统计和随机游走的方法存在偏差，偏向度数较多的节点；pair-wise的随机游走偏向具有较多离群点邻居的节点
-	PathSim的想法是，两个相似的节点不仅仅应该相互强链接，还需要share comparable visibility
-	在给定对称的meta path $P$下，两个同类型节点$x,y$的PathSim定义为：
	$$
	s(x, y)=\frac{2 \times\left|\left\{p_{x \leadsto y}: p_{x \sim y} \in \mathcal{P}\right\}\right|}{\left|\left\{p_{x \leadsto x}: p_{x \hookrightarrow x} \in \mathcal{P}\right\}\right|+\left|\left\{p_{y \leadsto y}: p_{y \leadsto y} \in \mathcal{P}\right\}\right|}
	$$
-	实际分母的节点回归概率就是visibility，作者在传统的pathcount上除以visibility，所有的路径都由edge weight累乘得到。
-	类似于对节点的度做了对称归一化。 

# Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification
-	任务：节点分类
-	异构类型：节点异构，包含三类节点，文本、实体、主题
-	解决办法：
	-	最朴素：扩充节点的特征空间，将三类节点的特征向量拼接起来，对于具体的某一节点，其不包含的特征向量位置全设为0
	-	异构图卷积：将相同节点类型的子图分离，每个子图单独做卷积，不同的子图通过参数变换矩阵投影到相同隐空间并相加激活作为下一层，具体而言，原始GCN为：
	$$
	H^{(l+1)}=\sigma\left(\tilde{A} \cdot H^{(l)} \cdot W^{(l)}\right)
	$$
	而异构GCN为：
	$$
	H^{(l+1)}=\sigma\left(\sum_{\tau \in \mathcal{T}} \tilde{A}_{\tau} \cdot H_{\tau}^{(l)} \cdot W_{\tau}^{(l)}\right)
	$$
	其中$\tilde{A}_{\tau}$的行是所有节点，列是某一类型的所有节点，这样就抽离出了同构的连接子图，即对于每个节点，我们分别考虑他的邻域里类型a的节点，做信息聚合得到编码a，再考虑邻域里类型b的节点，做信息聚合得到编码b，编码a和b通过各自的变换矩阵变换到同一隐空间再相加。这样的设计是符合逻辑的。
	-	作者还考虑了以下情况：对于某一节点，不同类型的邻域节点的贡献不一样，同一类型下不同的邻域节点贡献也不一样。显然这里需要注意力。作者就提出了对偶注意力(即双层注意力)，一层是type level的，一层是node level的，先用某一类型邻域节点embedding的均值作为type embedding，然后根据当前节点embedding与type embedding 计算出type attention weight，同理用具体的邻域节点embedding和当前节点embedding再加上type attention得到node attention，利用计算出的node attention替换GCN里的对称归一化邻接矩阵。


# Heterogeneous Graph Neural Network
-	任务：图表示学习
-	异构类型：节点异构、边异构、节点多属性
-	解决办法：
	-	四步走：异构邻居采样、多属性编码、同类型邻居聚合、不同类型聚合
	-	异构邻居采样：基于重启的随机游走，先随机游走，且有一定概率p返回初始节点（重启），直到采样了一定数量的邻域节点。每种类型的邻域节点有上限值以确保所有类型的邻居都能采样到。再等比例缩小，对每个邻域节点类型$t$，只取$K_t$个邻域节点，分好组。
	-	多属性编码：根据多模态内容，预先编好码，例如文本用paragraph2vec，图像用CNN，同一邻域节点的不同属性信息用BiLSTM编码
	-	同类型邻居聚合：用BiLSTM聚合同类型下多个邻域节点的特征
	-	不同类型聚合：再用一个注意力机制聚合不同类型的特征

# Heterogeneous Graph Attention Network
-	任务：图表示学习
-	异构类型：节点异构
-	解决办法：
	-	实现node-level和metapath-level的双层注意力。
	-	需要在所有的metapath neighbour里加上自身，类似于GCN里的内环。
	-	node-level注意力，对一条metapath上的不同节点进行注意力加权。因为不同类型的节点特征表示空间不同，因此针对每一种类型对应一个特征转换矩阵，将不同类型节点映射到同一空间，之后通过自注意力机制对节点进行注意力的计算和加权(其中$\phi$代表metapath)：
	$$
	h_i^{\prime} = M_{\phi _i} \cdot h_i \\
	e^{\phi}_{ij} = attn_{node}(h^{\prime}_i,h^{\prime}_j;\phi) \\
	$$
	在计算attention需要做mask，只对邻域节点计算attention并做softmax。值得注意的是这里的自注意力的非对称性对异构图来说很重要，因为在一个节点对里，两个节点的邻域不同，相互的影响不是等量的。简单来说，对某一个节点，计算其在某一类metapath下所有邻域节点的注意力权重,输入是两个节点的h以及metapath specific的一个参数向量，输出注意力权重，然后对某一节点，加权求和其某一类metapath下所有邻域节点的h，得到该节点的某一metapath的表示。
	$$
	\alpha_{i j}^{\Phi}=\operatorname{softmax}_{j}\left(e_{i j}^{\Phi}\right)=\frac{\exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{j}^{\prime}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}^{\mathrm{\Phi}}} \exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{k}^{\prime}\right]\right)\right)} \\
	$$
	计算出注意力之后对变换后的metapath邻域节点特征加权：
	$$
	\mathbf{z}_{i}^{\Phi}=\sigma\left(\sum_{j \in \mathcal{N}_{i}^{\Phi}} \alpha_{i j}^{\Phi} \cdot \mathbf{h}_{j}^{\prime}\right)
	$$
	-	在实际加权的时候作者参考了multi-head的做法，计算了k个attention加权特征并拼接起来
	-	metapath-level的attention即对某一节点所有不同类的metapath embedding进行加权。先将每一条metapath的embedding变换到同一隐空间，然后参数化计算出注意力权重并softmax。需要注意的是softmax之前的attention logit是在所有节点上计算某一类型metapath的平均，分母是节点数，分子是包含该类型metapath的节点的metapath embedding累加，而之前的node-level是针对某一节点某一metapath下所有的邻域节点平均：
	$$
	w_{\Phi_{i}}=\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{q}^{\mathrm{T}} \cdot \tanh \left(\mathbf{W} \cdot \mathbf{z}_{i}^{\Phi}+\mathbf{b}\right)
	$$
	-	之后再在所有metapath类型上做softmax，得到权重，加权每个节点不同metapath embedding得到最终embedding
	-	整个双层attention的logit以及softmax的范围有点绕，时而局部时而全局，需要仔细考虑清楚。
-	可以看到整个过程是可以在节点层次并行化计算的
-	从结果来看效果达到了SOTA，而且可视化的结果可以看到节点embedding的聚类效果更好，attention也带来了一定可解释性。

# Representation Learning for Attributed Multiplex Heterogeneous Network
-	任务：图表示学习
-	异构类型：节点异构、边异构、节点多属性
-	解决办法：
	-	考虑某一节点在不同类型边连接下有不同的embedding，将节点总的overall embedding拆成与边无关的base embedding和与边相关的edge embedding
	-	edge embedding与边类型相关，通过相同类型的边相连的邻域节点aggregate得到，这里的aggreagator function可以采用GraphSage里的做法。
	-	经过k层聚合之后，对于每个节点都得到了k种边类型的edge embedding，通过self attention将这些edge embedding加权求和，乘上比例再加上base embedding就得到了最终的overall embedding
	-	以上是直推式(transductive)模型，对于未观测数据，需要归纳式(inductive)的方法。具体做法很简单，将base embedding和edge embedding参数化为节点attribute的函数，而不是随机初始化之后完全根据已有的图学习。这样即便有图中没看见的节点，只要节点有属性，一样可以进行overall embedding的提取
	-	最后做基于meta-path的random walk得到训练对，使用skip gram训练，加入了负采样。

# Semi-supervised Learning over Heterogeneous Information Networks by Ensemble of Meta-graph Guided Random Walks
-	任务：节点分类
-	异构类型：节点异构，包含三类节点，文本、实体、主题
-	解决办法：meta-path guided random walk

# Joint Embedding of Meta-Path and Meta-Graph for Heterogeneous Information Networks
