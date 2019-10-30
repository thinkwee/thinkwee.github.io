---
title: NLP中对异构网络的一些处理
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


# Semi-supervised Learning over Heterogeneous Information Networks by Ensemble of Meta-graph Guided Random Walks
-	任务：节点分类
-	异构类型：节点异构，包含三类节点，文本、实体、主题
-	解决办法：meta-path guided random walk

# Heterogeneous Graph Attention Network

# Heterogeneous Graph Neural Network

# Representation Learning for Attributed Multiplex Heterogeneous Network

# Joint Embedding of Meta-Path and Meta-Graph for Heterogeneous Information Networks