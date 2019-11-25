---
title: 知识图谱应用论文选读
date: 2019-11-13 09:36:19
categories: 自然语言处理
tags:
  - acl
  - machine learning
  - deep learning
  -	natural language processing
  - knowledge graph
mathjax: true
html: true
---

临时抱佛脚，一些关于Graph（大部分是KG）的Embedding、NLG、Matching方向论文选读。

<!--more-->

# Cross-lingual Knowledge Graph Alignment via Graph Matching Neural Network

-	研究：跨语言知识图谱中实体的对齐

-	一般的做法，在各自语言的知识图谱中投影到低维向量，然后学习到一个相似度计算函数

-	问题在于，上面的做法依赖于相同实体在不同语言知识图谱下的邻域结构相同这一假设，但这一假设并不总是成立，因此传统的做法对少对齐邻域节点/少邻域节点的实体不友好

-	作者提出主题实体图来编码实体节点的上下文信息，将节点embedding之间的Match转换为主题实体图之间的图Match

-	这里的主题实体指的是需要对齐的实体，主题实体图即主题实体周围一跳邻域实体与本体一同构建的子图，假如在原知识图谱里这两个实体没有直接相连的话就在知识图谱里添加一条边

-	得到主题实体图之后，经过四层网络计算出两个主题实体图之间的相似度：
	
	-	输入表示层：对主题实体图中的每一个实体学习到embedding。首先用word-level lstm学习到初始embedding，然后因为是有向图，因此需要将邻域节点区分为输入和输出邻居，分别做聚合（FFN+mean pooling），拼接到上一次得到的实体embedding并更新（FFN），迭代K次
	
	-	节点级别的局部匹配层：两个图互相匹配所有实体节点，这里用了基于注意力的Matching，即先用cosine距离得到主题实体图1里某个节点i和主题实体图2所有节点之间的相似度，用这个相似度作为注意力权重去加权主题实体图2所有节点得到图2的graph embedding，然后用这个graph embedding和图1的query entity embedding之间计算一个多角度cosine距离作为local match score，这个多角度是指l个角度用l个加权向量(d维，同embedding维度）表示。一个角度下的cosine距离就用一个加权向量（逐元素相乘）加权两个embedding再计算cosine距离，l个角度合在一起就是一个$W \in R^{l*d}$的矩阵，如下：
	$$
	score_{perspective_k} = cosine(W_k \cdot embedding_1, W_k \cdot embedding_2)
	$$
	
	-	全局匹配层：局部匹配层存在着之前提到的对少共现邻居节点不友好的问题，这里就需要全局匹配。具体做法是，再用一个GCN将局部的match embedding（上一层多角度cosine score得到的向量）传递到整个主题实体图，之后在整个图上做一个FFN + max/mean pooling，得到两个图的graph matching vector
	
	-	预测层:将两个主题实体图的graph matching vector拼接送入一个softmax预测

-	训练时，对每一个正样本启发式的生成了20个负样本，两个方向各十个，这里直接基于word-level average embedding作为实体的特征向量，匹配十个最相似的实体

# Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling

-	作者希望在语言模型当中维持一个local knowledge graph来管理已经探测到的事实，并利用该图谱来查询未知的事实用于文本生成，称之为KGLM(Knowledge Graph Language Model)

-	假设实体集合为$\xi$，则KGLM要预测的是
	$$
	p(x_t,\xi _t|x_{1,t-1},\xi_ {1,t-1})
	$$

-	生成下一个词的流程可以拆分如下：
	
	-	下一个词不是实体：那就正常词典范围上计算概率
	
	-	下一个词是全新实体：在正常词典以及所有实体范围上计算概率
	
	-	下一个词是与已经看见的实体相关的实体：先挑出一个已经看见的实体作为父节点，再挑出一个子节点，之后在正常词典以及该子节点的所有别名上计算概率

-	作者使用LSTM作为LM的基础模型，所有的挑选：挑新实体、挑父节点、挑子节点，均利用LSTM的隐状态（切分为三部分），并加上实体和关系的预训练embedding作为输入依赖，之后通过softmax计算概率

-	为了实现这样的模型数据集应该提供实体信息，作者提出了Linked WikiText2数据集，该数据集的构建流程如下：
	
	-	根据维基百科上的链接创造实体之间的链接，借助neural-el来识别wikidata数据库中额外的链接，使用stanford corenlp来做共指消解。
	
	-	构造local knowledge graph：接下来需要建立实体之间的parents relation，每遇到一个实体a，将其在wikidata中相关联的所有实体{b}加入matching的候选，加入相关联的某一实体b在之后的文段中出现了，则将实体a作为实体b的父节点
	
	-	以上的做法只是构建了初始集合，需要不断扩展，作者还对日期、量词等做了alias table
	
	-	以下是一句话在Linked WikiText2中的表示，相比WikiText使用api query构造的方法，Linked WikiText2直接对原始html操作，保留了更多链接信息：
	![MY7je0.jpg](https://s2.ax1x.com/2019/11/14/MY7je0.jpg)

-	Train and Inference：首先对实体和关系使用TransE算法做一个Pretraining，给定三元组(p,r,e)，目标是最小化距离：
	$$
	\delta\left(\mathbf{v}_{p}, \mathbf{v}_{r}, \mathbf{v}_{e}\right)=\left\|\mathbf{v}_{p}+\mathbf{v}_{r}-\mathbf{v}_{e}\right\|^{2}
	$$
	采用Hinge Loss，使得正负样本得分之差不超过$\gamma$:
	$$
	\mathcal{L}=\max \left(0, \gamma+\delta\left(\mathbf{v}_{p}, \mathbf{v}_{r}, \mathbf{v}_{e}\right)-\delta\left(\mathbf{v}_{p}^{\prime}, \mathbf{v}_{r}, \mathbf{v}_{e}^{\prime}\right)\right)
	$$

-	虽然整个过程是生成式的，但是所有变量均可见，因此可以端到端的直接训练，对于有多个父节点的实体节点，需要对概率做归一化

-	在推理过程中，我们没有标注信息，我们希望计算的也是单词$x$在实体$\xi$求和得到的边缘概率，而不是联合概率（我们只希望得到词，词的实体信息被marginalize了），然而实体太多，不可能对所有实体计算联合概率再求和，因此作者采用了重要性采样:
	$$
	\begin{aligned} p(\mathbf{x}) &=\sum_{\mathcal{E}} p(\mathbf{x}, \mathcal{E})=\sum_{\mathcal{E}} \frac{p(\mathbf{x}, \mathcal{E})}{q(\mathcal{E} | \mathbf{x})} q(\mathcal{E} | \mathbf{x}) \\ & \approx \frac{1}{N} \sum_{\mathcal{E} \sim q} \frac{p(\mathbf{x}, \mathcal{E})}{q(\mathcal{E} | \mathbf{x})} \end{aligned}
	$$
	其中proposed distribution使用判别式的KGLM得到，即另外训练一个KGLM判断当前token的annotation

-	结果非常漂亮，KGLM仅仅用了LSTM，参数量也不大，和超大规模的GPT-2模型相比，在实体词的预测上有着明显优势。

# DyKgChat: Benchmarking Dialogue Generation Grounding on Dynamic Knowledge Graphs

-	本文作者提出了一个新的任务，动态知识图谱对话生成，也就是希望抓住图谱中的关系，来将基于知识图谱的对话生成推广到zero-shot

-	任务的详细描述分为两步：
	
	-	每轮对话t，给定输入x和图谱K，希望生成正确的回答y，而且包含正确的知识图谱实体
	
	-	当知识图谱更新之后（这里只可能更新关系和受体），回答y能够相应更改回答。

-	为了有效衡量动态知识图谱对话的质量，作者提出了两类指标：
	
	-	知识实体建模：包括已知要预测实体，实体词命中的准确率；判别要预测实体还是通用词的TP\FN\FP;整个知识图谱所有实体的TP\FN\FP
	
	-	图自适应：作者提出了三种改变图的方法，包括shuffle和随即替换实体，观察生成的序列是否替换且替换正确

-	作者提出了一个平行语料库，包含中英两个电视剧的语料，并做了详细的处理

-	作者提出的模型Qadpt在seq2seq的基础上修改，首先将decoder的当前状态$d_t$生成一个controller$c_t$来决定是从KG里挑实体还是从generic vocab里生成一般词汇。和copy mechanism一样这个选择不是hard，而是分别计算概率，最后将两部分词表拼到一起，最后依概率选择：
	$$
	\begin{aligned} P\left(\{K B, \mathcal{W}\} | y_{1} y_{2} \ldots y_{t-1}, \mathbf{e}(x)\right) \\=\operatorname{softmax}\left(\phi\left(\mathbf{d}_{t}\right)\right) \\ \mathbf{w}_{t}=P\left(\mathcal{W} | y_{1} y_{2} \ldots y_{t-1}, \mathbf{e}(x)\right) \\ c_{t}=P\left(K B | y_{1} y_{2} \ldots y_{t-1}, \mathbf{e}(x)\right) \\ \mathbf{o}_{t}=\left\{c_{t} \mathbf{k}_{t} ; \mathbf{w}_{t}\right\} \end{aligned}
	$$

-	至于如何产生实体候选列表，就是在知识图谱上做reasoning，不同于一般的attention based graph embedding的做法，作者采用了multi-hop reasoning
	
	-		首先将path matrix和adjacency matrix合成transition matrix，其中的path matrix是指用$d_t$学习到的每个实体选择每一种关系的概率，之后依概率选择受体节点：
	$$
	\begin{aligned} \mathbf{R}_{t} &=\operatorname{softmax}\left(\theta\left(\mathbf{d}_{t}\right)\right) \\ \mathbf{A}_{i, j, \gamma} &=\left\{\begin{array}{ll}{1,} & {\left(h_{i}, r_{j}, t_{\gamma}\right) \in \mathcal{K}} \\ {0,} & {\left(h_{i}, r_{j}, t_{\gamma}\right) \notin \mathcal{K}}\end{array}\right.\\ \mathbf{T}_{t}=\mathbf{R}_{t} \mathbf{A} \end{aligned}
	$$
	
	-	之后取一个初始向量$s$（均匀分布？），用transition matrix做n次transform，得到每个实体出现的概率并提供给controller计算，这里会使用one hot ground truth计算一个交叉熵作为辅助损失

# Graph2Seq: Graph to Sequence Learning with Attention-based Neural Networks

-	顾名思义，输入为图结构组织的数据，生成的是序列

-	以往的做法，将图编码成固定长度的序列，再用Seq2Seq，作者认为这样存在信息丢失，本身Enc Seq 2 Dec Seq就存在信息丢失，现在Graph 2 Enc Seq会再丢失一层信息

-	因此比较自然的做法应该是，解码器在编码的图节点上做attention，直接利用图的信息

-	首先是图编码器，参考了GraphSage的做法，值得注意的是作者处理的是有向图，因此将邻居节点按两个方向做了区分，分别做Aggregate和Update的操作，做了k跳之后再拼接回来

-	作者试了Mean、LSTM、Pooling三种，由于邻居是无序的，因此LSTM没有时序上的效果，作者直接随机排列邻居用LSTM Aggregate

-	作者认为传给解码器的不只node embedding还需要graph embedding。作者采用了两种方法获取graph embedding
	
	-	Pooling-based：先将所有的node embedding经过一个全连接层，然后逐元素做max、min、average pooling，作者发现三种方法的实际效果相差不大，就使用max pooling作为默认的池化方法
	
	-	Node-based：在图中加入一个超级节点，该节点与图中其他所有节点相连，用该节点经过图编码之后的embedding作为Graph embedding

-	基于注意力的解码器：graph embedding作为解码器的初始状态输入，之后decoder每一步生成在所有node embedding上的attention并加权作为该时间步decoder的隐状态

-	重点关注NLG task，作者测试了SQL2Text任务，首先将SQL Query建图，然后使用Graph2Seq。效果显著好于SQL Query到Text的Seq2seq

-	另外在Aggregate的比对实验中发现，Mean Pooling的效果最好，对于Graph Embedding，Pooling based的效果显著好于Node based

# Graph Matching Networks for Learning the Similarity of Graph Structured Objects
-	google出品，实验和可视化结果一如既往的丰富
-	两点贡献：
	-	证明了GNN可以产生用于相似度计算的graph embedding
	-	提出了attention based的Graph Matching Networks，并超越了baseline

## Graph Embedding Model
-	baseline:Graph Embedding Model，一个简单的encode-propagation-aggregate模型
-	encode：将点和边的特征通过MLP编码得到embedding
-	proprgation：将中心点，邻接点，邻接边的embedding传递到下一层的中心点embedding
	$$
	\begin{aligned} \mathbf{m}_{j \rightarrow i} &=f_{\text {message }}\left(\mathbf{h}_{i}^{(t)}, \mathbf{h}_{j}^{(t)}, \mathbf{e}_{i j}\right) \\ \mathbf{h}_{i}^{(t+1)} &=f_{\text {node }}\left(\mathbf{h}_{i}^{(t)}, \sum_{j:(j, i) \in E} \mathbf{m}_{j \rightarrow i}\right) \end{aligned}
	$$
-	aggregate：作者用门控的方式将各个节点的embedding加权求和得到最后的graph embedding
	$$
	\mathbf{h}_{G}=\operatorname{MLP}_{G}\left(\sum_{i \in V} \sigma\left(\operatorname{MLP}_{\operatorname{gate}}\left(\mathbf{h}_{i}^{(T)}\right)\right) \odot \operatorname{MLP}\left(\mathbf{h}_{i}^{(T)}\right)\right)
	$$

## Graph Matching Networks
-	GMN并不像Baseline一样分别对两个图先生成embedding再match，而是接受两个图作为输入直接输出similarity score。
	$$
	\begin{aligned} \mathbf{m}_{j \rightarrow i} &=f_{\text {message }}\left(\mathbf{h}_{i}^{(t)}, \mathbf{h}_{j}^{(t)}, \mathbf{e}_{i j}\right), \forall(i, j) \in E_{1} \cup E_{2} \\ \boldsymbol{\mu}_{j \rightarrow i} &=f_{\text {match }}\left(\mathbf{h}_{i}^{(t)}, \mathbf{h}_{j}^{(t)}\right) \\ \forall i \in V_{1}, j & \in V_{2}, \text { or } i \in V_{2}, j \in V_{1} \\ \mathbf{h}_{i}^{(t+1)} &=f_{\text {node }}\left(\mathbf{h}_{i}^{(t)}, \sum_{j} \mathbf{m}_{j \rightarrow i}, \sum_{j^{\prime}} \mu_{j^{\prime} \rightarrow i}\right) \\ \mathbf{h}_{G_{1}} &=f_{G}\left(\left\{\mathbf{h}_{i}^{(T)}\right\}_{i \in V_{1}}\right) \\ \mathbf{h}_{G_{2}} &=f_{G}\left(\left\{\mathbf{h}_{i}^{(T)}\right\}_{i \in V_{2}}\right) \\ s &=f_{s}\left(\mathbf{h}_{G_{1}}, \mathbf{h}_{G_{2}}\right) \end{aligned}
	$$
-	从上面的公式可以看到，在propagation阶段，GMN做出了两点改动
	-	因为一次性输入一对图，因此第一步的邻域节点是从两张图的范围内找。但是一般而言两张图之间是没有节点连接的，除非两张图里的相同节点共享邻域?
	-	除了邻域信息的传递之外，作者还计算了两张图之间的match，这里用了一个最简单的attention机制，用待匹配两个节点embedding的距离加权两个节点embedding之间的差：
	$$
	\begin{aligned} a_{j \rightarrow i} &=\frac{\exp \left(s_{h}\left(\mathbf{h}_{i}^{(t)}, \mathbf{h}_{j}^{(t)}\right)\right)}{\sum_{j^{\prime}} \exp \left(s_{h}\left(\mathbf{h}_{i}^{(t)}, \mathbf{h}_{j^{\prime}}^{(t)}\right)\right)} \\ \boldsymbol{\mu}_{j \rightarrow i} &=a_{j \rightarrow i}\left(\mathbf{h}_{i}^{(t)}-\mathbf{h}_{j}^{(t)}\right) \end{aligned}
	$$
	-	这样在update到下一层节点embedding时，match的那部分实际上计算了a图某一结点与b图所有节点的加权距离：
	$$
	\sum_{j} \boldsymbol{\mu}_{j \rightarrow i}=\sum_{j} a_{j \rightarrow i}\left(\mathbf{h}_{i}^{(t)}-\mathbf{h}_{j}^{(t)}\right)=\mathbf{h}_{i}^{(t)}-\sum_{j} a_{j \rightarrow i} \mathbf{h}_{j}^{(t)}
	$$
	-	这样计算的复杂度就升到了$O(V(G_1)V(G_2))$，但正是这逐点的比较能够区分那些细微的变化，而且可视化更加具有可解释性。所以该算法的使用场景应该是小图且对区分精度要求高
-	对于这样的匹配问题可以用pair 或者triplet loss，前者比较相似不相似，后者比较和两个候选相比跟哪个更相似，作者分别给出了两种形式下的margin loss：
	$$
	L_{\text {pair }}=\mathbb{E}_{\left(G_{1}, G_{2}, t\right)}\left[\max \left\{0, \gamma-t\left(1-d\left(G_{1}, G_{2}\right)\right)\right\}\right] \\
	L_{\text {triplet }}=\mathbb{E}_{\left(G_{1}, G_{2}, G_{3}\right)}\left[\max \left\{0, d\left(G_{1}, G_{2}\right)-d\left(G_{1}, G_{3}\right)+\gamma\right\}\right] \\
	$$
-	作者还特意提到，为了加速运算，可以对Graph Embedding做二值化处理，这样在衡量距离的时候就是用汉明距离，牺牲掉了一些欧式空间的部分，具体做法是将整个向量过tanh并作平均内积用作训练时的图相似度，并设计损失将正样本对的汉明距离推向1，负样本对的汉明距离推向-1，假如在推断检索下使用汉明距离进行检索，这样的损失设计比margin loss更加稳定：
	$$
	s\left(G_{1}, G_{2}\right)=\frac{1}{H} \sum_{i=1}^{H} \tanh \left(h_{G_{1} i}\right) \cdot \tanh \left(h_{G_{2} i}\right) \\
	L_{\text {pair }}=\mathbb{E}_{\left(G_{1}, G_{2}, t\right)}\left[\left(t-s\left(G_{1}, G_{2}\right)\right)^{2}\right] / 4 \\
	\begin{aligned} L_{\text {triplet }}=\mathbb{E}_{\left(G_{1}, G_{2}, G_{3}\right)}\left[\left(s\left(G_{1}, G_{2}\right)-1\right)^{2}+\right.\\\left.\left(s\left(G_{1}, G_{3}\right)+1\right)^{2}\right] / 8 \end{aligned} \\
	$$
	其中除以4或者除以8是为了约束损失的范围在[0,1]区间内。

# Integration of Knowledge Graph Embedding into Topic Modeling with Hierarchical Dirichlet Process

# Learning to Update Knowledge Graphs by Reading News 

# Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs