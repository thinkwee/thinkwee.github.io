---
title: Notes for NLP with Graph-Structured Representations
date: 2020-04-05 21:22:54
tags:
  - gnn
  - math
  - graph
category: ML
mathjax: true
html: true
---

阅读来自University of Alberta
的Bang Liu博士的论文Natural Language Processing and Text Mining with Graph-Structured Representations，做一些笔记

<!--more-->

# 基于图结构表示的自然语言处理

- 论文主要囊括了主要包含四个方向的工作：
  - 事件抽取、故事生成
  - 语义匹配
  - 推荐
  - 阅读理解

# 相关领域及方法论

- 在NLP当中构图的相关工作
- 词作为节点
  - syntactic信息作为边：Learning substructures of document semantic graphs for document summarization
  - 共现信息作为边： Graph-of-word and tw-idf: new approach to ad hoc ir； Shortest-path graph kernels for document similarity；Directional skip-gram: Explicitly distinguishing left and right context for word embeddings；
- 句子、段落、文档作为节点
  - 词共现、位置作为边：Textrank: Bringing order into texts；
  - 相似度作为边：Evaluating text coherence based on semantic similarity graph；
  - 链接作为边：Pagerank
  - 混合：Graph methods for multilingual framenets
- 方法论
  - 明确输入输出、决定语义的细粒度、构图（定义节点和边，抽取节点和边的特征）、基于图的表示重构问题、进行实验
  - 在NLP中引入图的关键是引入结构信息和关系信息。
  - 基于图的表示重构问题，例如
    - 语义匹配：树或者图的匹配
    - 事件发现：社区发现
    - phrase挖掘：节点分类和排序
    - ontology creation：    关系鉴别
    - 问题生成：节点选择
      ![GG2kes.png](https://s1.ax1x.com/2020/04/02/GG2kes.png)

# 事件抽取及故事生成

## 相关工作

- 文本聚类
  - 基于相似度的方法，需要指定聚类数目
  - 基于密度的方法，不太适合文本这样的高维稀疏空间
  - 基于非负矩阵分解的方法（谱聚类）
  - 基于概率模型，例如PLSA，LDA，GMM
- 故事结构生成
  - 持续的将新事件归类到已有聚类当中
  - 想要为一系列时序事件生成故事总结，传统的基于summarization的方法不能持续生成，现在多采用基于贝叶斯模型的方法，但是gibbs sampling太耗时
- 作者提出了EventX方法，构建了Story Forest系统，与以上两个路线相关，做的是开放域新闻文档事件抽取

## Story Forest: Extracting Events and Telling Stories from Breaking News(TKDD)

- Story Forest：从新闻中抽取事件，生成故事
- 定义：一个topic相当于一个story forest，包含多个story tree；每个story tree的节点是一个event，event作为最小处理单元，实际上是关于某一个event的多篇新闻文档。本系统假设每篇文章只报道一个event
- 整体架构：
  ![GrQJun.png](https://s1.ax1x.com/2020/04/05/GrQJun.png)
- 主要关注EventX中如何做cluster events，是一个两层聚类
  - 构建一个keyword co-occurrence graph，节点是keyword，建边考虑两点：单篇文档内的相关性，即同一篇文档内共现次数超过阈值就建边；语料上的相关性，即条件概率超过阈值：
    
    $$
    \operatorname{Pr}\left\{w_{i} | w_{j}\right\}=\frac{D F_{i, j}}{D F_{j}}
    $$
  - 在这个keyword graph上做community detection，将keyword做一次聚类，认为每个类的Keyword描述同一个topic。这样每个topic是一系列keywords的集合，相当于一个文档（bag of words）
  - 计算每篇文档和每个topic之间的相似度，将文档分配给具有最大相似度的topic，这样就完成了第一层：文档按主题聚类
  - 将文档按topic分开之后，每个topic下还要细分event，这就是第二层聚类，event cluster的大小通常严重不均衡，作者提出了一种基于监督学习指导的聚类方法
  - 现在将每篇文档看成节点，希望谈论同一个event的文档之间建边，这里不太好人为设计规则，就使用了监督学习，训练了一个SVM来判断是否描述同一个event，获得document graph之后接着做community detection，完成第二层聚类

# 语义匹配

## 相关工作

- 文档级别的语义匹配，相关工作：文本匹配、文档图结构表示
- 文本匹配
  - 交互放在最后，先通过孪生网络提取embedding，然后待匹配的embedding pair进行score
  - 交互提前，先提取pair-wise的交互作为特征，然后通过神经网络聚合交互，最后score
- 文档图结构表示
  - word graph:通过句法剖析得到spo三元组构图，还可以通过wordnet扩展合并；基于窗口的方法，nodes represent unique terms and directed edges represent co-occurrences between the terms within a fixed-size sliding window；将依存关系作为边；基于超链接的构图；
  - text graph:节点是句子、段落、文档，边基于词级别的相似度、位置、共现建立。
  - concept graph:基于知识图谱，提取文档中的实体作为节点，例如DBpedia，然后通过最多两跳在知识图谱中进行dfs，找出outgoing relation and entity，构图；或者基于wordnet,verbnet，找出semantic role，用semantic/syntactic relation建边
  - hybrid graph:即异构图，多种节点以及多种边，lexical，tokens, syntactic structure nodes, part of speech nodes

## Matching Article Pairs with Graphical Decomposition and Convolutions(ACL 2019)

- 作者在文本匹配和文档图结构表示方向的工作是提出了一种文档建图的方式，然后用GCN提取文档特征，进行文档级别的匹配
- 这一部分可以改进上一个工作中两篇文档是否讨论同一个event的部分。
- 整体流程：
  ![GrQUEV.png](https://s1.ax1x.com/2020/04/05/GrQUEV.png)
- 构图：Concept Interaction Graph
  - 先构建key graph，抽取文章中的keywords和实体作为节点，假如两个节点出现在同一句中就建边
  - 可以将key graph中的节点直接作为concept，也可以在key graph上做一个overlapping community detection,将Key graph切分成多个相交的子图，子图作为concept节点，相交就建边
  - 现在concept 和 sentence都是bag of words，就可以计算相似度，将句子分配到concept
  - 现在concept是句子的集合，将其看成bag of words，concept之间就可以根据句子集合之间的tfidf vector similarity建边
  - 很关键的一点，由于是做matching，输入是句子对，在这一步变成了图对，作者将两个CIG合并成一个大CIG，将描述同一个concept的两篇文章的句子集合放在一个concept 节点中
- 构建matching network with GCN
  - 得到一个大的CIG之后，两篇文档之间的matching变成了大CIG当中每一个节点里来自两篇文档的sentence set之间的matching
  - 构建了一个端到端的流程：用孪生网络提取节点特征、用GCN做节点直接的特征交互、聚合所有特征做预测
  - 这里每个节点包含了两个句子集，将其拼接成两个长句，进孪生网络，分别用bilstm或者cnn提取到特征，再进行一个特征的聚合：
    
    $$
    \mathbf{m}_{A B}(v)=\left(\left|\mathbf{c}_{A}(v)-\mathbf{c}_{B}(v)\right|, \mathbf{c}_{A}(v) \circ \mathbf{c}_{B}(v)\right)
    $$
  - 这样就得到每个节点上的matching vector，可以理解为在该节点(concept)上两篇文档的相似度相关特征。此外作者还提取了一些传统特征相似度（tfidf,bm25,jaccard,Ochiai）的值拼接到matching vector当中
  - 接下来过GCN
  - 聚合所有节点特征就是一个简单的mean pooling，然后过一个线性层做分类（0，1）
- 在长篇新闻语料上（avg 734 token)，graph matching的效果远好于传统的双塔模型(DUET,DSSM,ARC都在50~60的f1，graph model达到了70~80)

## Matching Natural Language Sentences with Hierarchical Sentence Factorization(WWW 2018)

![GrQrv9.png](https://s1.ax1x.com/2020/04/05/GrQrv9.png)

- 句子对语义匹配任务，主要利用的是AMR剖析得到的句子结构
- 五个步骤
  - AMR parsing and alignment：AMR将句子表示为有向无环图，如果将有多个父节点的节点复制拆分，则可以将有向无环图转换为树。AMR的叶子节点代表一个概念，其余的是关系，代表概念之间的关联或者某个概念是另外一个概念的参数。得到AMR图之后还需要对齐，将句子中具体的token和概念建立连接。作者使用了现有的工具JAMR
  - AMR purification：一个token可能与多个概念建立连接，作者只选择最浅层的概念建立连接，之后将关系的内容删去，只保留边不保留边的label，得到简化之后的AMR tree，如上图所示
  - Index mapping：加入root节点，重设节点坐标
  - Node completion：类似于padding，保证两个句子的树的深度一样
  - Node traversal：做一次dfs,使得每个非叶子节点的内容拼接了其子树的内容
- 这样root节点就得到原句子的一个重新组织方式，类似于predicate-argument的形式，即谓语接谓语操作的词，类似于最初始的AMR purification得到的只保留概念的N叉树先做后序遍历再逆序，作者的意思大致是这样可以统一的表达两个句子，避免原文表示中各种因词序和其他不那么重要的词（助词介词）的现象导致接下来语义匹配中产生错误。
- 接下来使用Ordered Word Mover Distance。如下式，D代表传输的cost(embedding距离），T代表传输量(词占比），$\alpha,\beta$代表两个句子中各个词的词频向量，这里作者直接用均匀分布替代。传统的WMD没有考虑词的顺序，作者引入了两个惩罚项，当T主要集中在对角线上时I较大，P是T的一个理想分布，其第i行第j列个元素满足i,j到对角线位置的距离的标准高斯分布，也是希望T是一个对角阵。对角阵的意义就是会在传输时考虑顺序，不发生相对距离较远的传输。
  
  $$
  \begin{array}{ll}
\underset{T \in \mathbb{R}_{+}^{M \mathrm{XN}}}{\operatorname{minimize}} & \sum_{i, j} T_{i j} D_{i j}-\lambda_{1} I(T)+\lambda_{2} K L(T \| P) \\
\text { subject to } & \sum_{i=1}^{M} T_{i j}=\beta_{j}^{\prime} \quad 1 \leq j \leq N^{\prime} \\
& \sum_{j=1}^{N} T_{i j}=\alpha_{i}^{\prime} \quad 1 \leq i \leq M^{\prime}
\end{array} \\
  $$
  
  $$
  I(T)=\sum_{i=1}^{M^{\prime}} \sum_{j=1}^{N^{\prime}} \frac{T_{i j}}{\left(\frac{i}{M^{\prime}}-\frac{j}{N^{\prime}}\right)^{2}+1} \\
  $$
  
  $$
  P_{i j}=\frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{l^{2}(i, j)}{2 \sigma^{2}}} \\
  $$
  
  $$
  l(i, j)=\frac{\left|i / M^{\prime}-j / N^{\prime}\right|}{\sqrt{1 / M^{\prime 2}+1 / N^{\prime 2}}} \\
  $$
- 通过上述方法就能得到针对两个句子AMR表示的OWMD距离，完成无监督的文本相似度计算。也可以充分利用AMR Tree来完成监督学习。
- 注意到在AMR树中，不同层的节点其实代表了整句不同粒度的语义切分，例如第0层是整句，第一层是短句"Jerry little",第二层是单一的概念"Jerry"，作者选取前三层，将每一层不同粒度的语义单元之间输入context layer，同一语义单元内的token embedding相加作为单一embedding，并假设每一层的子节点最多k个，不足的padding。
  ![GbgQUA.png](https://s1.ax1x.com/2020/04/11/GbgQUA.png)

# 推荐

## 相关工作

## A User-Centered Concept Mining System for Query and Document Understanding at Tencent(KDD 2019)

## Scalable Creation of a Web-scale Ontology(SIGMOD 2020)

# 阅读理解

## Learning to Generate Questions by Learning What not to Generate(WWW 2019)

## Asking Questions the Human Way: Scalable Question-Answer Generation from Text Corpus(WWW 2020)