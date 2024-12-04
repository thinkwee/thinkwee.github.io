---
title: Note for Graph-based Summarization
date: 2019-10-03 17:18:15
categories: NLP
tags:
  - graph neural network
  - deep learning
  - summarization
  - natural language processing
mathjax: true
html: true
---

Graph-based Automatic Summary Related Paper Selection Reading

- AMR Generative Summary
- AMR Multi-document Summarization Two Papers
- pagerank in encoder attention
- Build a graph based on thematic modeling, use ILP for extractive summarization
- Multi-document Extractive Summary Based on GCN
- STRUCTURED NEURAL SUMMARIZATION
  
<!--more-->

{% language_switch %}

{% lang_content en %}
# Toward Abstractive Summarization Using Semantic Representations

- Explored how to construct an AMR graph for summarization from the original AMR graph, i.e., graph summarization
- Three-step approach, source graph construction, subgraph prediction, text generation

## Source Graph Construction

- This step involves merging multiple graphs into a source graph, and some concept nodes need to be combined
- Firstly, AMR does not repeat the modeling of the mentioned concept, but supplements the frequency mentioned as a feature into the node embedding
- Node merging consists of two steps: merging the subtrees of some nodes, followed by merging nodes with the same concepts
- The node names after subtree merging contain information from all subtree nodes, and only nodes that are completely identical can be merged thereafter. Therefore, nodes that have undergone one subtree merge are difficult to merge again (difficult to be completely identical), and some coreference resolution work needs to be done (future work)
- Some nodes will have multiple edges, take the two edges with the most occurrences, merge them, and discard the other edges
- Directly merge the same concept nodes
- Add a total root node to connect the root nodes of each sentence
- The source graph connected in this way has a low edge coverage for the gold summary graph, so further post-processing is done on the source graph, adding null edges between all nodes to improve coverage

## Subgraph Prediction

- Subgraph prediction problem is a structured prediction problem
  
- The author constructs the subgraph scoring function for the model parameters as a linear weighted combination of edge and node features:
  
  $$
  \operatorname{score}\left(V^{\prime}, E^{\prime} ; \boldsymbol{\theta}, \boldsymbol{\psi}\right)=\sum_{v \in V^{\prime}} \boldsymbol{\theta}^{\top} \mathbf{f}(v)+\sum_{e \in E^{\prime}} \boldsymbol{\psi}^{\top} \mathbf{g}(e)
  $$
- decoding: Selects the subgraph with the highest score based on ILP. The constraint is that the selected subgraph must be legal and connected, which can be described by the indicator function v, e, and the flow f. $v\_i=1$ means that the i-th node is selected, $e\_{i,j}=1$ means that the edge between nodes i and j is selected, and $f\_{i,j}$ represents the flow from i to j. Then, the legal condition is:
  
  $$
  v_{i}-e_{i, j} \geq 0, \quad v_{j}-e_{i, j} \geq 0, \quad \forall i, j \leq N
  $$
- Union: The flow that originates from the root reaches each selected conceptual node, each conceptual node consumes a flow, and the flow can only pass through when the edges are selected. These three constraints are described mathematically as:
  
  $$
  \begin{array}{r}{\sum_{i} f_{0, i}-\sum_{i} v_{i}=0} \\ {\sum_{i} f_{i, j}-\sum_{k} f_{j, k}-v_{j}=0, \quad \forall j \leq N} \\ {N \cdot e_{i, j}-f_{i, j} \geq 0, \quad \forall i, j \leq N}\end{array}
  $$
- The author only assumes that each concept has only one parent node, i.e., constructed in the form of a tree
  
  $$
  \sum _j e_{i,j} \leq 1, \quad \forall i, j \leq N
  $$
- This form of ILP has appeared in sentence compression and dependency parsing, and the author has completed optimization using Gurobi's ILP algorithm
  
- A constraint can be added to limit the length of the summary, for example, the total number of selected edges not exceeding L
  
- The above is decoding, i.e., selecting subgraphs, but the selection of graphs is based on scores, and the scores are weighted by parameters, thus also including an optimization of parameters. We need a loss function to measure the gap between the decoded graph and the gold summary graph, however, the gold summary graph may not be in the source graph. The authors refer to ramp loss from machine translation, comparing the perceptron loss used by the perceptron, the hinge loss in structured SVM, and the ramp loss, where $G$ is the source graph, and $G^{\*}$ is the gold summary graph:
  
  $$
  \begin{array}{ll}{\text {perceptron loss: }} & {-\text { score }\left(G^{*}\right)+\max _{G} \text { score }(G)} \\ {\text {hinge loss: }} & {-\text { score(G^{*} ) }+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)} \\ {\text {ramp loss: }} & {-\max _{G}\left(\text {score}(G)-\operatorname{cost}\left(G ; G^{*}\right)\right)+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)}\end{array}
  $$
- cost penalty for redundant edges
  
- Perceptron loss is very simple, it is to minimize the score gap between the gold graph and the decoded graph
  
- hinge loss is added to the ILP with a penalty for redundant edges, making the score of the decoded graph as large as possible, not just close to the gold graph, here the score of the decoded graph will be slightly lower than the graph obtained by direct score calculation
  
- ramp loss compared to hinge loss is that an inverse penalty is added to the former, while the actual ramp loss still narrows the score gap between the two images, with one image having a slightly higher score than the best decoded graph and the other slightly lower, relaxing the conditions
  

## Generation

- Authors currently only counted the text span corresponding to the concept nodes in the decoded graph, and did not generate a readable summary, therefore only ROUGE-1 was calculated

# Abstract Meaning Representation for Multi-Document Summarization

- This is an extension of the previous one
- Using AMR to construct a rooted directed acyclic graph, where nodes are concepts and edges are semantic relationships:
  - Node: May be a frameset (frame) in PropBank, an ordinary English word, a special category word, or a string,
  - Edge: It can be a predicate relationship from PropBank or a modified relationship
- The entire system consists of three parts
  - source sentence selection: input a series of articles and then pick out sentences from different aspects of a certain topic
  - Content planning: Input a series of sentences, output an abstract diagram
  - Surface realization: Convert diagrams into readable summary sentences
- Three components can be optimized with domain-specific small corpora separately

## Source Sentence Selection

- Because it is a multi-document summary, spectral clustering is performed for each input example (multiple documents), and several sentences are then selected from each cluster
- There are multiple sentence groups, and just like the modified input summary model, it is necessary to reconstruct training pairs. Here, it is to construct the training pairs to be provided for content planning, that is, the sentence groups and their corresponding gold summary AMR graphs. For each sentence in the gold summary, calculate an average similarity with the sentence group, and select the one with the highest similarity as the sentence group in the training pair. The average similarity includes:
  - LCS
  - VSM
  - Smatch method, referring to the paper Smatch: an evaluation metric for semantic feature structures
  - Concept Coverage, i.e., the concept in the maximum coverage gold summary AMR graph
- Four similarity measures have also been ablated

## Content Planning

- Training is for the AMR graph of sentence pairs and summaries, naturally this part is about learning this transformation process
  
- Firstly, the summary in the sentence group needs to be converted into an AMR graph; the author tried two AMR Parsers, JAMR and CAMR
  
- After that, convert each sentence in the sentence group into an AMR graph and then merge them (this part of the paper is not described clearly)
  
  - Are same concept nodes merged?
  - Perform coreference resolution, merge nodes with the same reference concepts
  - Some special nodes require integrating subtrees into the node information, called mega-nodes, which is essentially canceling unnecessary expansion and directly writing the specific expansion information into the node, for example, date entity: year 2002: month 1: day 5. These mega-nodes can only be merged if they are completely identical.
  - Generate a root node finally, and connect the root nodes of various subgraphs
  - Through the last operation, does the merging of seemingly identical concept nodes represent the merging of the same nodes within the same subgraph?
- Next, design an algorithm to identify the summarized AMR graph from the source AMR graph, which includes two parts
  
  - Graph Decoding: Identifies an optimal summary graph through Integer Linear Programming (ILP): First, construct a parameterized graph scoring function, where each node feature and edge feature is weighted by parameters and summed to obtain a score; here, the features are manually constructed, referring to Fei Liu's series of AMR summarization papers; next, perform an ILP to find a subgraph that maximizes the score, with the constraint of L nodes and the subgraph being connected.
    
  - Parameter update: Minimize the gap between the abstract graph decoded by the system and the gold summary graph. This step optimizes the feature weighting parameters in the scoring function from the previous step. Construct a loss function to measure the gap between the decoded graph and the gold graph. Sometimes, the gold graph cannot be decoded from the source graph, in which case structured ramp loss is used, considering not only the score but also the cost, i.e., the degree of agreement between the gold graph and the decoded graph on whether to include a certain node or edge in the summary.
    
    $$
    L_{ramp}(\theta, \phi) = max_G (score(G)+cost(G;G_{gold})) - max_G(score(G) - cost(G;G_{gold}))
    $$

## Surface Realization

- Convert image to sentence
- AMR graphs do not convert into sentences because the graphs do not contain grammatical information; a graph may generate multiple sentences that are not grammatically correct. The author takes a two-step approach, first converting the AMR graph into PENMAN form, and then using the existing AMR-to-text to convert PENMAN into sentences

# Towards a Neural Network Approach to Abstractive Multi-Document Summarization

- This paper is an extension of the previous paper, expanding from single-document summarization to multi-document summarization, mainly focusing on how to transfer pre-trained models on large-scale single-document summarization datasets to the multi-document summarization task
  
- Compared to the single-document model, an additional document-level encoding layer is added on the encoding side. There is no dependency or sequential relationship between documents, so there is no need to use RNN. The authors directly use linear weighting. It is worth noting that the weights of this weighting should not be fixed or directly learned, but should be determined based on the document itself. Therefore, the authors add a dependency relationship learned from the document itself and the relationship between the document set:
  
  $$
  w_{m}=\frac{\mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}{\sum_{m} \mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}
  $$
- The mechanism of attention remains essentially unchanged, with the decoder's initial state transitioning from single-document encoding to multi-document encoding, and the attention weighting shifting from the number of sentences in a single document to the number of sentences in multiple documents. One issue that arises here is that the number of sentences in multi-documents is too large, with many attentions being distributed very evenly, resulting in an excessive amount of information after weighting. Therefore, the authors truncate the global soft attention, allowing only the top k sentences to be weighted, with the rest of the sentences being discarded directly during encoding.
  
- The migration from single-document to multi-document actually is not the focus of the paper. The author trains the single-document model part on CNN/DM and then trains the multi-document part on a small DUC dataset, but these two datasets are quite consistent. Many works trained on CNNDM and tested on DUC can achieve good results.
  
- The paper's ablation is very detailed, comparing the effects under various functional graph model methods, including Textrank, Lexrank, Centroid
  
- It is noteworthy that the author uses edit distance to measure the abstractness of the abstract
  

# Abstractive Document Summarization with a Graph-Based Attentional Neural Model

- A paper by the team of Teacher Wan, with very good ideas, the important parts are in two points:
  
  - hierarchical encoder and decoder: Since encoding and decoding at the sentence level are required to adapt to the graph scoring operation, a hierarchical seq2seq is adopted, with both encoding and decoding at the word-level and sentence-level
  - graph-attention: The graph used here is actually a fully connected graph from pagerank, where similarity is directly measured by the inner product of the hidden vectors of enc-dec, and then the topic-aware pagerank is used to recalculate the sentence-level attention weights.
- In the encoding-decoding stage, we use hidden layers to calculate similarity, which is the same as the original attention, but the original attention adds a parameter matrix (modern attention doesn't even bother to add a parameter matrix), so this similarity can reflect the attention weight (score). Then, graph-attention directly calculates the Markov chain iteration of pagerank on this similarity, considering the stable distribution $f$ of the Markov chain to be the sentence score after re-ranking. There is something the paper doesn't mention; the author makes an assumption that the state obtained during encoding-decoding is already in a stable state, rather than starting from scratch, so we can let $f(t+1)=f(t)=f$ and directly calculate the stable distribution:
  
  $$
  \mathbf{f}(t+1)=\lambda W D^{-1} \mathbf{f}(t)+(1-\lambda) \mathbf{y} \\
  \mathbf{f}=(1-\lambda)\left(I-\lambda W D^{-1}\right)^{-1} \mathbf{y} \\
  $$
- The basic form is consistent with pagerank, part of which is based on salience allocation from a similarity matrix, and the other part supplements a uniform distribution y to ensure the convergence of the Markov chain (here it seems to be abbreviated, writing the uniform transition matrix multiplied by f directly as a uniform distribution). It is noteworthy that this calculation is done on the encoding and decoding hidden layer states at the sentence level, therefore it is the graph attention score of various encoding sentences given a certain decoding sentence. How to reflect this certain decoding sentence? That is to use topic-aware pagerank, treat the decoding sentence as a topic, add this topic sentence to the pagerank graph, and change y from a uniform distribution to a one-hot distribution, which ensures the influence of the decoding sentence in the graph and thereby influences other sentences.
  
- Afterwards, the distraction attention mechanism was adopted to prevent repeated attention:
  
  $$
  \alpha_{i}^{j}=\frac{\max \left(f_{i}^{j}-f_{i}^{j-1}, 0\right)}{\sum_{l}\left(\max \left(f_{l}^{j}-f_{l}^{j-1}, 0\right)\right)}
  $$
- Some minor techniques have also been applied at the decoding end, including:
  
  - Handling OOV, use @entity+word length as a label to replace all entities that are prone to become OOV, and attempt to restore the entity labels generated in the decoded sentence, searching in the original text according to word length
  - hierarchical beam search: word-level beam search scoring considers the original sentence of "attend to" and the bigram overlap of the currently generated part, hoping for a larger overlap; sentence-level beam search hopes that the original sentence attended to is different for each generated sentence, this description is not very clear, it should be that N different original sentences are attended to when generating each sentence, producing N different decoded sentences
- The hierarchical decoding in this article actually plays a very crucial role; the author did not use word-level attention all at once but rather reordered based on the sentence relationship component diagram and also fully utilized two levels of information in beam search
  
# Topical Coherence for Graph-based Extractive Summarization

- Build a graph based on thematic modeling, use ILP for extractive summarization
- The author used a bipartite graph, with one side being sentence nodes and the other side being topic nodes, connected by edges. The weight of the edges is the sum of the logarithms of the probabilities of all words in a sentence under a certain topic, normalized by the length of the sentence
- Using HITS algorithm to calculate the importance of sentences on a bipartite graph

# Graph-based Neural Multi-Document Summarization

- Using GCN for extractive summarization, here GCN plays a role of feature supplementation. The original approach is a two-level GRU, where documents are clustered to create embeddings, with each sentence having an embedding. Then, similar to IR, a comparison is made between sentence embeddings and document embeddings to calculate salience scores. Afterward, a greedy method is used to extract sentences based on the scores, with the overall framework still being the scoring-extraction approach
- GCN added two layers between the GRUs, i.e., the embedding of sentences under a sentence relationship graph was performed with three layers of GCN, followed by the generation of document embeddings by the GRU at the document level
- Here are two points to focus on: how to construct the sentence relationship diagram
- The author of the sentence relationship diagram tried three methods:
  - The most naive, TF-IDF cosine distance
  - ADG in the paper "Towards Coherent Multi-Document Summarization"
  - Author's Improved PDG on ADG
- After that, simply apply GCN propagation
{% endlang_content %}

{% lang_content zh %}

# Toward Abstractive Summarization Using Semantic Representations

- 探讨了如何从原文的AMR图构建摘要的AMR图，即graph summarization
- 三步走，source graph construction, subgraph prediction,     text generation

## Source Graph Construction

- 这一步是将多句graph合并为source graph，一些concept node需要合并
- 首先AMR不会重复建模提到的concept，而是将提到的频次作为特征补充进node embedding当中
- 节点合并包含两步：把一些节点的子树合并，接着把相同的概念节点合并
- 子树合并之后的节点名称里包含了所有子树节点的信息，之后只有完全相同的节点才能合并，因此经过一次子树合并之后的节点很难再次合并（很难完全相同），这里需要做一些共指消解的工作(future work)
- 一些节点之间会有多条边，取出现次数最多的两个边的label，合并，抛弃其他边
- 相同的概念节点直接合并
- 加一个总的root节点连接各个句子的root节点
- 这样连接出来的source graph对于gold summary graph的边覆盖度不高，因此对于source graph还后处理一下，将所有节点之间加入null 边，提高覆盖率

## Subgraph Prediction

- 子图预测问题是一个structured prediction problem
- 作者构建子图打分函数为模型参数线性加权边和节点的特征：
  
  $$
  \operatorname{score}\left(V^{\prime}, E^{\prime} ; \boldsymbol{\theta}, \boldsymbol{\psi}\right)=\sum_{v \in V^{\prime}} \boldsymbol{\theta}^{\top} \mathbf{f}(v)+\sum_{e \in E^{\prime}} \boldsymbol{\psi}^{\top} \mathbf{g}(e)
  $$
- decoding：基于ILP选出得分最大的子图。这里的约束条件是选出的子图必须合法且是联通的，可以通过指示函数v,e和流量f来描述。$v_i=1$即第i个节点被选中，$e_{i,j}=1$即i,j两个节点之间的边被选中，$f_{i,j}$代表从i流向j的流量，那么合法即：
  
  $$
  v_{i}-e_{i, j} \geq 0, \quad v_{j}-e_{i, j} \geq 0, \quad \forall i, j \leq N
  $$
- 联通即：从根流出的流量到达选中的每一个概念节点，每一个概念节点消耗一个流量，只有边被选中时流量才可能通过，这三个约束用数学描述为：
  
  $$
  \begin{array}{r}{\sum_{i} f_{0, i}-\sum_{i} v_{i}=0} \\ {\sum_{i} f_{i, j}-\sum_{k} f_{j, k}-v_{j}=0, \quad \forall j \leq N} \\ {N \cdot e_{i, j}-f_{i, j} \geq 0, \quad \forall i, j \leq N}\end{array}
  $$
- 另外作者只假设了每一个概念只有一个父节点，即构建为树的形式
  
  $$
  \sum _j e_{i,j} \leq 1, \quad \forall i, j \leq N
  $$
- 这种形式的ILP在sentence compression和dependency parsing中都出现过，作者使用gurobi的ILP算法完成最优化
- 可以附加一个约束来限制摘要的长度，例如选中的边总数不大于L
- 以上是decoding，即选子图，但选图基于分数，而分数由参数加权，因此还包含了一个参数的优化。我们需要一个损失函数来衡量decoded graph和gold summary graph之间的差距，然而gold summary graph可能不在source graph当中，作者借鉴了机器翻译中的ramp loss，作者对比了感知机所用的perceptron loss, structured SVM中的hinge loss以及ramp loss，其中$G$是source graph，$G^{*}$ 是gold summary graph：
  
  $$
  \begin{array}{ll}{\text {perceptron loss: }} & {-\text { score }\left(G^{*}\right)+\max _{G} \text { score }(G)} \\ {\text {hinge loss: }} & {-\text { score(G^{*} ) }+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)} \\ {\text {ramp loss: }} & {-\max _{G}\left(\text {score}(G)-\operatorname{cost}\left(G ; G^{*}\right)\right)+\max _{G}\left(\text {score}(G)+\operatorname{cost}\left(G ; G^{*}\right)\right)}\end{array}
  $$
- cost对多余的边惩罚
- perceptron loss很简单，就是希望缩小gold graph与decoded graph之间的分数差距
- hinge loss在ILP中加入对多余边的惩罚，使得decoded graph的分数尽可能大，而不仅仅是和gold graph接近，这里decoded的graph会比直接计算分数得到的graph分值上差一点
- ramp loss相比hinge loss就是在前面一项加了一个反向的惩罚，实际ramp loss依然是在缩小两个图的分数差距，只不过一个图比best decoded graph分值高一点，另一个比best decoded graph低一点，放宽松了条件

## Generation

- 作者目前只统计了decoded graph中概念节点对应的text span，并没有生成可读的摘要，因此只计算了ROUGE-1

# Abstract Meaning Representation for Multi-Document Summarization

- 这是上一篇的扩展
- 用AMR构建有根有向无环图，节点是概念，边是语义关系:
  - 节点：可能是PropBank里的一个frameset（命题），一个普通英语单词，一个特殊类别词，一个字符串，
  - 边：可以是PropBank里的命题关系，或者魔改之后的关系
- 整个系统三个部分
  - source sentence selection：输入一系列文章，然后挑出关于某一主题不同方面的句子
  - content planning：输入一系列句子，输出摘要图
  - surface realization：将图转换为可读的摘要句
- 三个组件可分别用领域内小语料优化

## Source Sentence Selection

- 因为是多文档摘要，因此对每一个输入样例（多篇文档），做谱聚类，每个簇再挑若干句子
- 这样就有多个句子组，之后和更改了输入的摘要模型一样，需要重新构造训练对，这里是要构造接下来提供给content planning的训练对，即句子组和对应的gold summary的AMR graph。就对gold summary里的每一句，和句子组算一个平均相似度，选相似度大的作为训练对里的句子组。平均相似度有：
  - LCS
  - VSM
  - Smatch方法，参考了论文Smatch: an evaluation metric for semantic feature structures
  - Concept Coverage，即最大覆盖gold summary AMR graph里的concept
- 四种相似度也做了ablation

## Content Planning

- 训练对是句子组和summary的AMR graph，自然这个部分就是学习这个转换过程
- 首先要把句子组里的summary转成AMR graph，作者试用了两种AMR Parser，JAMR和CAMR
- 之后把句子组里的每一句也转成AMR graph，并且做合并（这一部分论文描述并不清楚）
  - 相同概念节点合并？
  - 做共指消解，把相同指代概念节点合并
  - 一些特殊节点需要把子树整合进节点信息里，叫mega-node，其实就是取消不必要的展开，将展开的具体信息直接写进节点里，例如date entity :year 2002 :month 1 :day 5。这些mega-node只有完全相同时才能合并
  - 最后生成一个root节点，把各个子图的root节点连起来
  - 通过最后一个操作貌似相同概念节点合并是同一子图内相同节点合并？
- 接下来设计算法，从源AMR graph中识别出摘要的AMR graph，包含两部分
  - graph decoding：通过整数线性规划(ILP)识别出一个最优摘要图：首先构造一个参数化的图打分函数，将每一个节点特征和边特征通过参数加权并累加得到分数，这里的特征是手工构造，参考Fei Liu他的一系列AMR summarization的论文；接下来做一个ILP，要求找一个子图，使得得分最大，限制为L个节点而且子图是连接的。
  - parameter update：最小化系统解码出的摘要图和gold summary图之间的差距。这一步优化的是上一步打分函数中的特征加权参数。构造损失函数来衡量decoded graph和gold graph之间的差距。有时gold graph不能从source graph中解码出来，这时就采用structed ramp loss，不仅仅考虑score，还考虑cost，即gold graph和decoded graph就是否将某个节点或者边加入摘要达成一致的程度
    
    $$
    L_{ramp}(\theta, \phi) = max_G (score(G)+cost(G;G_{gold})) - max_G(score(G) - cost(G;G_{gold}))
    $$

## Surface Realization

- 将图转成句子
- AMR图并不好转成句子，因为图并不包含语法信息，一个图可能生成多句不合法的句子，作者两步走，先将AMR图转成PENMAN形式，然后用现有的AMR-to-text来将PENMAN转成句子

# Towards a Neural Network Approach to Abstractive Multi-Document Summarization

- 这篇论文是上篇论文的扩展，从单文档摘要扩展到多文档摘要，主要是如何将大规模单文档摘要数据集上预训练好的模型迁移到多文档摘要任务上
- 相比单文档模型，编码端又加了一层文档级别的编码，文档之间并没有依存或者顺序关系，因此没必要用RNN，作者直接用了线性加权,值得注意的是这个加权的权重不应该是固定或者直接学习出来的，而应该根据文档本身决定，因此作者给权重加了一个依赖关系学习出来，依赖文档本身和文档集的关系：
  
  $$
  w_{m}=\frac{\mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}{\sum_{m} \mathbf{q}^{T}\left[\mathbf{d}_{m} ; \mathbf{d}_{\Sigma}\right]}
  $$
- 注意力的机制基本不变，decoder的初始状态从单文档变成多文档编码，注意力加权从单篇文档句子数量到多篇文档句子数量。这里带来的一个问题是多文档的句子数量太大了，很多注意力被分散的很均匀，加权之后包含的信息量太大。因此作者将global soft attention给截断了一下，只有top k个句子可以用权重加权，其余的句子直接在编码中被抛弃
- 单文档到多文档的迁移其实并不是论文的重点，作者在CNN/DM上训练单文档的模型部分，之后在少量DUC数据集上训练多文档的部分，但是这两个数据集挺一致的，很多工作在CNNDM上训练在DUC上测试也能取得不错的效果。
- 论文的ablation做的非常详细，对比了多种功能图模型方法下的效果，包括Textrank,Lexrank,Centroid
- 值得注意的是作者使用了编辑距离来衡量文摘的抽象程度

# Abstractive Document Summarization with a Graph-Based Attentional Neural Model

- 万老师团队的一篇论文，想法非常的好，重要的部分在两点：
  - hierarchical encoder and decoder：由于需要在句子级别上做编解码以适应图打分的操作，所以采用了分层的seq2seq，无论编码解码都是word-level加sentence-level
  - graph-attention：这里用的图是其实是pagerank里的全连接图，相似度直接用enc-dec的隐层向量内积来衡量，然后利用topic-aware pagerank来重新计算句子级别注意力权重。
- 在编解码阶段，我们利用隐层来计算相似度，这和原始的attention是一样的，只不过原始的attention加了一个参数矩阵（现代的attention连参数矩阵都懒得加了）使得这个相似度能够体现出注意力权重（分数），那么graph-attention就是在这个相似度上直接计算pagerank的markov链迭代，认为马氏链的稳定分布$f$就是重新rank之后的句子分数，这里有一点论文里没讲，作者做了一个假设，即编解码时拿到的已经是稳定状态，而不是从头迭代，因此可以令$f(t+1)=f(t)=f$，直接算出稳定分布：
  
  $$
  \mathbf{f}(t+1)=\lambda W D^{-1} \mathbf{f}(t)+(1-\lambda) \mathbf{y} \\
\mathbf{f}=(1-\lambda)\left(I-\lambda W D^{-1}\right)^{-1} \mathbf{y} \\
  $$
- 基本形式与pagerank一致，一部分是基于相似矩阵的salience分配，另一部分补上一个均匀分布$y$保证马氏链收敛(这里感觉应该是简略了了，把均匀转移矩阵乘以f直接写成了均匀分布)，值得注意的是这是在sentence-level的编解码隐层状态做的计算，因此是计算给定某解码句下，各个编码句的graph attention score，如何体现这个给定某解码句？那就是用topic-aware pagerank，将解码句看成topic，把这个topic句加入pagerank的图里，并且y从均匀分布改成one-hot分布，即保证了解码句在graph中的影响力，并借此影响其他句子。
- 之后借鉴了distraction attention使得注意力不重复：
  
  $$
  \alpha_{i}^{j}=\frac{\max \left(f_{i}^{j}-f_{i}^{j-1}, 0\right)}{\sum_{l}\left(\max \left(f_{l}^{j}-f_{l}^{j-1}, 0\right)\right)}
  $$
- 在解码端也做了一些小技巧，包括：
  - OOV的处理，用@entity+单词长度来作为标签替换所有容易成为OOV的实体，并尝试把解码句中生成的实体标签还原，根据单词长度在原文中查找
  - hierarchical beam search：word-level的beam search打分考虑了attend to的原文句子和当前生成部分的bigram overlap，希望这个overlap越大越好；sentence-level的beam search则希望生成每一句时attend to的原文句子不相同，这一段描述不是很清楚，应该是生成每一句时会attend N个不同的原文句产生N个不同的decoded sentence
- 本文的层次编解码其实起到了很关键的作用，作者并没有一股脑用单词级别的注意力，还是根据句子关系构件图并重排序，在beam search也充分利用了两个层次的信息
- 从ablation来看，graph attention和sentence beam的效果其实不大，影响ROUGE分数最大的是考虑了bigram overlap的word-level beam search，这也暴露了ROUGE的问题，即我们之前工作中提到的OTR问题

# Topical Coherence for Graph-based Extractive Summarization

- 基于主题建模构建图，使用ILP做抽取式摘要
- 作者使用了二分图，一边是句子节点，一边是主题节点，两组节点之间用边连接，边的权值是句子中所有单词在某一主题下概率的对数和，除以句子长度做归一化
- 使用HITS算法在二分图上计算句子的重要程度

# Graph-based Neural Multi-Document Summarization

- 用GCN做抽取式摘要，在这里GCN起到了一个特征补充的作用，原始的做法就是一个two-level GRU，documents cluster做一个embedding，其中每一个sentence有一个embedding，然后类似IR，拿sentence embedding和documents embedding做一个比较算出salience score，之后再用一个贪心的方法根据分数抽句子，大框架依然是打分-抽取的思路
- GCN加进了两层GRU之间，即句子的embedding在一个句子关系图下做了三层GCN，之后再由documents层次的GRU生成documents embedding
- 这里就关注两点：句子关系图如何构建
- 句子关系图作者试了三种：
  - 最naive的，tfidf的cosine距离
  - Towards Coherent Multi-Document Summarization一文中的ADG
  - 作者在ADG上改进的PDG
- 之后直接套GCN传播就行了

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
