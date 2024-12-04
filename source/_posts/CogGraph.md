---

title: Study Notes for Cognitive Graph
date: 2019-08-13 14:22:49
categories: NLP
tags:
- machine learning
- gnn
- bert
- natural language processing
mathjax: true
html: true

---

Note for paper "Cognitive Graph for Multi-Hop Reading Comprehension at Scale." 


<!--more-->

![mClGgH.png](https://s2.ax1x.com/2019/08/13/mClGgH.png)

{% language_switch %}

{% lang_content en %}
Task
====

*   The framework proposed by the author is called CogQA, which is a framework based on cognitive graphs to address question-answering in machine reading comprehension, including general question-answering (selecting entities) and comparative question-answering (the relationship between two entities). Setting cognitive graphs aside for the moment, let's look at what this task is.
*   The uniqueness of this question-answering task lies in its extension to multi-hop. Multi-hop should not be a task type but rather a method for completing entity-based question-answering tasks, that is, finding entities in the question, then locating clues in their corresponding context descriptions, and using these clues to find the next entity as the next hop. The next-hop entity is used to jump to the corresponding context, where clues are then found again, and this process is repeated until, after multiple hops, the correct entity is found in the correct description as the answer. For a question, you can solve it using a single-hop approach or a multi-hop approach. In fact, most question-answering models based on information retrieval are based on a single-hop approach, which simply compares the question and the context to find the most relevant sentence and then extract entities from these sentences. Essentially, this is a pattern matching approach, but the problem is that if the question itself is multi-hop, then the single-hop pattern matching may not be able to find the correct entity at all, because the answer is not in the candidate sentences.
*   This is actually very similar to the way humans answer questions. For example, if we ask who the authors are of the cognitive graph published in ACL2019, we would first find the two entities, ACL2019 and cognitive graph, and then separately find all the corresponding authors of ACL2019 papers and the various meanings of the cognitive graph (it may be neuroscience, it may be education, it may be natural language processing), and then find more entities and descriptions (different authors of different papers, different explanations of meanings), ultimately finding one or more answers. Humans might directly search for the term "cognitive graph" in all the paper titles of ACL2019, while computers might extend and jump between ACL2019 and cognitive graph multiple times before merging into a single entity, namely the author's name, and then output it as an answer.
*   The multi-hop connections among multiple entities and their topological relationships constitute the cognitive graph, a directed graph. This directed graph is both inferable and interpretable, unlike black-box models, with a clear inference path. Therefore, the issue boils down to:
    *   How to construct a graph?
    *   With the diagram, how to reason?
*   The author first proposed using the dual-process theory from cognitive science to explain their approach.

Two-process model
=================

*   The dual-process model in cognitive science refers to the fact that humans solve problems in two steps:
    *   System 1: Initially, attention is allocated through an implicit, unconscious, and intuitive process to retrieve relevant information
    *   System Two: Reasoning is completed through another explicit, conscious, and controllable process
    *   System one provides resources to system two, system two guides the search of system one, and they iterate in this manner
*   The above two processes can actually correspond to two major schools of thought in artificial intelligence, connectionism and symbolism. The first process, although difficult to explain and completed through intuition, is not innate but actually obtained through hidden knowledge gained from life experience. This part can correspond to the black-box models currently completed by deep learning, which learn from a large amount of data to produce models that are unexplainable but can achieve the intended purpose. While the second process requires causal relationships or explicit structures to assist in reasoning.
*   In the context of machine question answering, the authors naturally employed existing neural network models to accomplish these two tasks:
    *   The first item requires attention to retrieve relevant information, so I directly use the self-attention model to complete entity retrieval.
    *   The second item requires an explicit structure, so I construct a directed cognitive graph and complete reasoning on the cognitive graph.

How to Construct a Graph
========================

*   The author uses BERT to complete the work of System 1. BERT itself can be used as a one-step machine reading comprehension, and here the author follows the one-step approach, with the input sentence pairs being the question and the sentence to be annotated, and the output being the annotation probability, i.e., the probability that each word is the start or end position of an entity. However, to achieve multi-hop, the author made some modifications:
    *   The input sentence pairs are not based on the problem as a unit, but on each entity within each problem. Specifically, the A sentence of each input sentence pair is composed of the problem and a clue to a certain entity within the problem, while the B sentence is about all the sentences in the descriptive context of that entity.
        *   sentence A:$[CLS]Question[SEP]clues(x,G)[SEP]$
        *   sentence B:$Para(x)$
    *   What is the clue of a certain entity? The clue is the sentence describing the entity extracted from the context of all parent nodes in the cognitive graph. It may sound a bit awkward, but this design runs through the entire system and is the essence of its cognitive reasoning, as illustrated by the examples given in the paper:
        *   Who made a movie in 2003 with a scene shot at the Los Angeles Quality Cafe?
        *   We found the entity "quality cafe," and found its introduction context: "......This is a coffee shop in Los Angeles. The shop is famous for being the filming location for several movies, including Old School, Gone in 60s, and so on."
        *   We then proceed to traverse these movie name entities, and then find the introduction context of the movies, such as "Old School is an American comedy film released in 2003, directed by Todd Phillips," and through other cognitive reasoning, we deduce that this "Todd Phillips" is the correct answer. Then, what is the clue for this director entity? What kind of clues do we need as supplementary input to obtain "Old School is an American comedy film released in 2003, directed by Todd Phillips," where "Todd Phillips" is the answer we seek?
        *   The answer is "This store is famous for being the filming location for several movies, including Old School, Gone in 60 Seconds, etc." This sentence corresponds to the input format in BERT.
        *   sentence A:
            *   Who made a movie in 2003 with a scene shot at the Los Angeles Quality Cafe?
            *   clues(x,G): “This store is famous for being the filming location for several movies, including 'Old School' and 'Gone in 60 Seconds' etc.”
        *   "Old School is an American comedy film released in 2003, directed by Todd Phillips."
        *   Entity x is referred to as "old school."
    *   This design completes the iterative part in both System 1 and System 2, connecting the two systems. This part allows System 2 to use graph structures to guide System 1 in retrieval. And through cycles, it is possible that System 2 updates the features of a certain entity's parent node or adds a new parent node, all of which may lead to the acquisition of new clues. System 1 can then use these clues again to predict and find new answer entities or next-hop entities that were not previously identified.
    *   How does System 2 depend on the results of System 1? This also divides into two parts
        *   Perform two span predictions: System 1's BERT separates the prediction start and end positions of the answer entity and the next-hop entity, using four parameter vectors to combine the word feature vectors output by BERT to predict the start and end positions of the answer entity, the start and end positions of the next-hop entity, totaling four quantities. After obtaining the answer and next-hop entities, they are added to the cognitive graph as sub-nodes of the current entity, connecting the edges.
        *   Of course, it is not enough to just connect the edges; node features are also required. Just as BERT's position 0 extracts features of the entire sentence pair, the authors use it as a node feature $sem(x,Q,clues)$ and supplement it to the diagram.
    *   This system one provides topological relationships and node features for the expansion of the graph, thereby providing resources for system two.

How to Reason on Graphs
=======================

*   This section directly employs GNN to perform spectral transformation on a directed graph to extract one layer of node features
    
    $$
    \Delta = \sigma ((AD^{-1})^T) \sigma (XW_1)) \\
    X^1 = \sigma (XW_2 + \Delta) \\
    $$
    
*   The subsequent predictions only require adding a simple network on top of the transformed node features for regression or classification.
    
*   Note that although model one extracts the answer span, both the answer span and the next-hop entity span are added as nodes to the cognitive graph, because there may be multiple answer nodes that require judgment of confidence by System Two. The reason for BERT to predict both the answer and the next-hop separately is:
    
    *   Both should have different features obtained through BERT and require independent parameter vectors to assist in updating
    *   Both are equally included in the cognitive graph, but only the next-hop node will continue to input into the system to make further predictions
*   The model's loss consists of two parts, namely the System One's span prediction (answer & next hop) loss and the System Two's answer prediction loss, both of which are relatively simple and can be directly referred to in the paper.
    

Data
====

*   Authors used the full-wiki part of HotpotQA for training and testing, with 84% of the data requiring multi-hop reasoning. Each question in the training set provided two useful entities, as well as multiple descriptions of the context and 8 irrelevant descriptions for negative sampling. During validation and testing, only the questions were provided, requiring answers and relevant descriptions of the context.
*   To construct a gold-only cognitive graph, i.e., the initialized total cognitive graph, the authors perform fuzzy matching on every sentence in the description context of all entities y and a certain entity x. If a match is found, (x, y) is added as an edge to the initialized graph

Overall Process
===============

*   Input: System One, System Two, Issue, Prediction Network, Wiki Dataset
*   Initialize the gold-only graph with entities from the problem and mark these entities as parent nodes, and add the entities found through fuzzy matching during initialization to the boundary queue (the queue to be processed)
*   Repeat the following process
    *   Pop an entity x from the boundary queue
    *   Collect clues from all ancestors of x
    *   Input the clues, issues, and descriptive context of the entity into the unified system, obtaining the cognitive graph node representation $sem(x^1,Q,clues)$
    *   If entity x is the next-hop node, then:
        *   Entity span for generating answers and next-hop
        *   For the next-hop entity span, if it exists in the Wiki database, create a new next-hop node in the diagram and establish an edge; if it is already in the diagram but has not established an edge with the current entity x, add an edge and include the node in the boundary queue
        *   For the answer entity span, nodes and edges are directly added without the need for judgment from the Wikipedia database, because the answer may not be in the database
    *   Through the second system updating node features
*   Until there are no nodes in the boundary queue, or the cognitive graph is sufficiently large
*   Through predicting the network's return results.
*   Through the above process, it can be seen that for each training data, before using the prediction network to predict the results, two systems need to interact iteratively multiple times until feature extraction is complete. The condition for stopping iteration is when the boundary queue is empty. Then, what kind of nodes will join the boundary queue? Nodes that have already been in the graph and established new edges for the next hop may bring new clues, therefore, all such nodes must be processed, allowing system two to see all clues before making predictions.

Other details
=============

*   In System One, there may not be Sentence B, i.e., there may be no descriptive context for a certain entity. In this case, we can simply obtain the node features $sem(x^1,Q,clues)$ through BERT, without predicting the answer and the next-hop entity, i.e., this node acts as a leaf node in the directed graph and no longer expands.
*   At the initialization of the cognitive graph, it is not necessary to obtain node features; only the prediction of spans is needed to construct edges
*   The author found that using the feature at position 0 of the last layer of BERT as node features was not very good, because the features of higher layers are transformed to be suitable for span prediction, so after experimentation, the author took the third-to-last layer of BERT to construct node features
*   When performing span prediction, it actually specifies a maximum span length, then predicts the top k beginning positions, and then predicts the end positions within the span maximum length
*   The author also employed negative sampling to prevent span prediction on irrelevant sentences. Specifically, it first samples irrelevant samples, sets the \[CLS\] position probability of these samples to 0, and sets the position probability of the positive samples to 1. In this way, BERT can learn the probability that sentence B is a positive sample at the \[CLS\] position. Only the topk spans selected previously will be retained if their begin position probability is greater than the \[CLS\] position probability.
*   In the process described in the pseudo-algorithm, every time the system updates the cognitive graph structure, system two runs once. In fact, the author found that it is the same effect, and more efficient, to let system one traverse all the boundary nodes first, wait until the graph no longer changes, and then let system two run multiple times. In actual implementation, this algorithm is also adopted.
*   HotpotQA includes special questions, non-traditional questions, and traditional questions. The author has constructed prediction networks for each, where special questions are regression models, and the other two types are classification models.
*   When initializing the cognitive graph, it is not only necessary to establish edges between entities and the next-hop entities, but also to mark the begin and end positions of the next-hop entities and feed them into the BERT model
*   The author also conducted an ablation study, mainly focusing on the differences in the initial entity sets, and the experimental results show that the model is relatively dependent on the quality of the initial entities

Results
=======

*   Dominating the HotpotQA leaderboard for several months before this April, until recently being surpassed by a new BERT model, but at least this model can provide a good interpretability, as shown in the three cognitive graph reasoning scenarios in the following figure ![mClr8g.png](https://s2.ax1x.com/2019/08/13/mClr8g.png) 

Conclusion
==========

*   This model can be simply regarded as an extension of GNN in NLP, with the powerful BERT used for node feature extraction. However, the difficulty of using GNN in NLP lies in the definition of edge relationships. This paper presents a very natural definition of relationships, consistent with the intuition of humans in completing question-answering tasks, and BERT not only extracts node features but also completes the construction of edges. I feel that this framework is a good way to combine black-box models and interpretable models, rather than necessarily explaining black-box models. The black box will let it do what it is good at, including feature extraction of natural language and reasoning networks, while humans can design explicit rules for adding edge relationships. Both work together, complementing each other rather than being mutually exclusive.


{% endlang_content %}

{% lang_content zh %}


# 任务

- 作者提出的框架叫CogQA，也就是用基于认知图谱的框架来解决机器阅读理解里的问答，包括一般性问答（选择实体），以及比较性问答（两个实体之间的关系）。先抛开认知图谱不说，看看这个任务是什么。
- 这个问答任务特殊的地方在于延伸到了多跳，multi-hop。多跳其实不应该是任务类型，而是指完成实体类问答任务的一种方式，即找到问题中的实体，根据这些实体在其对应介绍上下文中找到线索（clues),在这些线索里接着找实体作为下一跳(hop),下一跳的实体用于跳转到对应的介绍上下文，并在其中接着找线索，如此往复，直到多跳之后在正确的描述中找到正确的实体作为答案。一个问答，你可以用一跳的思路解决，也可以用多跳的思路解决。实际上大多数基于信息检索的问答模型就是基于一跳的，这类模型就只是比较问题和上下文，找出最相关的句子，再从这些句子中找出实体。这样做本质上是一种模式匹配，其问题在于，加入问题本身是多跳的，那么基于一跳的模式匹配可能根本找不出正确的实体，因为答案都不在候选的句子里。
- 这和人类回答问题的方式其实很类似，比方我们问发表于ACL2019的认知图谱的作者是谁，我们会先找到ACL2019和认知图谱这两个实体，再分别到其线索中找到ACL2019所有论文对应作者和认知图谱的多种含义（可能有神经科学，可能有教育学，可能有自然语言处理），再找到更多的实体和描述（不同论文作者、不同含义的解释），最终找到一个或者多个答案。人类的思路可能会直接在ACL2019的所有论文标题里找认知图谱四个字，而计算机处理起来可能是ACL2019和认知图谱两部分延伸多跳之后在某一节点合并到一个实体，即作者的名字，然后作为答案输出。
- 以上多个实体的多跳以及它们之间的拓扑关系就组成了认知图谱，一个有向图。这个有向图是可推理可解释的，不同于黑箱模型，有清晰的推理路线。那么问题就归结为：
  - 如何构造图？
  - 有了图，如何推理？
- 作者首先提出了用认知科学里的双过程解释他们的做法。

# 双过程模型

- 认知科学里的双过程是指，人类解决问题时会分两个步骤：
  - 系统一：先通过一个隐式的、无意识的、符合直觉的过程来分配注意力，检索相关信息
  - 系统二：再通过另一个显式的、有意识的、可控的过程来完成推理
  - 系统一给系统二提供资源，系统二指导系统一的检索，两者迭代进行
- 这上面两个过程，其实可以对应到人工智能里的两大流派，联结主义和符号主义。第一个过程虽然是难以解释的、通过直觉完成的，但直觉不是天生的，实际上是通过生活经验得到的隐藏知识。这部分可以对应现在用深度学习完成的黑箱模型，通过对大量数据学习得到不可解释，但是能完成目的的模型。而第二个过程需要因果关系，或者需要显式的结构来帮助推理。
- 具体到机器问答中，作者很自然的用了现有的神经网络模型来完成这两项工作：
  - 第一项需要分配注意力来检索相关信息，那么我就直接用自注意力模型，来完成实体的检索。
  - 第二项需要显式的结构，那么我就构造有向认知图谱，在认知图谱上完成推理。

# 如何构造图

- 作者使用BERT来完成系统一的工作。BERT本身就可以用作一跳机器阅读理解，在这里作者沿用了一跳的做法，输入的句子对是问题和待标记实体的句子，输出是标记概率，即每个词是实体开始位置或者结束位置的概率。但是为了实现多跳，作者做了一点改动：
  - 输入的句子对不是以问题为基本单位，而是以每个问题中的每个实体为基本单位，具体而言，每个输入句子对的A句子由问题和问题中某一实体的线索(clue)拼接而成，而句子B是关于该实体的描述上下文中的所有句子。即
    - sentence A:$[CLS]Question[SEP]clues(x,G)[SEP]$
    - sentence B:$Para(x)$
  - 那么某一实体的线索究竟是什么？**线索是该实体在认知图谱里的所有父节点的介绍上下文中，提取出该实体的那一句话**。可能有些拗口，但是这个设计是贯穿了整个系统，是其认知推理断的精髓所在，用论文给出的例子就是：
    - 问题：“谁在2003年拍了部电影，其中有一幕是在洛杉矶quality cafe拍的？”
    - 我们找到实体quality cafe，找到其介绍上下文：“......这是洛杉矶的一家咖啡店。这家店因其作为多部电影的取景地而出名，包括old school, gone in 60s等等。......”
    - 我们接着遍历这些电影名实体，接着找电影的介绍上下文，例如“old school是一部美国喜剧电影，拍于2003年，导演是todd phillips”，并且通过其他认知推理得到这个“todd phillips”就是正确答案，那么，这个导演实体的线索是什么？我们需要什么样的线索作为补充输入来得到“old school是一部美国喜剧电影，拍于2003年，导演是todd phillips”这句话中“todd phillips”就是我们想要的答案？
    - 答案就是“这家店因其作为多部电影的取景地而出名，包括old school, gone in 60s等等。”这句话。对应成BERT里的输入格式就是
    - sentence A:
      - Question：“谁在2003年拍了部电影，其中有一幕是在洛杉矶quality cafe拍的？”
      - clues(x,G)：“这家店因其作为多部电影的取景地而出名，包括old school, gone in 60s等等。”
    - sentence B:“old school是一部美国喜剧电影，拍于2003年，导演是todd phillips”
    - 其中实体x是“old school”
  - 这个设计完成了系统一和系统二中的迭代部分，将两个系统连接了起来。这部分是让系统二利用图结构来指导系统一检索。并且通过循环往复，可能系统二更新了某一实体的父节点的特征，或者添加了新的父节点，这些都可能会导致有新的线索获得，系统一可以再次把这些线索拿来预测，找出之前没有找出的新的答案实体或者下一跳实体。
  - 那么系统二如何依赖系统一的结果呢？这里也分为两个部分
    - 做两个span prediction：系统一的BERT将答案实体和下一跳实体的预测起始结束位置分开，用四个参数向量来分别结合BERT输出的词特征向量来预测答案实体、下一跳实体的预测开始、结束位置共四个量。获得了答案和下一跳实体之后，将其加入认知图谱当中，作为当前实体的子节点，连上边。
    - 单单连上边当然不够，还需要节点特征。刚好BERT的位置0是提取整个句子对的特征，作者就将其作为节点特征$sem(x,Q,clues)$补充到图中。
  - 这样系统一就为图的扩展提供了拓扑关系和节点特征，从而为系统二提供了资源。

# 如何在图上推理

- 这一部分就直接使用了GNN，在有向图上进行谱变换提取一层节点特征
  
  $$
  \Delta = \sigma ((AD^{-1})^T) \sigma (XW_1)) \\
X^1 = \sigma (XW_2 + \Delta) \\
  $$
- 之后的预测也只需要在变换后的节点特征上接一层简单网络来做回归或者分类就好了。
- 注意虽然模型一提取了答案的span，但是答案span和下一跳实体span都作为节点加入到认知图谱当中，因为可能有多个答案节点，需要经过系统二来判断置信度，而需要BERT分别预测答案和下一跳的理由是：
  - 两者通过BERT得到的特征应该不同，需要独立的参数向量来辅助更新
  - 两者虽然是同等的加入认知图谱当中，但是只有下一跳节点会接着输入系统一来继续预测
- 模型的损失包含两部分，分别是系统一的span prediction(answer & next hop) loss和系统二的answer prediction loss，都比较简单，可以直接看论文。

# 数据

- 作者使用了HotpotQA的full-wiki部分来做训练测试，84%的数据需要多跳推理。训练集中每个问题提供了两个有用的实体，以及多个描述上下文和8个不相关描述上下文用于负采样。验证和测试时只有问题，需要给出答案和相关的描述上下文。
- 为了构造gold-only认知图谱，即初始化的总的认知图谱，作者将所有的实体y和某一实体x的描述上下文中每一句做模糊匹配，匹配上了就将(x,y)作为一条边加入初始化的图谱中

# 总体流程

- 输入：系统一、系统二、问题、预测网络、维基数据集
- 用问题里的实体初始化构造gold-only图谱，并把这些实体标记为父节点，把初始化中模糊匹配找到的实体加入边界队列（待处理队列）
- 重复以下过程
  - 从边界队列中弹出一个实体x
  - 从x的所有父节点那收集线索
  - 将该实体的线索、问题、该实体的描述上下文输入系统一，得到认知图谱节点表示$sem(x^1,Q,clues)$
  - 假如实体x是下一跳节点，那么：
    - 生成答案和下一跳的实体span
    - 对于下一跳实体span，假如其在维基数据库当中，就在图中创建新的下一跳节点并建立边；假如已经在图中，但是没有和当前实体x建立边，那就添加一条边，并把该节点加入边界队列
    - 对于答案实体span，直接加节点和边，而不需要经过维基数据库的判断，因为答案有可能不在数据库中
  - 通过系统二更新节点特征
- 直到边界队列中没有节点，或者认知图谱足够大
- 通过预测网络返回结果。
- 通过以上流程可以看到，对每一条训练数据，在使用预测网络预测结果之前，需要两个系统交互迭代多次直到特征提取完全。迭代的条件是边界队列为空时停止，那么什么样的节点会加入边界队列？**已经在图中且建立了新的边的下一跳节点**，这一类节点可能带来新的线索，因此必须把这类节点都处理完，让系统二看到所有线索，之后才能做预测。

# 其他细节

- 在系统一当中可能没有sentence B，即没有某个实体的描述上下文，这时我们可以仅仅通过BERT得到节点特征$sem(x^1,Q,clues)$，而不预测答案和下一跳实体，即这个节点就作为有向图中的叶子节点，不再扩展。
- 在初始化认知图谱时，不需要得到节点特征，仅仅预测span来构建边
- 作者发现使用BERT的最后一层的位置0的特征作为节点特征不太好，因为高层的特征被转换成适用于span prediction，因此作者试验之后取BERT的倒数第三层来构建节点特征
- 在做span prediction的时候，实际上是规定了一个span maximum length，然后预测top k个begin position，然后在span maximum length内预测end position
- 作者还做了负采样来防止在无关句子上做span prediction，具体做法是先负采样出不相关的样本，将这些样本的[CLS]位置概率设为0，而正样本的该位置概率设为1。这样BERT就能在[CLS]位置学到这个sentence B是正样本的概率。之前选出的topk个span，只有begin position probability大于[CLS]位置概率才会被保留下来
- 在伪算法描述的流程里，每次系统一更新了认知图谱结构，系统二就运行一次，实际上作者发现让系统一先把所有的边界节点遍历完，等图不再改变，再让系统二运行多次是一样的效果，而且效率更高。实际实现上也是采用这种算法。
- HotpotQA包含特殊问题、非传统问题和传统问题，作者分别构建了预测网络，其中特殊问题是回归模型，其余两类是分类模型。
- 初始化认知图谱的时候，不仅仅需要建立实体与下一跳实体之间的边，下一跳实体的begin和end位置也要标出来，feed给BERT模型
- 作者还做了ablation study，主要是初始化的实体集不同，可以通过实验结果看出该模型还是比较依赖初始化的实体质量

# 结果

- 在今年4月份以前一直霸榜HotpotQA几个月，直到最近被一个新的BERT模型打破，但至少该模型能够提供一个很好的解释性，例如下图所示的三种认知图谱推理情况
  ![mClr8g.png](https://s2.ax1x.com/2019/08/13/mClr8g.png)

# 结论

- 这个模型可以简单的看成是GNN在NLP的扩展，只不过用了强大的BERT做节点特征提取，但是GNN用于NLP的难点在于边关系的定义。本文对于关系的定义非常自然，和人类完成问答任务的直觉保持一致，并且BERT也不仅仅是提取节点特征，还完成了边的构建。这样的框架我感觉是很好的将黑盒模型和可解释模型结合起来，而不是一定要解释黑盒模型。黑盒将让他做黑盒擅长的部分，包括自然语言和推理网络的特征提取，而人类可以设计显式的边关系添加规则，两者合作，互补而不是互斥。


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