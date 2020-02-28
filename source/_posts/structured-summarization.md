---
title: STRUCTURED NEURAL SUMMARIZATION 阅读笔记
date: 2020-02-28 10:22:26
categories: 自然语言处理
tags:
  - graph neural network
  - deep learning
  - summarization
  -	natural language processing
mathjax: true
html: true
---

STRUCTURED NEURAL SUMMARIZATION 阅读笔记

<!--more-->

# Intuition
-	将GNN引入seq2seq的一个很自然的想法就是先用sequence encoder得到token representations，然后再构建token之间的关系，建图，将图和token representations送入GNN，得到新的token表示。
-	问题在于，对于摘要，直觉的想法是以句子为节点，构建句子之间的关系图，这样最后得到的是句子的表示，不能到词级别的细粒度，这样的话attention based decoder就不太好做。
-	本文作者的想法就很粗暴，干脆就做词级别的GNN，一篇文章900个词，就构建900个节点的图，而边的关系是异构的，分三种：
	-	一句里的所有词连向一个额外添加的句子节点、一个实体里的所有词连向一个额外添加的实体节点，这类边都叫做IN
	-	前一个词连后一个词，前一句连后一句，这类边叫NEXT
	-	指代消解，这部分叫做REF
	整体如下图所示：
	![1Toe6P.png](https://s2.ax1x.com/2020/02/11/1Toe6P.png)

# GGNN
-	作者使用的GNN是Gated Graph Neural Network。原论文见：GATED GRAPH SEQUENCE NEURAL NETWORKS
$$
\begin{aligned} \mathbf{h}_{v}^{(1)} &=\left[\boldsymbol{x}_{v}^{\top}, \mathbf{0}\right]^{\top} \\ \mathbf{a}_{v}^{(t)} &=\mathbf{A}_{v:}^{\top}\left[\mathbf{h}_{1}^{(t-1) \top} \ldots \mathbf{h}_{|\mathcal{V}|}^{(t-1) \top}\right]^{\top}+\mathbf{b} \\ \mathbf{z}_{v}^{t} &=\sigma\left(\mathbf{W}^{z} \mathbf{a}_{v}^{(t)}+\mathbf{U}^{z} \mathbf{h}_{v}^{(t-1)}\right) \end{aligned} \\ 
\begin{aligned} \mathbf{r}_{v}^{t} &=\sigma\left(\mathbf{W}^{r} \mathbf{a}_{v}^{(t)}+\mathbf{U}^{r} \mathbf{h}_{v}^{(t-1)}\right) \\ \widetilde{\mathbf{h}_{v}^{(t)}} &=\tanh \left(\mathbf{W} \mathbf{a}_{v}^{(t)}+\mathbf{U}\left(\mathbf{r}_{v}^{t} \odot \mathbf{h}_{v}^{(t-1)}\right)\right) \\ \mathbf{h}_{v}^{(t)} &=\left(1-\mathbf{z}_{v}^{t}\right) \odot \mathbf{h}_{v}^{(t-1)}+\mathbf{z}_{v}^{t} \odot \widetilde{\mathbf{h}_{v}^{(t)}} \end{aligned} \\
$$
-	GGNN发布于2015年，在2009年的GNN模型上改进。
-	原始的GNN模型相当于用图的拓扑关系，在多层线性网络中Mask掉部分边，节点在每一层的表示通过上一层中相邻的节点线性变换而来，即propagation，最后一层linear做output输出节点标签
![3082wD.png](https://s2.ax1x.com/2020/02/27/3082wD.png)
-	GGNN考虑了边的有向和异构。其邻接矩阵A如上图所示，是一个$\mathbf{A} \in \mathbb{R}^{D|\mathcal{V}| \times 2 D|\mathcal{V}|}$的线性层，两倍的宽代表双向两个输出频道，输入节点的表示矩阵$\mathbb{R}^{D|\mathcal{V}|}$，这里的$A$包含了参数，依赖于边的类型和方向，相当于这个包含了邻接信息的线性层也是通过embedding lookup得到的。接下来就是类似于GRU的更新，$z$和$r$分别是更新门和重置门。所以公式含义如下
	-	1:初始化节点embedding，根据具体任务给每个节点补上手工特征，并Padding到相同长度
	-	2:通过包含了邻接信息的线性层，得到propagate之后的信息
	-	3:根据上一层状态和propagate信息计算更新门
	-	4:根据上一层状态和propagate信息计算重置门
	-	5,6:同GRU
-	output部分，简单的线性层作用于每一个节点就可以做节点级别的任务，如果要获得整张图的表示，可以用一个门控机制来获取（原文表述为attention）：
	$$
	\mathbf{h}_{\mathcal{G}}=\tanh \left(\sum_{v \in \mathcal{V}} \sigma\left(i\left(\mathbf{h}_{v}^{(T)}, \boldsymbol{x}_{v}\right)\right) \odot \tanh \left(j\left(\mathbf{h}_{v}^{(T)}, \boldsymbol{x}_{v}\right)\right)\right)
	$$

# GGSNN
-	门控的GNN还可以扩展为sequence output，即GATED GRAPH SEQUENCE NEURAL NETWORK。
![3BkTun.png](https://s2.ax1x.com/2020/02/28/3BkTun.png)
-	如上图所示，一般的seq2seq，需要把输入的seq编码，之后再逐步解码，但是再graph当中，一步的graph就已经包含了所有seq token信息，多层只是深度层次上的叠加，而不是时序的层次，因此我们可以在任意的深度层次开始解码，类似于CRF，如图所示：$o$为输出，$X^k$为第k步输出时节点的input embedding matrix，$H^{k,t}$代表第k步输出，同时整个输入的节点embedding在深度上传递了t步时，节点的hidden state matrix。类似于转移与发射矩阵，作者分别用了两个GGNN$F_o,F_x$来完成hidden state的转移和发射。两者可以共享propagation部分的参数。虽然只写了$F_x$将$H$转移到$X$，但实际上类似于LSTM，$X^{k+1}$同时还由$X^k$决定：
$$
\boldsymbol{x}_{v}^{(k+1)}=\sigma\left(j\left(\mathbf{h}_{v}^{(k, T)}, \boldsymbol{x}_{v}^{(k)}\right)\right)
$$
-	同样，也可以不需要从$H$到$X$的转移，直接输入每一解码步的$X$，类似于teacher forcing
-	论文里的实验都是状态空间比较小，不同于文本任务。直接看STRUCTURED NEURAL SUMMARIZATION里的用法

# Sequence GNNs
-	作者将GGNN引入编码端，相当于传统的seq2seq encoder最后用GNN补充了一次编码，但是encoder的输出不变，decoder不变（抛弃了GGSNN的decoder设计）
-	首先作者用更加清晰的语言描述了GGNN，每一步包含propagation 与 update
	-	propagation：$\boldsymbol{m}_{v}^{(i)}=g\left(\left\{f_{k}\left(\boldsymbol{h}_{u}^{(i)}\right) | \text { there is an edge of type } k \text { from } u \text { to } v\right\}\right.)$，即用边相关的线性变换收集邻域节点信息求和，其中$f$是线性层，$g$是求和
	-	update：$\boldsymbol{h}_{v}^{(i+1)}=\operatorname{GRU}\left(\boldsymbol{m}_{v}^{(i)}, \boldsymbol{h}_{v}^{(i)}\right)$
-	seq2seq中的encoder至少要提供两点信息：token representation 和 context representation。token级别的已经拿到了，即GNN之后的节点 embedding，context级别即图的表示，这里作者除了沿用GGNN里门控算权重求和各节点之外，还拼接了输入图之前、RNN编码之后的hidden state，看代码貌似是把所有节点输入图前后的hidden state拼接起来，作为最终的节点embedding；把RNN的输出直接和图embedding表示拼接起来再过一个线性层。这里注意RNN的输出实际上是对图（整个序列）的一个表示，和graph embedding是同一级别的，所以直接拼接：
	$$
	\left[\mathbf{e}_{1}^{\prime} \ldots \mathbf{e}_{N}^{\prime}\right]=\operatorname{GNN}\left(\left(S,\left[R_{1} \ldots R_{K}\right],\left[\mathbf{e}_{1} \ldots \mathbf{e}_{N}\right]\right)\right) \\
	\sigma\left(w\left(\boldsymbol{h}_{v}^{(T)}\right)\right) \in[0,1] \\
	\hat{\mathbf{e}}=\sum_{1<i<N} \sigma\left(w\left(\mathbf{e}_{i}^{\prime}\right)\right) \cdot \aleph\left(\mathbf{e}_{i}^{\prime}\right) \\
	Embedding_{graph} = W \cdot(\mathbf{e} \hat{\mathbf{e}}) \\
	$$
-	在实际工程实现中，不同大小的图打包成batch不方便，作者也是采用了两个trick
	-	GNN的常规做法：把小图拼接成有多个连接子图的大图作为一个batch
	-	由于copy和attention机制需要在整个输入序列上计算权重，拼接成大图之后作者也保留了每个节点在小图当中的index，然后通过tensorflow的unsorted segment *操作符（即对不同长度的段分别做操作），可以完成一个efficient and numerically stable softmax over the variable number of representations of the nodes of each graph.
-	作者只用了简单的LSTM的encoder和decoder配置，基本在pointer generator的代码上做改动。GNN叠加八层
-	最后的结果并没有超过pointer generator，但是引入pointer机制后的ablation比较明显，如下图，
	![1TTHPJ.png](https://s2.ax1x.com/2020/02/11/1TTHPJ.png)
-	作者也没有对结果做太多分析，因为论文做了三个数据集，其余两个是代码摘要，有比较自然的结构化数据，因此表现很好，在CNNDM这种纯自然语言数据集上表现并不是特别亮眼。
-	但是在消融实验中值得注意的是即便不添加指代信息，仅仅是让GNN处理句子结构，表现也比LSTM要好。