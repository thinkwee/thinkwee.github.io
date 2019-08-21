---
title: ACL/NAACl 2019 自动摘要相关论文选读
date: 2019-08-15 10:51:37
categories: 自然语言处理
tags:
  - acl
  - deep learning
  - summarization
  -	natural language processing
mathjax: true
html: true
---

ACL/NAACL 2019 自动摘要相关论文选读
<!--more-->

# Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization
-	这和我们组其他人做的很像，用DPPs处理抽取式文摘
-	在上文的文摘指标论文中也提到了，搞文摘，尤其是抽取式文摘，就是三个词，qd，qd，还是qd!
	-	q:quality，哪个句子重要，需要被抽出来作为文摘。这一步是特征的构造
	-	d:diversity,抽取出来的句子之间不能冗余，不能老抽一样的句子。重要的一样的句子多了，也就变得不重要了。这一步是抽样方法的构造
-	DPPs(Determinantal Point Processes)就是一种抽样方法，保证抽出来的句子重要（根据特征计算好的重要性数值），而且不重复。本文作者就是思路非常清晰：我要改进抽取式文摘中的DPPs，怎么改进？DPPs依赖句子之间的相似度来避免抽重复的句子，那我就改进不管DPPs怎么改，直接改相似度的计算，所以问题就换到了一个非常成熟的领域：语义相似度计算。
-	接下来就随便套网络来做语义相似度计算就行了。作者比较新潮，用了胶囊网络，这个网络本来是提出用于解决计算机视觉中物体的位置相对变化问题的，作者认为可以将其泛化到提取底层语义特征的空间与方位信息，这里我对胶囊网络及其在NLP的应用不太了解，但是就作者给出的对比实验来说，改进其实也就1个点，而整个DPP相比之前最好系统（2009）也就好2个点，感觉还是有点刻意为之。
-	另外作者给出的网络是真的复杂，不是原理复杂，而是用了很多组件，包括：
	-	三四五六七大小卷积核的CNN用于提取底层特征
	-	胶囊网络提取高层特征，用到了近年来的参数共享和路由等技巧
	-	还是用了one-hot向量，即一个词是否存在在某一个句子里
	-	各种特征的融合，包括内积，绝对差，再和所有独立的特征全部拼接到一起，预测两个句子的相似度
	-	而且相似度还只是一部分目标，另外作者还用了LSTM来重构两个句子，将重构损失加入最终的总损失
![ezCge0.png](https://s2.ax1x.com/2019/08/12/ezCge0.png)
-	粗看图片绝对以为这是一篇CV的work
-	作者好歹是把最近能用的技巧都用上了，做了一个集大成的网络，可能学术上看没那么简洁优美，但是就工业上来说很多这样的网络集成操作就是很work
-	另外虽然是抽取式摘要，但作者的工作是完全监督，因此还需要构造数据集
	-	从生成式摘要数据集中构造有监督抽取式摘要数据集
	-	从生成式摘要数据集中构造有监督句子相似度计算据集
	-	这种构造也一定程度上限制了其泛化能力
-	作者的出发点其实非常好：因为传统的相似度停留在词的粒度，没有深入到语义特征，因此构造网络提取特征的方向没错，只不过稍显复杂，而且由于仅仅改进了抽取式文摘中某一种抽样方法的相似度计算中的句子特征提取部分，对整体的影响并没有特别大，最后的结果虽然吊打了很多传统方法，但是相比传统最佳方法并没有提高多少，相比纯DPPs更是只有1个点左右的提高。

# STRASS: A Light and Effective Method for Extractive Summarization Based on Sentence Embeddings
![mkkJHO.png](https://s2.ax1x.com/2019/08/14/mkkJHO.png)
-	又是一篇用监督方法做抽取式文摘的，从标题就可以大致猜出来内容，基于embedding，还要light and effective，那最简单的目标就是把摘要的embedding和gold embedding保持一致。
-	困难的地方在于这是抽取式的，是离散的，因此需要一个流程把抽取、embedding、比较打分三个部分统一起来，软化使其可导，可以端到端训练，作者给出的是四个步骤：
	-	将文档embedding映射到比较空间
	-	抽句子组成摘要
	-	去近似抽出的摘要的embedding
	-	和gold summary的embedding比较
-	首先给定一篇文档，直接用sent2vec获得doc embedding和文档里每一句的sentence embedding
-	之后只用一层全连接作为映射函数f(d)，这里作者给出了第一个假设：抽取的摘要句子应该和文档具有相似度：
	$$
	sel(s,d,S,t) = sigmoid (ncos^{+}(s,f(d),S)-t)
	$$
-	其中s是句子embedding，S代表句子集，t为阈值。 sel代表select，即选择这个句子组成摘要的置信度，这个式子说明选出句子的embedding和文档embedding之间的相似度应该大于阈值t，且使用sigmoid做了软化，将{0,1}软化为[0,1]
-	之后进一步软化，作者并不根据分数选出句子组成文摘，而是根据分数直接近似文摘的embedding
	$$
	app(d,S,t) = \sum _{s \in S} s * nb_w(s) * sel(s,d,S,t)
	$$
-	其中nb_w是number of words，即使用每个句子的字数和select score对所有的句子embedding加权求和得到generated summary的embedding
-	最后一步，和gold summary比较embedding相似度计算损失，这里作者加入了一个正则项，希望提出来的摘要压缩比越高越好，这里我感觉是上一步一系列软化操作带来的补偿，因为没有选择句子，而是对所有句子加权，因此需要正则强迫模型放弃一些句子：
	$$
	loss = \lambda * \frac{nb_w(gen_sum)}{nb_w(d)} + (1-\lambda) * cos_{sim}(app(d,S,t),ref_{sum})
	$$
-	这里有一个问题，gold summary的embedding是怎么得到的？
-	另外为了保证能够对所有文档使用同一个阈值，作者还对cosine相似度计算的结果做了归一化
-	从结果来看其实ROUGE还不如生成式的方法好，当然一方面原因是因为数据集本来就是生成式的，但是强在简单，快，而且用监督的方法做抽取式也不用考虑redundency的问题。

# A Robust Abstractive System for Cross-Lingual Summarization
-	其实一句话就能概括这篇论文：做多语言生成式文摘，别人是先摘要再翻译，这篇文章是先翻译再生成摘要
-	均是用已有框架实现
	-	翻译：Marian: Fast Neural Machine Translation in C++
	-	摘要：pointer-generator
-	作者其实有充足的监督数据，之前以为是多语言、小语料的摘要，或者是不借助于翻译的摘要，能够挖掘多语种共有的摘要特征
-	但是这篇论文确实实现了robust：一般要做robust就是引入噪声，本文正好用了回译引入噪声：先把英语翻译成小语种，翻译回来，在这样的bad english document上训练生成式摘要模型，使得模型对噪声更加鲁棒，最后的结果也是提高了许多，并且对没有训练过的阿拉伯语也取得了较好的效果，说明不同的人翻译各有各的正确，而机器翻译的错误总是相似的。

# Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction
![mtX0eS.png](https://s2.ax1x.com/2019/08/21/mtX0eS.png)
-	这篇论文来自号称全球最大电信公司NTT的智能实验室，提出了一个多任务模型：阅读理解+自动摘要
-	论文做了很多实验和分析，并且详细分析了在何种情况下他们的module works，对于摘要部分也利用了许多近年来提出的技巧，而不是简单的拼凑。
-	这篇论文同样也是在HotpotQA数据集上做，和CogQA那一篇一样，但那一篇用的是full wiki setting，即没有gold evidence，而这篇需要gold evidence因此用了HotpotQA 的distractor setting。
-	对于HotpotQA的distractor setting，监督信号有两部分：answer和evidence，输入有两部分:queryh和context，其中evidence是context当中的句子。作者沿用了HotpotQA论文里的baseline:Simple and effective multi-paragraph reading comprehension，及上图中Query-Focused Extractor以外的部分。基本思想就是将query和context结合，加上一堆FC,attention，BiRNN提取特征，最终在answer部分输出一个answer type的分类和answer span的sequence labelling，而在evidence部分直接接BiRNN输出的结果对每个句子做二分类。
![mtXczq.png](https://s2.ax1x.com/2019/08/21/mtXczq.png)
-	作者将evidence这边的监督任务细化为一个query based的summarization，就在BiRNN后面加了一个模块，称之为Query-Focused Extractor(QFE)，相比原始的简单二分类，QFE强调了evidence应该是在query条件下从context中抽取出来的summary，因满足：
	-	summary内的句子之间应该不能冗余
	-	summary内不同句子应该有着query上不同的注意力
-	针对第一点，作者在QFE内设计了一个RNN，使得在生成注意力乃至抽取摘要时都能注意到之前已经抽取出来的句子,其中RNN的时间步定义为每一次抽取一个句子，输入即某一时间步抽取出的句子的vector
-	针对第二点，作者在QFE内增加了一个针对query的注意力，加权之后的query向量称为glimpse，注意这里是QA context对QA query的注意力，attention里的key和value都是QA的query，而attention里的query不是直接拿整个QA context，而是RNN的输出，即已经抽取出的句子集经过RNN编码的context，这样的设计也是符合直觉的。
-	在RNN编码已抽取句子、注意力加权query形成glimpse向量之后，QFE拿到这两部分向量，结合每一个context未抽取句子的向量来输出每一个句子被抽取的概率，并选择最大概率的句子加入已抽取句子集合，然后接着循环计算RNN和glimpse。整个系统的依赖关系在上图中展示的很清晰。
-	由于gold evidence的句子数目不固定，作者采用添加一个EOE的dummy sentence的方法来动态抽取，当抽取到EOE时，模型就不再接着抽取句子。
-	在训练时，evidence这边的损失函数为：
	$$
	L_E = - \sum _{t=1}^{|E|} \log (max _{i \in E \ E^{t-1}} Pr(i;E^{t-1})) + \sum _i min(c_i^t, \alpha _i^t)
	$$
	这里$E$是gold evidence的句子集合，$E^t$是QFE抽取出来的句子集合，$\alpha _i^t$是t时间步query里第i个词的注意力，这里的时间步和前文一致，是抽句子的时间步。而$c^t = \sum _{i=1}^{t-1} \alpha ^i$是coverage向量。
	损失的前半部分指的是gold evidence的负对数似然损失，依次在抽取句子集合里找拥有最大QFE预测概率的gold sentence，算损失，然后排除这个句子接着找下一个最大的，直到gold sentence找完或者抽取句子集合里找不到gold sentence，后半部分是coverage机制的一个正则化应用，保证挑出来计算损失的句子不会在query上拥有过于重复（集中）的注意力。
-	作者在HotpotQA和文本蕴含数据集FEVER上做了结果，evidence部分的指标远好于baseline，而answer部分的指标也有较大提升，但不如evidence部分明显，且与BERT模型部分相比还差一点，在full wiki setting的测试集上也被CogQA全面超过，这里作者说存在dataset shift问题。但至少本文仅仅在baseline上添加了一个小模块，就获得了answer部分的8个点的提升，说明精心设计的summarization部分在多任务学习中确实帮助到了answer的选取。 