---
title: 论文阅读笔记2018下半年
date: 2018-07-03 15:18:52
tags:
  - abstractive summarization
  - math
  - machinelearning
  -	theory
  -	nlp
categories:
  - 机器学习
author: Thinkwee
mathjax: true
html: true
password: kengbi
---

下半年待填坑列表
主要关注图表示学习、文本生成和文本摘要、注意力机制、一些数学基础和模型的深入理解。
好像又有一大批贝叶斯网络的坑要来了。
读论文是不可能读完的，这辈子都不可能读完的。
有时间再补上阅读笔记。

<!--more--> 

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180804/E1lIhI4li1.jpg?imageslim)

# TODO列表
- [ ]	self-critical sequence training for image captioning
- [ ]	recurrent highway networks
- [ ]	recurrent convolutional neural networks for scene labeling
- [ ]	neural word embedding as implicit matrix factorization
- [ ]	hierarchically-attentive rnn for album summarization and storytelling
- [ ]	learning phrase representations using rnn encoder-decoder for statistical machine translation
- [x]	neural headline generation with sentence-wise optimization
- [x]	lda数学八卦
- [ ]	incorporating copying mechanism in sequence-to-sequence learning
- [ ]	generating sentences from a continuous space
- [x]	generating news headlines with recurrent neural networks
- [ ]	calculus on computational graphs: backpropagation
- [x]	convolutional sequence to sequence learning
- [ ]	cutting-off redundant repeating generations for neural abstractive summarization
- [ ]	attention and augmented recurrent neural networks
- [x]	attention is all you need
- [ ]	admissible stopping in viterbi beam search for unit selection in concatenative speech synthesis
- [ ]	abstractive document summarization with a graph-based attentional neural model
- [ ]	addressing the rare word problem in neural machine translation
- [ ]	abstractive sentence summarization with attentive recurrent neural networks
- [x]	unsupervised machine translation using monolingual corpora only
- [x]	the nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies
- [ ] 	neural architecture search with reinforcement learning
- [ ] 	an introduction to conditional random fields

# 一些文摘方面的SOTA
-	结果从上到下依次是ROUGE-1,ROUGE-2,ROUGE-L
-	强调是pyrouge是因为相比原作者用Perl写的系统，pyrouge测试值偏高

<table style="border-collapse:collapse;border-spacing:0;border-color:#ccc" class="tg"><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#f0f0f0;text-align:center">论文</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#f0f0f0;text-align:center">日期</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#f0f0f0;text-align:center">数据集</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#f0f0f0;text-align:center">结果</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:bold;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#f0f0f0;text-align:center">是否用pyrouge</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2">Deep Communicating Agents for Abstractive Summarization</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2">2018.8</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">41.69<br>19.47<br>37.92</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2">是</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">NYT</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">48.08<br>31.19<br>42.33</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Document Modeling with External Attention for Sentence Extraction</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2018.7</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN news highlights</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">54.2<br>21.6<br>48.1</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">是</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Multi-Reward Reinforced Summarization with Saliency and Entailment</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2018.5</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">40.43<br>18.00<br>37.10</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center"></td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Ranking Sentences for Extractive Summarization with Reinforcement Learning</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2018.4</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">40.0<br>18.2<br>36.6</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">是</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2">A Deep Reinforced Model for Abstractive Summarization</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2">2017.11</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">41.16<br>15.82<br>39.08</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center" rowspan="2"></td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">NYT</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">47.22<br>30.72<br>43.27</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Abstractive Document Summarization with a Graph-Based Attentional Neural Model</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2017.8</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">38.1<br>13.9<br>34.0</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center"></td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Get To The Point: Summarization with Pointer-Generator Networks</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2017.4</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">39.53<br>17.82<br>36.38</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">是</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2016.11</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">39.6<br>16.2<br>35.3</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center"></td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Distraction-Based Neural Networks for Document Summarization</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2016.1</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">CNN/DM</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">27.1<br>8.2<br>18.7</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center"></td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">Neural Summarization by Extracting Sentences and Words</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">2016.7</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">500 DailyMail</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center">21.2<br>8.3<br>12.0</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:inherit;color:#333;background-color:#fff;text-align:center"></td></tr></table>

# Distraction-Based Neural Networks for Document Summarization
-	不仅仅使用注意力机制，还使用注意力分散机制，来更好地捕捉文档的整体含义。实验证明这种机制对于输入为长文本时尤其有效。
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/H6IBk7f0bk.png?imageslim)
-	在编码器和解码器之间引入控制层，实现注意力集中和注意力分散，用两层GRU构成：
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/LjAaidg72e.png?imageslim)
-	这个控制层捕捉$s_t^{'}$和$c_t$之间的联系，前者编码了当前及之前的输出信息，后者编码经过了注意力集中和注意力分散处理的当前输入，而$e(y_{t-1})$是上一次输入的embedding。
-	三种注意力分散模型
	-	M1：计算c_t用于控制层，在输入上做分散，其中c_t是普通的注意力编码出来的上下文c_t^'，减去了历史上下文得到，类似coverage机制
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/j3ijmk54kj.png?imageslim)
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/3dCkK5ie7F.png?imageslim)
	-	M2：在注意力权重上做分散，类似的，也是减去历史注意力，再做归一化
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/Ei4m9iKA9D.png?imageslim)
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/JbHf12J8gG.png?imageslim)
	-	M3：在解码端做分散，计算当前的$c_t$，$s_t$，$\alpha _t$和历史的$c_t$，$s_t$，$\alpha _t$之间的距离，和输出概率一起作为解码时束搜索所依赖的得分。
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/IJ3Li6AA3m.png?imageslim)
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/HIfge59Ji8.png?imageslim)

# Document Modeling with External Attention for Sentence Extraction
-	构造了一个抽取式文摘模型，由分层文档编码器和基于外部信息注意力的抽取器组成。
在文摘任务中，外部信息是图片配字和文档标题。
-	通过隐性的估计每个句子与文档的局部和全局关联性，显性的考虑外部信息，来决定每句话是否应该加入文摘。

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/I5GHHELAdb.png?imageslim)

-	句子级编码器：如图所示，使用CNN编码，每个句子用大小为2和4的卷积核各三个，卷积出来的向量做maxpooling最后生成一个值，因此最后生成的向量为6维。
-	文档级编码器：将一个文档的句子6维向量依次输入LSTM进行编码。
-	句子抽取器：由带注意力机制的LSTM构成，与一般的生成式seq2seq不同，句子的编码不仅作为seq2seq中的编码输入，也作为解码输入，且一个是逆序一个是正序。抽取器依赖编码端输入$s_t$，解码端的上一时间步状态$h_t$，以及进行了注意力加权的外部信息$h_t^{'}$：

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/k8hHm3A3CL.png?imageslim)

# Get To The Point: Summarization with Pointer-Generator Networks
-	介绍了两种机制，Pointer-Generator解决OOV问题，coverage解决重复词问题
-	Pointer-Generator:通过context，当前timestep的decoder状态及输入学习到指针概率

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/k707Ael05k.png?imageslim)
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/5idDIiI9ik.png?imageslim)

-	指针概率指示是否应该正常生成，还是从输入里按照当前的注意力分布采样一个词汇复制过来，在上式中，如果当前的label是OOV，则左边部分为0，最大化右边使得注意力分布能够指示该复制的词的位置；如果label是生成的新词（原文中没有），则右边部分为0，最大化左边即正常的用decoder生成词。综合起来学习正确的指针概率。

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/JhmIg667i7.png?imageslim)

-	Coverage:使用coverage机制来修改注意力，使得在之前timestep获得了较多注意力的词语之后获得较少注意力
-	普通注意力计算

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/gjFeh5GHm9.png?imageslim)

-	维护一个coverage向量，表示每个词在此之前获得了多少注意力:

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/deabGEiC00.png?imageslim)

-	然后用其修正注意力的生成，使得注意力生成考虑了之前的注意力累积

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/099iBDA88m.png?imageslim)

-	并在损失函数里加上coverage损失

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/36LFjHbch5.png?imageslim)

-	使用min的含义是，我们只惩罚每一个attention和coverage分布重叠的部分，也就是coverage大的，如果attention也大，那covloss就大；coverage小的，不管attention如何，covloss都小

# SummaRuNNer A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/F08I4Ll22f.png?imageslim)

-	用RNN做抽取式文摘,可以用可视化展示模型决策过程，并且使用了一种端到端的训练方法
-	将抽取视为句子分类任务，对每个句子按原文顺序依次访问，决定是否加入文摘，且这个决策考虑了之前决策的结果。
-	用一层双向GRU在词级别上编码，再用一层双向GRU在句子级别上编码，两层输出的编码都经过了正反拼接和均值pooling

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/a77EbCH3Eg.png?imageslim)

-	其中d是整篇文档的编码，$h_j^f$和$h_j^b$代表句子经过GRU的正反向编码
-	之后根据整篇文档的编码、句子的编码以及文摘在当前句子位置的动态表示来训练一个神经网络做二分类，决定每个句子是否应该加入文摘：

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/eG9Ka8G1lA.png?imageslim)

-	其中sj为到j位置为止已经产生的文摘的表示，用每个句子的二分类概率对之前句子的编码加权求和得到：

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/f2Fg71Hjdg.png?imageslim)

-	第一行：参数为当前句子编码，表示当前句子的内容
-	第二行：参数为文档编码和句子编码，表示当前句子对文档的显著性
-	第三行：参数为句子编码和文摘动态编码，表示当前句对已产生文摘的冗余。（We squash the summary representation using the tanh operation so that the magnitude of summary remains the same for all time-steps.）
-	第四行和第五行：考虑了句子在文档中的相对位置和绝对位置。（The absolute position denotes the actual sentence number, whereas the relative position refers to a quantized representation that divides each document into a fixed number of segments and computes the segment ID of a given sentence.）
-	最后对整个模型做最大似然估计:

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/KBeh4fEfIh.png?imageslim)

-	作者将这种抽取式方法应用在生成式文摘语料上，也就是如何用生成式的文摘为原文中每个句子打上二分类的label。作者认为label为1的句子子集应该和生成式文摘ROUGE值最大，但是找出所有的子集太费时，就用了一种贪心的方法：一句一句将句子加入子集，如果剩下的句子没有一个能使当前子集ROUGE值上升，就不加了。这样将生成式文摘语料转换为抽取式文摘语料。
-	还有一种方式，直接在生成式文摘语料上做训练，将上面提到的动态文摘表示，取它最后一句也就是包含了整个文档的文摘表示s，输入一个解码器，解码出来生成式文摘。因为文摘表示是解码器的唯一输入，训练解码器的同时也能学习到好的文摘表示，从而完成抽取式文摘的任务。
-	因为在生成二分类概率时包含了几个部分，将它们归一化可以得到各个部分做出的贡献，从而可视化决策过程：

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/J1C5aaef5j.png?imageslim)

# Attention Is All You Need
-	抛弃了RNN和CNN做seq2seq任务，直接用multi head attention组成网络块叠加，加入BN层和残差连接构造深层网络

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/g1BhFElcff.png?imageslim)

-	完全使用attention的一个好处就是快。
-	为了使用残差，所有的子模块（multi-head attention和全连接）都统一输出维度为512
-	编码端：6个块，每个块包含attention和全连接两个子模块，均使用了残差和bn。
-	解码端：也是6个块，不同的是加了一个attention用于处理编码端的输出，而且与解码端输入相连的attention使用了mask，保证了方向性，即第i个位置的输出只与之前位置的输出有关。
-	编码与解码的6个块都是堆叠的(stack)，
-	Attention的通用模型是指将一个query和一系列键值对映射到输出的一种机制，其中输出是值的加权和，而每个值的权重将对应的键和query输入一个兼容性函数计算得到，传统的attention键和值相同，都是输入每个位置上的隐藏层状态，query就是当前输出，兼容性函数就是各种attention计算方法。图中指向attention的三个箭头分别代表key,value,query。

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/DbhA9j0iba.png?imageslim)

-	Multi-head attention由多个scaled dot-product attention并行组成。
-	Scaled dot-product attention如图所示，query和key先做点积，再放缩，如果是解码器输入的attention还要加上mask，之后过softmax函数与value做点积得到attention权重。实际计算时为了加速都是一系列query,key,value一起计算，所以Q,K,V都是矩阵。做放缩是为了防止点积attention在k的维度过大时处于softmax的两端，梯度小。

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/mJFA1DJ8DC.png?imageslim)

-	Multi-head attention就是有h个scaled dot-product attention作用于V,K,Q的h个线性投影上，学习到不同的特征，最后拼接并进行线性变换。作者认为这种multi-head的设计能使模型学习到不同位置的表示子空间的信息。

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/0IBA23dbEJ.png?imageslim)

-	论文中取h=8个head，为了保证维度一致，单个q,k,v的维度取512/8=64
-	这种multi-head attention用在了模型的三个地方：
	-	编码解码之间，其中key,value来自编码输出，query来自解码块中masked multi-head attention的输出。也就是传统的attention位置
	-	编码端块与块之间的自注意力
	-	解码端块与块之间的自注意力
-	在每个块里还有一个全连接层，这个层包含两个线性变换，中间插入了ReLU激活，且每个输入位置都有相同的参数，但不同的块的全连接层参数不同

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/AKjaJ4KLLj.png?imageslim)

-	完全使用注意力的话会抛弃了序列的顺序信息，为了利用这部分信息，加入了三角函数位置编码来利用相对位置信息：

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180908/LELGH5F9ge.png?imageslim)