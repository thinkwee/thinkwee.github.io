---
title: Future of Computing Salon - Reading Comprehension Session
date: 2018-10-13 17:49:06
tags: [comprehension,NLI,]
categories: NLP
---

去清华的FIT听了一次轻沙龙，介绍了关于机器阅读理解的一些进展，有趣的是上午九点演讲的博士还说有一个还没公开的工作：BERT，很牛逼，很有钱，八块p100训一年，结果十点半机器之心就发了报道，下午就知乎满天飞了，说NLP新的时代到来了......
这个沙龙是一个系列，之后可能会有机器翻译、深度贝叶斯、迁移学习和知识图谱啥的，要是有时间的话再听再记录吧

<!--more-->

# 2018.10.13 机器阅读理解

- 三场演讲，第一场是概述；第二场是当时在SQuAD2.0上拿到第一名的nlnet作者的presentation，国防科大和微软合作的成果；第三场是一位清华的博士，介绍了他关于开放领域问答中噪声过滤和信息集合的研究。

## 概述

- 现在的阅读理解和人们所期望的人工智能阅读理解差了太多，研究者把阅读理解的过程分解成了任务，例如选词、选span、生成短文本。深度学习兴起之前都是一些手工设计特征，一些Pipiline的操作，使用深度学习之后就专注于输入到输出的端到端研究，绕过了很多阅读理解过程所需要的东西。
- 以前的关于阅读理解的研究可以作为一个测试方法，检验模型对于词法、修辞、利用知识的能力。
- 目前的大规模机器阅读理解数据集处于很低级的推断阶段，提到了一篇论文：Efficient and Robust Question Answering from Minimal Context over Documents。里面讲到如果用深度学习，只用你找出的span来训练，砍掉上下文，其实结果不会差很多，因此端到端的学习并没有“通读全文掌握大意”的过程，而是“你问什么我答什么，别问我为什么这么答，背的”。提到了东京大学一份工作，建立了对模型阅读理解能力的评价指标，30多项，包括消除歧义、指代消解等等，大而简单的数据集无法体现这些特征，而设计巧妙的数据集规模不够大。
- 还提到了一篇关于衡量模型推断能力的论文，TOWARDS AI-COMPLETE QUESTION ANSWERING:A SET OF PREREQUISITE TOY TASKS。
- 工业上使用阅读理解所需要解决的问题：简化模型或者加速模型，介绍了诸如SKIM-RNN之类的技巧，虽然训练的时候会变复杂，但推断时能加速。论文：NEURAL SPEED READING VIA SKIM-RNN
- 现在NLP的迁移学习，预训练词嵌入或者预训练语言模型，用的最多最广泛的，比如Glove和Elmo，然后就提到了BERT，4亿参数，无监督，最重要的是其模型是双向设计，且充分训练了上层参数：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.另外一种迁移学习是直接将某一任务中训练好的模块做其他任务，例如直接将seq2seq中训练好的encoder拿出来计算语义表示（深度学习本来就是表示学习），记得fast Disan就直接一个函数返回句子的向量表示，也是类似的思想。
- 最新的研究领域：open domain question-answering和learning to ask。前者实际上是加了一个信息检索的过程，阅读理解所需要的相关语料是通过提问检索到的。后者是将回答任务反过来做提问，演讲者提到反向可以辅助阅读理解，且有一个工业上比较有用的设计：做检索时不用query和文档（或者文档关键词）相比较，而是和针对文档生成的提问相比较，相当于两个提问之间计算相似度。
- 演讲者提到了他关于attention的一个观点：attention是从模型中筛选信息，不代表原模型没有表示出此信息的能力。
- 介绍了当前比较流行的几个数据集，2015之前用MCTest、ProcessBank，15到17年之间用CNNDM、SQuAD、MS MARCO，17年之后用TriviaQA、SQuAD2.0、CoQA、QuAC、HotpotQA。（然而文摘还在用CNNDM......）

## NLNet

- 论文：Squad上可以看到，但是好像还没发？
- NLNet的设计初衷是为了解决阅读理解问题中的鲁棒性和有效性，都是针对集成模型说的，所以NLNet是在集成模型的基础上加了一个蒸馏的过程，使用单模型提升效率，另外还有一个read and verify的过程来提升鲁棒性，所以在加入了对抗样本的SQuAD2.0数据集上表现优异，目前第一。在1.0上落后于四处碾压的BERT，但其实落后也不多。不过1.0版本中nlnet的ensemble版本要好于单模型版本，2.0中没有提交ensemble版本，就很迷......
- 蒸馏的意思没太听明白，效果是12个模型压缩成一个模型，模型的结构完全相同，但是初始化不同。不是简单的选最优，单一的模型是训练出来的，论文里叫那12个模型为teacher，单一模型为student，student使用teacher训练的结果来指导训练。
- 设计了一个read and verify机制，在抽取出span回答问题之后还会根据该回答和问题计算一个置信度，置信度太低就认为是没有答案，也就是squad2.0里对抗样本的情况。感觉听下来就是有问题就加loss。
- 听说一些选取特征的细节没有在论文中表出，而且最后用强化学习优化了一下模型？

## Open Domain QA噪声过滤和信息集合

- 论文（ACL 2018）：Denoising Distantly Supervised Open-Domain Question Answering
- 这个噪声是指在检索文档的过程搜到了很多相关但提供的不是正确答案的文档，是对文档的过滤。这一步的过滤本来应该放在检索的过程里，但是作者最后也是用深度学习算概率加loss的方式解决了。
- 去噪过程是一个document selector，然后阅读理解是一个reader，作者认为对应于人做阅读理解的fast skimming 和careful reading & summarizing。
- 信息集合没太注意听，就是充分利用多篇文档的信息提取出答案
