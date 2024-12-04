---
title: Future of Computing Salon - Reading Comprehension Session
date: 2018-10-13 17:49:06
tags: [comprehension,NLI,]
categories: NLP
---

Attended a light salon at Tsinghua University's FIT, which introduced some advancements in machine reading comprehension. Interestingly, the PhD who spoke at 9 am also mentioned an unpublished work: BERT, which is very impressive and well-funded; it took eight p100 GPUs to train for a year. By 10:30, Machine Intelligence had already published a report, and by the afternoon, Zhihu was buzzing with discussions, saying that a new era for NLP had arrived... This salon is part of a series, and there may be future sessions on machine translation, deep Bayesian, transfer learning, and knowledge graphs, so if you have the time, you might as well listen and take notes.


<!--more-->

{% language_switch %}

{% lang_content en %}
Machine Reading Comprehension
========================================

*   Three speeches, the first being an overview; the second a presentation by the author of nlnet, which won first place on SQuAD2.0, a collaboration between National University of Defense Technology and Microsoft; the third by a Ph.D. from Tsinghua University, who introduced his research on noise filtering and information aggregation in open-domain question answering.

Abstract
--------

*   The current reading comprehension is far behind what is expected from artificial intelligence reading comprehension. Researchers have decomposed the reading comprehension process into tasks such as word selection, span selection, and generating short texts. Before the rise of deep learning, it involved some manually designed features and Pipeline operations. With the advent of deep learning, the focus shifted to end-to-end research from input to output, bypassing many elements required in the reading comprehension process.
*   Previous research on reading comprehension can be used as a testing method to assess the model's ability in lexical, rhetorical, and knowledge utilization skills.
*   The current large-scale machine reading comprehension datasets are at a very low level of inference, as mentioned in a paper: "Efficient and Robust Question Answering from Minimal Context over Documents." It discusses that if deep learning is used, training only with the span you find, cutting out the context, the results actually won't be much different. Therefore, end-to-end learning does not involve a "reading the entire text to grasp the main idea" process, but rather "you ask, I answer, don't ask me why I answer this way, just memorize." It mentions a work from the University of Tokyo that established evaluation indicators for the model's reading comprehension ability, over 30, including the elimination of ambiguity, coreference resolution, etc. Large and simple datasets cannot reflect these features, and the cleverly designed datasets are not large enough in scale.
*   Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks
*   Industrial issues to be solved in the use of reading comprehension: simplifying models or accelerating models, introduces techniques such as SKIM-RNN, which become more complex during training but can accelerate inference. Paper: Neural Speed Reading via SKIM-RNN
*   BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Another form of transfer learning is to directly use a trained module from one task for another, such as directly taking the encoder trained in seq2seq to calculate semantic representations (deep learning is essentially representation learning). Remember, fast Disan directly returns a function that returns the vector representation of a sentence, which is similar in thought.
*   Latest Research Field: open domain question-answering and learning to ask. The former actually adds an information retrieval process, where the relevant corpus needed for reading comprehension is retrieved through asking questions. The latter reverses the answer task to asking questions, with the speaker mentioning that the reverse can assist in reading comprehension, and it has an industrially useful design: instead of comparing queries and documents (or document keywords) during retrieval, it compares them with the questions generated for the documents, which is equivalent to calculating similarity between two questions.
*   The speaker mentioned his view on attention: attention involves filtering information from the model, which does not imply that the original model lacks the ability to represent this information.
*   Presented several currently popular datasets, using MCTest and ProcessBank before 2015, CNNDM, SQuAD, and MS MARCO between 2015 and 2017, and TriviaQA, SQuAD2.0, CoQA, QuAC, and HotpotQA after 2017. (However, the abstract is still using CNNDM...)

NLNet
-----

*   Paper: It can be seen on Squad, but it seems that it hasn't been published yet?
*   NLNet was originally designed to address the robustness and effectiveness in reading comprehension problems, both of which are targeted at ensemble models. Therefore, NLNet adds a distillation process on top of ensemble models, using a single model to improve efficiency, and also includes a read and verify process to enhance robustness. Consequently, it performs exceptionally well on the SQuAD2.0 dataset with adversarial samples, currently ranking first. It lags behind the four-overpowering BERT on version 1.0, but the gap is not significant. However, the ensemble version of NLNet in version 1.0 is better than the single model version, while the 2.0 version did not submit an ensemble version, which is quite perplexing...
*   The meaning of distillation was not fully understood, the effect is to compress 12 models into one model, with the structure of the models completely the same but with different initializations. It is not simply selecting the best; the single model is trained, and the paper refers to the 12 models as teachers and the single model as a student, with the student using the training results of the teacher to guide its training.
*   Designed a read and verify mechanism, which, after extracting a span to answer a question, also calculates a confidence score based on the answer and the question. If the confidence score is too low, it is considered that there is no answer, which is akin to the adversarial sample scenario in SQuAD 2.0. It feels like if there is an issue, loss is added.
*   It is said that some details of feature selection were not presented in the paper, and the model was optimized with reinforcement learning at the end?

Open Domain QA Noise Filtering and Information Aggregation
----------------------------------------------------------

*   Paper (ACL 2018): Denoising Distantly Supervised Open-Domain Question Answering
*   This noise refers to the situation where many relevant documents are found during the retrieval process but do not provide the correct answers, which is a filtering of documents. This step of filtering should have been placed within the retrieval process, but the author ultimately solved it by using a deep learning algorithm to calculate probabilities and loss.
*   The denoising process is a document selector, and then reading comprehension is a reader, the author believes that it corresponds to the fast skimming and careful reading & summarizing that humans do in reading comprehension.
*   The information set did not pay much attention to listening, but fully utilized the information extraction from multiple documents to provide answers
{% endlang_content %}

{% lang_content zh %}


# 机器阅读理解

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