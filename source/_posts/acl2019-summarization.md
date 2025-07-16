---
title: Summarization-Related Papers Reading (ACL/NAACL 2019)
date: 2019-08-15 10:51:37
categories: NLP
tags:
  - deep learning
  - summarization
  - natural language processing
mathjax: true
html: true
---

<img src="https://i.mji.rip/2025/07/16/2f67b4f7fae34b1e08aa1c7cc17df6b8.png" width="500"/>


Selected Reading of ACL/NAACL 2019 Automatic Summarization Papers

*   DPPs Similarity Measurement Improvement
    
*   STRASS: Backpropagation for Extractive Summarization
    
*   Translate first, then generate the abstract
    
*   Reading Comprehension + Automatic Abstract
    
*   BiSET: Retrieve + Fast Rerank + Selective Encoding + Template Based
  
<!--more-->

{% language_switch %}

{% lang_content en %}
Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization
=============================================================================================================

*   This is very similar to what the others in our group have done, using DPPs to process extractive abstractive summaries
*   In the abstract indicator paper mentioned in the preceding text, it is also mentioned that creating an abstract, especially a key sentence abstract, is all about three words: qd, qd, and still qd!
    *   q: quality, which sentence is important and needs to be extracted as an abstract. This step is feature construction.
    *   d: diversity, the sentences extracted should not be redundant, and should not repeatedly use the same sentences. If there are too many important sentences that are the same, they become less significant. This step is the construction of the sampling method.
*   Determinantal Point Processes (DPPs) are a sampling method that ensures the extracted sentences are important (based on precomputed importance values) and non-repetitive. The authors of this paper have a very clear line of thought: I want to improve the DPPs in extractive summarization, how to improve them? DPPs rely on the similarity between sentences to avoid extracting duplicate sentences, so I will directly improve the calculation of similarity, thus the problem is shifted to a very mature field: semantic similarity computation.
*   Next, just use the web to do semantic similarity calculation. The author is quite innovative, using capsule networks, which were originally proposed to solve the problem of relative changes in object positions in computer vision. The author believes that it can be generalized to extract spatial and directional information of low-level semantic features. Here, I am not very familiar with capsule networks and their applications in NLP, but based on the comparative experiments provided by the author, the improvement is actually just one point, and the entire DPP is only 2 points better than the best system before (2009), which seems a bit forced.
*   The network provided by the author is truly complex, not in terms of principle, but due to the use of many components, including:
    *   CNN with three to seven different sizes of convolutional kernels for extracting low-level features
    *   Capsule networks extract high-level features, utilizing recent techniques such as parameter sharing and routing
    *   One-hot vectors were still used, i.e., whether a word exists in a certain sentence
    *   Fusion of various features, including inner product, absolute difference, and concatenation with all independent features, to predict the similarity between two sentences
    *   And the similarity is only part of the goal; the authors also used LSTM to reconstruct two sentences, incorporating the reconstruction loss into the final total loss ![ezCge0.png](https://s2.ax1x.com/2019/08/12/ezCge0.png) 
*   Absolutely, at first glance, one would assume this is a CV's work
*   The author at least used the latest available techniques, creating an integrated network, which may not be as concise and elegant in an academic sense, but in terms of industry, many such network integration operations are very effective
*   Another point is that although it is a sampled abstract, the author's work is fully supervised, so a dataset still needs to be constructed
    *   Constructing Supervised Extractive Summary Datasets from Generative Summary Datasets
    *   Constructing a supervised sentence similarity calculation dataset from generative abstract data sets
    *   This structure also limits its generalization ability to some extent
*   Authors' starting point is actually very good: because traditional similarity is at the word level, without delving into semantic features, the direction of constructing a network to extract features is correct, albeit somewhat complex. Moreover, since only the sentence feature extraction part of the similarity calculation in the extraction-based abstracting method has been improved, the overall impact is not particularly significant. The final result may have outperformed many traditional methods, but it has not improved much compared to the traditional best method, and it is only about 1 point better than pure DPPs.

STRASS: A Light and Effective Method for Extractive Summarization Based on Sentence Embeddings
==============================================================================================

![mkkJHO.png](https://s2.ax1x.com/2019/08/14/mkkJHO.png)

*   Another paper using supervised methods for extractive summarization, the content can be roughly guessed from the title, based on embedding, and aiming to be light and effective, the simplest goal is to keep the embedding of the summary consistent with the gold embedding.
    
*   The difficulty lies in the fact that it is extractive and discrete, thus requiring a process to unify the three parts of extraction, embedding, and comparison scoring, softening it to be differentiable, enabling end-to-end training. The authors propose four steps:
    
    *   Mapping document embeddings to a comparison space
    *   Extract sentences to form an abstract
    *   Extraction-based abstract embedding
    *   Comparison with gold summary embedding
*   First, given a document, directly obtain the doc embedding and the sentence embedding for each sentence in the document using sent2vec
    
*   After that, only one fully connected layer is used as the mapping function f(d), where the author proposes the first hypothesis: the extracted abstract sentences should have similarity to the document:
    
    $$
    sel(s,d,S,t) = sigmoid (ncos^{+}(s,f(d),S)-t)
    $$
    
*   s is the sentence embedding, S represents the set of sentences, t is the threshold. sel represents select, i.e., the confidence of selecting this sentence to form the summary. This formula indicates that the similarity between the selected sentence embedding and the document embedding should be greater than the threshold t, and sigmoid is used for softening, converting {0,1} to \[0,1\].
    
*   Afterward, further softening is applied; the author does not select sentences to form the abstract based on scores, but directly approximates the abstract's embedding based on scores
    
    $$
    app(d,S,t) = \sum _{s \in S} s * nb_w(s) * sel(s,d,S,t)
    $$
    
*   nb\_w is the number of words, i.e., the sum of the embedding of all sentences weighted by the number of words in each sentence and the select score to obtain the embedding of the generated summary
    
*   The final step involves comparing the embedding similarity calculation loss with the gold summary, where the authors introduce a regularization term to aim for a higher compression ratio of the extracted summary. I feel that this is a compensation brought about by a series of softening operations in the previous step, as no sentences are selected; instead, all sentences are weighted, thus necessitating regularization to force the model to discard some sentences:
    
    $$
    loss = \lambda * \frac{nb_w(gen_sum)}{nb_w(d)} + (1-\lambda) * cos_{sim}(app(d,S,t),ref_{sum})
    $$
    
*   What is the method for obtaining the embedding of the gold summary?
    
*   The authors also normalized the results of the cosine similarity calculation to ensure that the same threshold could be applied to all documents
    
*   The results actually show that ROUGE is not as good as generative methods, of course, one reason is that the dataset is inherently generative, but it is strong in simplicity, speed, and when using supervised methods for extraction, there is no need to consider the issue of redundancy.
    

A Robust Abstractive System for Cross-Lingual Summarization
===========================================================

*   In fact, one sentence can summarize this paper: Generate multilingual abstracts by first translating, while others first abstract and then translate
*   All are implemented using existing frameworks
    *   Marian: Fast Neural Machine Translation in C++
    *   Abstract: Pointer-generator
*   The author actually has sufficient supervisory data; it was previously thought that the abstracts were multilingual, with small amounts of corpus, or were summaries not relying on translation, which could extract common abstract features across multiple languages
*   However, this paper indeed achieved robustness: generally, to achieve robustness, one introduces noise, and this paper exactly used back-translation to introduce noise: first, English is translated into a minor language, then translated back, and trained a generative abstract model on this bad English document, making the model more robust to noise. The final results were also significantly improved, and it also achieved good effects on Arabic that had not been trained, indicating that different people's translations have their own correctness, while the errors in machine translation are always similar.

Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction
==========================================================================================

![mtX0eS.png](https://s2.ax1x.com/2019/08/21/mtX0eS.png)

*   This paper comes from the Smart Lab of NTT, the so-called largest telecommunications company in the world, and proposes a multi-task model: Reading Comprehension + Automatic Summary
    
*   The paper conducted many experiments and analyses, and provided a detailed analysis of the conditions under which their module works, and also utilized many techniques proposed in recent years for the abstract section, rather than simply patching together.
    
*   This paper is also based on the HotpotQA dataset, similar to the one in CogQA, but that one used the full wiki setting, which means there was no gold evidence. This paper, however, requires gold evidence, so it used the HotpotQA distractor setting.
    
*   For the distractor setting of HotpotQA, the supervisory signal consists of two parts: answer and evidence, with the input also having two parts: query and context, where the evidence is a sentence within the context. The authors adopt the baseline from the HotpotQA paper: Simple and effective multi-paragraph reading comprehension, and all parts except the Query-Focused Extractor shown in the above figure. The basic idea is to combine the query and context, add a lot of fully connected (FC), attention, and BiRNN to extract features, and finally output a classification of answer type and a sequence labeling of answer span in the answer part, while directly applying the output of the BiRNN to each sentence for binary classification. ![mtXczq.png](https://s2.ax1x.com/2019/08/21/mtXczq.png) 
    
*   The author refines the supervision task of evidence into a query-based summarization, adding a module called Query-Focused Extractor (QFE) after the BiRNN, emphasizing that the evidence should be a summary extracted from the context under the query conditions, satisfying:
    
    *   Sentences within the summary should not be redundant
    *   sentences within the summary should have different attention based on the query
*   For the first point, the author designed an RNN within the QFE, which allows attention to be paid to previously extracted sentences during the generation of attention and even the extraction of summaries. The time step of the RNN is defined as each time a sentence is extracted, with the input being the vector of the sentence extracted at that time step
    
*   In response to the second point, the author added an attention mechanism for the query within the QFE, with the weighted query vector referred to as glimpse. Note that this is the attention from the QA context to the QA query; both the key and value in the attention are the QA query, while the query in the attention does not directly take the entire QA context but rather the output of the RNN, i.e., the context encoded by the RNN after extracting a set of sentences. Such a design is also intuitive.
    
*   After the RNN encodes the extracted sentences and forms glimpse vectors with attention-weighted queries, QFE receives these two vectors, combines them with the vectors of unextracted sentences for each context, to output the probability of each sentence being extracted, and then selects the sentence with the highest probability to add to the set of extracted sentences. Subsequently, the system continues to cyclically calculate the RNN and glimpse. The dependency relationships of the entire system are clearly shown in the figure above.
    
*   Due to the variable number of sentences in gold evidence, the author employs the method of adding a dummy sentence with an EOE to dynamically extract, and when an EOE is extracted, the model no longer continues to extract sentences.
    
*   During training, the loss function for evidence is:
    
    $$
    L_E = - \sum _{t=1}^{|E|} \log (max _{i \in E / E^{t-1}} Pr(i;E^{t-1})) + \sum _i min(c_i^t, \alpha _i^t)
    $$
    
    Here, $E$ is the set of sentences of gold evidence, $E^t$ is the set of sentences extracted by QFE, $\alpha _i^t$ is the attention of the i-th word in the query at time step t, where the time step is consistent with the previous text, being the time step for extracting sentences. $c^t = \sum _{i=1}^{t-1} \alpha ^i$ is the coverage vector. The first half of the loss refers to the negative log-likelihood loss of the gold evidence, finding the gold sentence with the highest QFE prediction probability in the extracted sentence set, calculating the loss, and then excluding this sentence to find the next highest, until all gold sentences are found or no gold sentence can be found in the extracted sentence set. The second half is a regularization application of the coverage mechanism to ensure that the sentences selected for loss calculation do not have overly repetitive (concentrated) attention on the query.
    
*   Authors achieved results on the HotpotQA and textual entailment dataset FEVER, with the evidence part of the indicators far superior to the baseline, while the answer part also saw a significant improvement, though not as pronounced as the evidence part, and slightly inferior to the BERT model. On the full wiki test set, it was also comprehensively surpassed by CogQA. Here, the authors state that there is a dataset shift problem. However, at least this paper achieved an 8-point improvement on the answer part by simply adding a small module to the baseline, demonstrating that a well-designed summarization part indeed helps in the selection of answers in multi-task learning.
    

BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization
====================================================================================

*   Another model pieced together from various components, the title actually spells it all out: Bi-directional, selective encoding, template, together forming the BiSET module, and the other two preceding processes: Retrieve and Fast Rerank also follow the architecture from the paper "Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization." It should be based on soft template summarization, with the mechanism of selective encoding added, so these two papers are put together to discuss template-based generative summarization and its improvements.
*   The idea behind the soft template approach is not to let the model generate sentences entirely, but rather for humans to provide the template and for the model to only fill in the words. However, if the template is completely designed manually, it would regress to the methods of several decades ago. The author's approach is to automatically extract templates from existing gold summaries.
*   Generally divided into three steps:
    *   Retrieve: Extract candidates from the training corpus
    *   Rerank: Learning Template Saliency Measurement for seq2seq Models
    *   Rewriting: Let the seq2seq model learn to generate the final summary
*   This method should be more suitable for long sentence compression, or for single-sentence generative summarization, where the long sentences to be compressed can be used as queries for retrieval

Retrieve
--------

*   Utilizing the existing Lucene search engine, given a long sentence to be compressed as a query, search the document collection to identify the summaries of the top 30 documents as candidate templates

Rerank
------

*   The abstracts (soft template) retrieved through the search are sorted by relevance, but we require sorting by similarity. Therefore, we use the ROUGE score to measure the similarity between the soft template and the gold summary. Here, reranking is not about sorting out the results but rather considering the rank of each template comprehensively during the generation of the summary, and the loss can be observed in the parts that are omitted.
    
*   Specifically, first use a BiLSTM encoder to encode the input x and a certain candidate template r; here, the hidden layer states are encoded separately, but the same encoder is used, and then input the two hidden layer states into a Bilinear network to predict the ROUGE value between the gold summary y corresponding to the input x and r, which is equivalent to a network that makes a saliency prediction for r given x:
    
    $$
    h_x = BiLSTM(x) \\
    h_r = BiLSTM(r) \\
    ROUGE(r,y) = sigmoid(h_r W_s h_x^T + b_s) \\
    $$
    
*   This completes the supervised part of reranking
    

Rewrite
-------

*   This part is a standard seq2seq, still using the previously encoded $h_x, h_r$ to concatenate it and feed it into an attentional RNN decoder to generate an abstract, and calculate the loss

Jointly Learning
----------------

*   The model's loss is divided into two parts. The Rerank part ensures that the encoded template and the input, after passing through bilinear processing, can correctly predict the ROUGE value. The Rewrite part ensures the generation of a correct summary. This is equivalent to, in addition to the ordinary seq2seq summary generation, I also candidate some other gold summaries as input. This candidate is initially filtered through retrieval. When used, the Rerank part guarantees that the encoded part is the template component within the summary, i.e., the part that can be taken out and compared with the gold summary, thereby assisting the decoder in the Rewrite part's generation.

result
------

*   We know that in summarization, the decoder is actually very dependent on the encoder's input, which includes both the template and the original input. The authors provide several ideal examples, where the output summary is basically in the format of the template, but the key entities are extracted from the original input and filled into the template summary.
*   Although a somewhat esoteric rerank loss method was used for extracting the soft template, the role of the template is indeed evident. The model actually finds a summary that is very close to the gold summary as input, and makes slight modifications (rewrites) on this basis, which is much more efficient than end-to-end seq2seq. The authors also tried removing the retrieve step and directly finding the ROUGE score highest summary from the entire corpus as the template, with the final model's results reaching 50 ROUGE-1 and 48 ROUGE-L
*   This operation of taking the output as input is actually a compensation for the insufficient abstract ability of the decoder, and it is an empirical method derived from the observation of the dataset, which can effectively solve the problem

biset
-----

*   Replaced the rerank part with CNN+GLU to encode documents and queries, and then computed the sim matrix using the encoded vectors


{% endlang_content %}

{% lang_content zh %}

# Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization

- 这和我们组其他人做的很像，用DPPs处理抽取式文摘
- 在上文的文摘指标论文中也提到了，搞文摘，尤其是抽取式文摘，就是三个词，qd，qd，还是qd!
  - q:quality，哪个句子重要，需要被抽出来作为文摘。这一步是特征的构造
  - d:diversity,抽取出来的句子之间不能冗余，不能老抽一样的句子。重要的一样的句子多了，也就变得不重要了。这一步是抽样方法的构造
- DPPs(Determinantal Point Processes)就是一种抽样方法，保证抽出来的句子重要（根据特征计算好的重要性数值），而且不重复。本文作者就是思路非常清晰：我要改进抽取式文摘中的DPPs，怎么改进？DPPs依赖句子之间的相似度来避免抽重复的句子，那我就改进不管DPPs怎么改，直接改相似度的计算，所以问题就换到了一个非常成熟的领域：语义相似度计算。
- 接下来就随便套网络来做语义相似度计算就行了。作者比较新潮，用了胶囊网络，这个网络本来是提出用于解决计算机视觉中物体的位置相对变化问题的，作者认为可以将其泛化到提取底层语义特征的空间与方位信息，这里我对胶囊网络及其在NLP的应用不太了解，但是就作者给出的对比实验来说，改进其实也就1个点，而整个DPP相比之前最好系统（2009）也就好2个点，感觉还是有点刻意为之。
- 另外作者给出的网络是真的复杂，不是原理复杂，而是用了很多组件，包括：
  - 三四五六七大小卷积核的CNN用于提取底层特征
  - 胶囊网络提取高层特征，用到了近年来的参数共享和路由等技巧
  - 还是用了one-hot向量，即一个词是否存在在某一个句子里
  - 各种特征的融合，包括内积，绝对差，再和所有独立的特征全部拼接到一起，预测两个句子的相似度
  - 而且相似度还只是一部分目标，另外作者还用了LSTM来重构两个句子，将重构损失加入最终的总损失
    ![ezCge0.png](https://s2.ax1x.com/2019/08/12/ezCge0.png)
- 粗看图片绝对以为这是一篇CV的work
- 作者好歹是把最近能用的技巧都用上了，做了一个集大成的网络，可能学术上看没那么简洁优美，但是就工业上来说很多这样的网络集成操作就是很work
- 另外虽然是抽取式摘要，但作者的工作是完全监督，因此还需要构造数据集
  - 从生成式摘要数据集中构造有监督抽取式摘要数据集
  - 从生成式摘要数据集中构造有监督句子相似度计算据集
  - 这种构造也一定程度上限制了其泛化能力
- 作者的出发点其实非常好：因为传统的相似度停留在词的粒度，没有深入到语义特征，因此构造网络提取特征的方向没错，只不过稍显复杂，而且由于仅仅改进了抽取式文摘中某一种抽样方法的相似度计算中的句子特征提取部分，对整体的影响并没有特别大，最后的结果虽然吊打了很多传统方法，但是相比传统最佳方法并没有提高多少，相比纯DPPs更是只有1个点左右的提高。

# STRASS: A Light and Effective Method for Extractive Summarization Based on Sentence Embeddings

![mkkJHO.png](https://s2.ax1x.com/2019/08/14/mkkJHO.png)

- 又是一篇用监督方法做抽取式文摘的，从标题就可以大致猜出来内容，基于embedding，还要light and effective，那最简单的目标就是把摘要的embedding和gold embedding保持一致。

- 困难的地方在于这是抽取式的，是离散的，因此需要一个流程把抽取、embedding、比较打分三个部分统一起来，软化使其可导，可以端到端训练，作者给出的是四个步骤：
  
  - 将文档embedding映射到比较空间
  - 抽句子组成摘要
  - 去近似抽出的摘要的embedding
  - 和gold summary的embedding比较

- 首先给定一篇文档，直接用sent2vec获得doc embedding和文档里每一句的sentence embedding

- 之后只用一层全连接作为映射函数f(d)，这里作者给出了第一个假设：抽取的摘要句子应该和文档具有相似度：
  
  $$
  sel(s,d,S,t) = sigmoid (ncos^{+}(s,f(d),S)-t)
  $$

- 其中s是句子embedding，S代表句子集，t为阈值。 sel代表select，即选择这个句子组成摘要的置信度，这个式子说明选出句子的embedding和文档embedding之间的相似度应该大于阈值t，且使用sigmoid做了软化，将{0,1}软化为[0,1]

- 之后进一步软化，作者并不根据分数选出句子组成文摘，而是根据分数直接近似文摘的embedding
  
  $$
  app(d,S,t) = \sum _{s \in S} s * nb_w(s) * sel(s,d,S,t)
  $$

- 其中nb_w是number of words，即使用每个句子的字数和select score对所有的句子embedding加权求和得到generated summary的embedding

- 最后一步，和gold summary比较embedding相似度计算损失，这里作者加入了一个正则项，希望提出来的摘要压缩比越高越好，这里我感觉是上一步一系列软化操作带来的补偿，因为没有选择句子，而是对所有句子加权，因此需要正则强迫模型放弃一些句子：
  
  $$
  loss = \lambda * \frac{nb_w(gen_sum)}{nb_w(d)} + (1-\lambda) * cos_{sim}(app(d,S,t),ref_{sum})
  $$

- 这里有一个问题，gold summary的embedding是怎么得到的？

- 另外为了保证能够对所有文档使用同一个阈值，作者还对cosine相似度计算的结果做了归一化

- 从结果来看其实ROUGE还不如生成式的方法好，当然一方面原因是因为数据集本来就是生成式的，但是强在简单，快，而且用监督的方法做抽取式也不用考虑redundency的问题。

# A Robust Abstractive System for Cross-Lingual Summarization

- 其实一句话就能概括这篇论文：做多语言生成式文摘，别人是先摘要再翻译，这篇文章是先翻译再生成摘要
- 均是用已有框架实现
  - 翻译：Marian: Fast Neural Machine Translation in C++
  - 摘要：pointer-generator
- 作者其实有充足的监督数据，之前以为是多语言、小语料的摘要，或者是不借助于翻译的摘要，能够挖掘多语种共有的摘要特征
- 但是这篇论文确实实现了robust：一般要做robust就是引入噪声，本文正好用了回译引入噪声：先把英语翻译成小语种，翻译回来，在这样的bad english document上训练生成式摘要模型，使得模型对噪声更加鲁棒，最后的结果也是提高了许多，并且对没有训练过的阿拉伯语也取得了较好的效果，说明不同的人翻译各有各的正确，而机器翻译的错误总是相似的。

# Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction

![mtX0eS.png](https://s2.ax1x.com/2019/08/21/mtX0eS.png)

- 这篇论文来自号称全球最大电信公司NTT的智能实验室，提出了一个多任务模型：阅读理解+自动摘要

- 论文做了很多实验和分析，并且详细分析了在何种情况下他们的module works，对于摘要部分也利用了许多近年来提出的技巧，而不是简单的拼凑。

- 这篇论文同样也是在HotpotQA数据集上做，和CogQA那一篇一样，但那一篇用的是full wiki setting，即没有gold evidence，而这篇需要gold evidence因此用了HotpotQA 的distractor setting。

- 对于HotpotQA的distractor setting，监督信号有两部分：answer和evidence，输入有两部分:query和context，其中evidence是context当中的句子。作者沿用了HotpotQA论文里的baseline:Simple and effective multi-paragraph reading comprehension，及上图中Query-Focused Extractor以外的部分。基本思想就是将query和context结合，加上一堆FC,attention，BiRNN提取特征，最终在answer部分输出一个answer type的分类和answer span的sequence labelling，而在evidence部分直接接BiRNN输出的结果对每个句子做二分类。
  ![mtXczq.png](https://s2.ax1x.com/2019/08/21/mtXczq.png)

- 作者将evidence这边的监督任务细化为一个query based的summarization，就在BiRNN后面加了一个模块，称之为Query-Focused Extractor(QFE)，相比原始的简单二分类，QFE强调了evidence应该是在query条件下从context中抽取出来的summary，因满足：
  
  - summary内的句子之间应该不能冗余
  - summary内不同句子应该有着query上不同的注意力

- 针对第一点，作者在QFE内设计了一个RNN，使得在生成注意力乃至抽取摘要时都能注意到之前已经抽取出来的句子,其中RNN的时间步定义为每一次抽取一个句子，输入即某一时间步抽取出的句子的vector

- 针对第二点，作者在QFE内增加了一个针对query的注意力，加权之后的query向量称为glimpse，注意这里是QA context对QA query的注意力，attention里的key和value都是QA的query，而attention里的query不是直接拿整个QA context，而是RNN的输出，即已经抽取出的句子集经过RNN编码的context，这样的设计也是符合直觉的。

- 在RNN编码已抽取句子、注意力加权query形成glimpse向量之后，QFE拿到这两部分向量，结合每一个context未抽取句子的向量来输出每一个句子被抽取的概率，并选择最大概率的句子加入已抽取句子集合，然后接着循环计算RNN和glimpse。整个系统的依赖关系在上图中展示的很清晰。

- 由于gold evidence的句子数目不固定，作者采用添加一个EOE的dummy sentence的方法来动态抽取，当抽取到EOE时，模型就不再接着抽取句子。

- 在训练时，evidence这边的损失函数为：
  
  $$
  L_E = - \sum _{t=1}^{|E|} \log (max _{i \in E / E^{t-1}} Pr(i;E^{t-1})) + \sum _i min(c_i^t, \alpha _i^t)
  $$
  
  这里$E$是gold evidence的句子集合，$E^t$是QFE抽取出来的句子集合，$\alpha _i^t$是t时间步query里第i个词的注意力，这里的时间步和前文一致，是抽句子的时间步。而$c^t = \sum _{i=1}^{t-1} \alpha ^i$是coverage向量。
  损失的前半部分指的是gold evidence的负对数似然损失，依次在抽取句子集合里找拥有最大QFE预测概率的gold sentence，算损失，然后排除这个句子接着找下一个最大的，直到gold sentence找完或者抽取句子集合里找不到gold sentence，后半部分是coverage机制的一个正则化应用，保证挑出来计算损失的句子不会在query上拥有过于重复（集中）的注意力。

- 作者在HotpotQA和文本蕴含数据集FEVER上做了结果，evidence部分的指标远好于baseline，而answer部分的指标也有较大提升，但不如evidence部分明显，且与BERT模型部分相比还差一点，在full wiki setting的测试集上也被CogQA全面超过，这里作者说存在dataset shift问题。但至少本文仅仅在baseline上添加了一个小模块，就获得了answer部分的8个点的提升，说明精心设计的summarization部分在多任务学习中确实帮助到了answer的选取。 

# BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization

- 又是一篇将各个组件拼拼凑凑出来的一个模型，标题其实已经全写出来了：Bi-directional， selective encoding， template，共同组成了BiSET模块，另外两个前置过程：Retrieve和Fast Rerank也是沿用Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization这篇论文里的架构。应该大体是基于soft template的summarization，加上了selective encoding的机制，因此就把这两篇论文放在一起，讨论基于模板的生成式摘要及其改进。
- 基于软模板的思想是，不要完全让模型来生成句子，而是人给出模板，模型只负责填词。然而完全人工设计模板那就退化到几十年前的方式了，作者的思路是，从已有的gold summary中自动提取模板。
- 大体分为三步：
  - Retrieve：从训练语料中检索出候选软模板
  - Rerank：让seq2seq模型学习到template saliency measurement
  - Rewrite：让seq2seq模型学习到final summary generation
- 这种方法应该比较适用于长句压缩，或者说单句生成式摘要，这样待压缩的长句可以作为query进行retrieve

## Retrieve

- 使用现成的Lucene搜索引擎，给定要压缩的一个长句作为query，从文档集中搜索出top 30篇文档的summary作为候选模板

## Rerank

- 经过搜索搜出来的摘要（soft template)是按照搜索相关度排序的，但我们需要的是按照摘要相似度排序，因此我们使用ROUGE值衡量soft template和gold summary之间的相似程度，这里的rerank并不是要真的排序出来，而是在生成摘要时综合考虑每个template的rank程度，之后在损失部分可以看出来。

- 具体而言，先用一个BiLSTM编码器编码输入x和某一个候选模板r，这里是分别编码隐层状态，但是共用编码器，之后将两个隐层状态输入一个Bilinear网络预测出输入x对应的gold summary y和r之间的ROUGE值，相当于这是一个给定x，给r做出saliency prediction的网络：
  
  $$
  h_x = BiLSTM(x) \\
h_r = BiLSTM(r) \\
ROUGE(r,y) = sigmoid(h_r W_s h_x^T + b_s) \\
  $$

- 这就完成了rerank的监督部分

## Rewrite

- 这部分就是普通的seq2seq，依然是利用之前编码好的$h_x, h_r$，将其拼接起来送入一个attentional RNN decoder生成摘要，计算损失

## Jointly Learning

- 模型的损失分为两部分，Rerank部分要保证编码出来的template和输入再经过bilinear之后能正确预测ROUGE值，Rewrite部分要保证生成正确的摘要，相当于在普通的seq2seq生成摘要之外，我还候选了一些其他的gold summary作为输入，这个候选是通过retrieve的方式粗筛选的，具体使用时通过Rerank部分保证encode出来的部分是summary里的template成分，即可以拿出来和gold summary比对的部分，从而辅助rewrite部分decoder的生成。

## result

- 我们知道做summarization，decoder通过注意力其实是很依赖encoder的输入的，这里的encoder输入既包含template，又包含原始输入，作者给出了几个比较理想的例子，即输出的summary基本上按照template的格式，但是在关键实体部分从原始输入中提取实体填到template summary当中。
- 虽然如何提取soft template这方面使用了一个比较玄学的rerank loss的方式，但是template的作用确实很明显，模型实际上是找到和gold summary很接近的一个summary作为输入，在此基础上稍加更改(rewrite)，效率远比端到端的seq2seq好，作者还尝试了去掉retrieve，直接从整个语料中找ROUGE最高的summary作为template，最后模型出来的结果高达50的ROUGE-1，48的ROUGE-L
- 这种找输出作为输入的操作，其实是对decoder抽象能力不足的一种补偿，是对数据集观察得出的经验方法，能很实际的解决问题

## biset

- 将rerank部分换成了CNN+GLU来编码文档和查询，编码后的向量计算sim matrix，

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