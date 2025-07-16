---
title: Seq2seq based Summarization
date: 2018-07-04 15:58:59
tags:
  - abstractive summarization
  - seq2seq
  - machine learning
  - rnn
  - nlp
  - lstm
  - gru
categories:
  - NLP
mathjax: true
html: true
---

<img src="https://i.mji.rip/2025/07/16/2f67b4f7fae34b1e08aa1c7cc17df6b8.png" width="500"/>

A bachelor's graduation project involves developing a short sentence summarization model based on seq2seq and designing an emotional fusion mechanism. Now, let's provide a brief summary of the entire model

<!--more-->

![i0TGHH.png](https://s1.ax1x.com/2018/10/20/i0TGHH.png)

{% language_switch %}

{% lang_content en %}

Task
====

*   Automatic text summarization is a type of natural language processing (NLP) task. For a longer text, it generates a short text that covers the core meaning of the original text, which is the summary of the original text. Automatic summarization technology refers to constructing mathematical models on computers, inputting long texts into the model, and then automatically generating short summaries through computation. According to the scale of the corpus needed for generating summaries and the scale of the summaries, summaries can be divided into multi-document summaries, long document summaries, and short document summaries. This paper mainly studies short document summarization: for a sentence or a few sentences of text, generate a short summary that summarizes the key information of the original text, and is fluent and readable, trying to reach the level of summaries written by human authors.
*   Automatic text summarization is divided into extraction-based and generation-based methods, the former being the extraction of original sentences to form the summary, and the latter being the generation of the summary through a deep learning model, character by character. This paper mainly focuses on the generation of summaries and abstracts the problem into generating a short sentence of an average length of 8 words from a long sentence with an average length of 28 words.

![i0TYEd.jpg](https://s1.ax1x.com/2018/10/20/i0TYEd.jpg)

Preparatory Knowledge
=====================

Recurrent Neural Network
------------------------

*   Recurrent Neural Network (RNN), a variant of neural networks, is capable of effectively processing sequential data. All its hidden layers share parameters, with each hidden layer not only depending on the current moment's input but also on the state of the previous hidden layer. The data flow is not propagated between network layers as in traditional neural networks, but rather circulates as a state within its own network.
*   ![i0TtUA.jpg](https://s1.ax1x.com/2018/10/20/i0TtUA.jpg)
*   After time step expansion: ![i0TN4I.jpg](https://s1.ax1x.com/2018/10/20/i0TN4I.jpg) 

LSTM and GRU
------------

*   Recurrent neural networks can effectively capture the sequential information of sequence data and can construct very deep neural networks without generating a large number of parameters to be learned; however, due to parameter sharing, when gradients are chain-derived through time steps, it is equivalent to matrix power operations. If the eigenvalues of the parameter matrix are too small, it will cause gradient diffusion, and if the eigenvalues are too large, it will cause gradient explosion, affecting the backpropagation process, i.e., the long-term dependency problem of RNNs. When dealing with long sequence data, the long-term dependency problem can lead to the loss of long-term memory information in the sequence. Although people have tried to alleviate this problem by introducing gradient truncation and skip connection techniques, the effect is not significant until the long short-term memory neural networks and gated recurrent neural networks, as extended forms of RNNs, effectively solve this problem.
    
*   LSTM stands for Long Short-Term Memory, a type of neural network that introduces gate units in its nodes to capture long-term memory information. These gate units, as part of the network parameters, participate in training and control the extent to which the current hidden layer node memory (forgetting) past information and accepts new memory input. ![i0TaCt.jpg](https://s1.ax1x.com/2018/10/20/i0TaCt.jpg) 
    
*   GRU stands for Gated Recurrent Unit, which differs from LSTM in that GRU integrates the forget gate and input gate into a reset gate, with the control values of the forget gate and input gate summing to 1. Therefore, GRU simplifies the parameters on the basis of LSTM, allowing the model to converge faster. ![i0Td8P.jpg](https://s1.ax1x.com/2018/10/20/i0Td8P.jpg) 
    
    Word Embedding
    --------------
    
*   One of the major advantages of deep learning is its ability to automatically learn features. In natural language processing, we specifically use techniques like word2vec to learn the feature representations of words, i.e., word embeddings.
    
*   Word vectors, also known as word embeddings, represent words in the form of continuous value vectors (Distributed Representation), rather than using a discrete method (One-hot Representation). In traditional discrete representation, a word is represented by a vector of length V, where V is the size of the dictionary. Only one element in the vector is 1, with the rest being 0, and the position of the 1 indicates the word's index in the dictionary. Storing words in a discrete manner is inefficient, and the vectors cannot reflect the semantic and grammatical features of words, whereas word vectors can address these issues. Word vectors reduce the dimension of the vector from V to $\sqrt[k] V$ (usually k takes 4), with the values of each element no longer being 1 and 0, but continuous values. Word vectors are a byproduct of supervised learning obtained from the Neural Network Language Model (NNLM) for corpus, and the proposal of this model is based on a linguistic assumption: words with similar semantics have similar contexts, that is, the NNLM model can determine the corresponding central word under the given context.
    
*   The figure below illustrates the skipgram model in word2vec: ![i0Twgf.jpg](https://s1.ax1x.com/2018/10/20/i0Twgf.jpg) 
    
*   The obtained word embedding matrix is as follows: ![i0T0v8.jpg](https://s1.ax1x.com/2018/10/20/i0T0v8.jpg) 
    
*   Mikolov et al. proposed the Word2Vec model based on NNLM, where the input and output for supervised learning are respectively the center word and its context (i.e., the Skip Gram model) or the context and the center word (i.e., the CBOW model). Both methods can effectively train high-quality word vectors, but the CBOW model calculates the center word based on the context, has a fast training speed, and is suitable for training on large corpora; the Skip Gram model can fully utilize the training corpus, and its meaning is "jumping grammar model." It not only uses adjacent words to form the context of the center word but also uses words that are one word apart as part of the context. As shown in Figure 2-1, the context of a center word in the corpus includes four words. If there are Wtotal words in the corpus, the Skip Gram model can calculate 4 · Wtotal times of loss and perform backpropagation learning, which is four times the number of learning times for the corpus compared to the CBOW model, so this model is suitable for fully utilizing small corpora to train word vectors.
    
*   Word2Vec model training is completed, and the weight matrix between the input layer and the hidden layer is the Word Embedding Matrix. Multiplying the vector representing the discrete word with the matrix yields the word vector, which is actually equivalent to looking up the corresponding word vector (Embedding Look Up) in the word embedding matrix. Word vectors can effectively represent the semantic relationships between words, essentially providing a method for machine learning models to extract text information features, facilitating the numerical input of words into the model for processing. Traditional language model training of word vectors incurs too much overhead in the output Softmax layer.
    
*   The Word2Vec model employs both Hierarchical Softmax and Noise Contrastive Estimation techniques, significantly accelerating the training process, making it possible to train high-quality word vectors with large-scale corpus in natural language processing.
    

Attention
---------

*   In NLP tasks, the attention mechanism was first applied to machine translation, where a weight matrix is introduced to represent the contribution degree of each element in the encoder sequence to each word generated by the decoder. In practical implementation, the attention mechanism generates an attention weight, which performs attention-weighted generation of intermediate representations for the hidden layer states of various encoder elements, rather than simply using the hidden layer state of the last element. The simplest attention is the decoder's attention to the encoder, which is divided into global and local attention. Global attention generates attention weights for the entire encoder sequence, while local attention first trains an attention alignment position and then takes a window around this position, weighting only the sequence within the window, making the attention more precise and focused. One byproduct of the attention mechanism is the alignment between words (Alignment). In machine translation, the alignment relationship can be understood as the degree of association between words and their translations. In automatic summarization, the application of the attention mechanism can effectively alleviate the problem of information loss when long sequences are encoded into intermediate representations by the encoder. ![i0TDKS.jpg](https://s1.ax1x.com/2018/10/20/i0TDKS.jpg) 

Sequence to sequence
--------------------

*   seq2seq model, which uses an RNN as an encoder to encode the input sequence data into intermediate semantic representation, and then utilizes another RNN as a decoder to obtain the serialized output from the intermediate semantic representation. Generalized sequence-to-sequence and end-to-end learning may not necessarily use RNNs; CNNs or pure attention mechanisms can also be used. ![i0TrDg.jpg](https://s1.ax1x.com/2018/10/20/i0TrDg.jpg) 
*   Some personal understanding of sequence-to-sequence models:
    *   (Information Theory) If the specific implementation forms of the encoder and decoder are not considered, and it is only assumed that the encoder can convert sequence data into an intermediate representation, and the decoder can convert the intermediate representation back into sequence data, then the entire sequence-to-sequence model is equivalent to one round of encoding and decoding of abstract information. Since the dimension of the intermediate representation is much smaller than the total dimension of the encoder, this encoding is lossy. The sequence-to-sequence model aims to make the result of the lossy encoding extract the abstract information from the original text, so the goal of training the network is to let the loss part be the redundant information that is not needed in the abstract. The decoder is equivalent to the inverse process of the encoder, restoring the abstract sequence data from the intermediate representation that contains the abstract information.
    *   (Study and Application) If the entire model is likened to the human brain, then the encoder's hidden layer states are equivalent to the knowledge stored within the brain, while the decoder utilizes this knowledge to solve problems, that is, the encoder is a learning process and the decoder is an application process. This analogy can vividly explain various subsequent improvements to the sequence-to-sequence model: learning at time steps corresponds to learning on the real timeline, with earlier learned knowledge being more easily forgotten in the brain (hidden layer information from earlier time steps is severely lost by the final step), and the brain's capacity being finite (the information storage capacity of the intermediate representation is fixed). Therefore, during learning, we selectively remember and forget information (application of LSTM and GRU), and even highlight key points in the process of memorization (attention mechanism), and can also rely on querying information from the external environment to solve problems without entirely depending on one's own memory (memory networks).
    *   (Circular Recurrent Neural Network) From the perspective of data flow and network structure, the entire sequence-to-sequence model can be regarded as a long RNN with certain time steps limited for input and output, as illustrated in Figure 3-1, the model is an RNN with 8 time steps, where the first 5 time steps have no output, and the last 3 time steps pass the output to the next time step to guide state changes. Joint training of the encoder and decoder is equivalent to training this large RNN. This RNN, which only performs input and output at some nodes, is structurally suitable for handling sequence-to-sequence tasks.

Sequence Loss
-------------

*   The decoder outputs the probability distribution of the dictionary at each step, selecting the word with the highest probability (or performing 束搜索), with the loss being the cross-entropy between the probability at each step and the 01 distribution of the standard word, summed and averaged. In practice, a mask is also applied to address the issue of varying sentence lengths.

Basic Model
===========

*   Preprocessing: For the application of sequence-to-sequence models, certain preprocessing of the data is required, in addition to the commonly used stop word removal and UNK replacement, as well as padding. It also requires the design of start and end symbols for decoding, as follows: ![i0TsbQ.jpg](https://s1.ax1x.com/2018/10/20/i0TsbQ.jpg) 
*   After training word embeddings, perform an embedding lookup operation on the input to obtain features ![i0T6Ej.jpg](https://s1.ax1x.com/2018/10/20/i0T6Ej.jpg) 
*   Feature input encoder receives intermediate representation ![i0Tg5n.png](https://s1.ax1x.com/2018/10/20/i0Tg5n.png) 
*   Obtain the intermediate representation and output the abstract (equivalent to label), input the decoder for decoding ![i0TIrF.png](https://s1.ax1x.com/2018/10/20/i0TIrF.png) 
*   The complete sequence-to-sequence model structure after incorporating the attention mechanism is as follows: ![i0TRCq.jpg](https://s1.ax1x.com/2018/10/20/i0TRCq.jpg) 

Emotional fusion mechanism
==========================

*   The emotional mechanism primarily supplements the emotional features of text, manually constructing a six-dimensional feature through the search of an emotional dictionary, and it is hoped in the future to carry out this work by automatically constructing features.
*   Firstly, train an emotion classifier, filter the original corpus to form an emotional corpus, and test the model on both the emotional corpus and the general corpus ![i0TW80.jpg](https://s1.ax1x.com/2018/10/20/i0TW80.jpg) 
*   Obtain sentiment vectors (i.e., sentiment features) from a dictionary ![i0Tf2V.jpg](https://s1.ax1x.com/2018/10/20/i0Tf2V.jpg) 
*   Directly concatenate the emotional features after the intermediate representation, input decoder ![i0ThvT.jpg](https://s1.ax1x.com/2018/10/20/i0ThvT.jpg) 

Results
=======

*   Results are recorded in the form of ROUGE-F1 values, a comparison of various methods under sentiment corpus ![i0T5KU.png](https://s1.ax1x.com/2018/10/20/i0T5KU.png) 
*   Comparative Study of Sentiment Fusion Schemes under Common Corpus ![i0Tb5R.png](https://s1.ax1x.com/2018/10/20/i0Tb5R.png) 
*   Emotional classification accuracy, as a reference, the previously trained emotional classifier accuracy was 74% ![i0Tob4.png](https://s1.ax1x.com/2018/10/20/i0Tob4.png) 
*   Because it is large corpus with small batch training, only ten iterations were trained, and the effect of the test set in each iteration is ![i0T7VJ.png](https://s1.ax1x.com/2018/10/20/i0T7VJ.png) 

Problem
=======

*   Problem of unknown replacement: Many literature mentions the use of pointer switch technology to solve the rare words (unk) in generated abstracts, that is, selecting words from the original text to replace the unk in the abstract. However, since ROUGE evaluates the co-occurrence degree of words, even if the words from the original text are replaced, regardless of the position or word accuracy, it may result in an increase in ROUGE value, causing the evaluation results to be overestimated. This paper designs comparative experiments and finds that even random replacements without any mechanism can improve the ROUGE value
*   Corpus Repetition Issue: During the examination of the corpus, we found a large number of short texts that are different but have the same abstracts. The repeated corpus comes from different descriptions of the same event or some functional text, such as "...... Wednesday...... Gold prices rose" and "...... Thursday...... Gold prices rose" both generating the same abstract "Gold prices rose." The repetition of short abstracts can cause the model to learn some phrases that should not be solidified at the decoding end. Moreover, if there are repeated identical abstracts in the training set and test set, the solidified abstracts can lead to an artificially high accuracy of generated abstracts. For such texts, this paper conducted four groups of experiments:
    *   Without deduplication: Retain the original text, do not deduplicate, and perform training and testing
    *   De-duplication: Remove all short texts corresponding to duplicate abstracts from the corpus.
    *   Training De-duplication: Partial de-duplication, only removing duplicate data from the training corpus, meaning that the well-trained model is not affected by duplicate text.
    *   Test deduplication: Partial deduplication, only removing the parts of the test set that are duplicated in the training set. Good learning models were affected by duplicate texts, but there was no corresponding data in the test set for the duplicate texts. Under duplicate corpus training, both ROUGE-1 and ROUGE-L exceeded 30, far beyond the normal training models, and after deduplication, they returned to normal levels. The results of the two partial deduplication methods indicate: when training deduplication, models not affected by duplicate corpus did not show significant reactions to the duplicate data in the test set, approximating the normal models with complete deduplication; when testing deduplication, although the model was affected by duplicate corpus, there was no duplicate data in the test set for the model to utilize the learned fixed abstracts, so the results would not be overly high. Moreover, due to learning the pattern of the same abstract corresponding to different short texts, the encoding end actually has a more flexible structure, leading to ROUGE scores higher than those of training deduplication. ![i0TLP1.png](https://s1.ax1x.com/2018/10/20/i0TLP1.png) 

Environmental Implementation
============================

*   Here is the GitHub address: - Abstract\_Summarization\_Tensorflow
*   Ubuntu 16.04
*   Tensorflow 1.6
*   CUDA 9.0
*   Cudnn 7.1.2
*   Gigawords dataset, trained on part of the data, approximately 300,000
*   GTX1066, training time 3 to 4 hours

References
==========

![i0TX26.jpg](https://s1.ax1x.com/2018/10/20/i0TX26.jpg)
![i0TO8x.jpg](https://s1.ax1x.com/2018/10/20/i0TO8x.jpg)

{% endlang_content %}

{% lang_content zh %}

# 任务

- 自动文摘是一类自然语言处理（Natural Language Processin，NLP）任务。对于一段较长的文本，产生一段能够覆盖原文核心意义的短文本，这段短文本就是原文本的摘要。自动文摘技术是指在计算机上构建数学模型，将长文本输入模型，通过计算之后模型能自动生成短摘要。根据需要产生摘要的语料规模和摘要的规模，可以将摘要分为多文本摘要、长文摘要、短文摘要。本文主要研究的是短文摘要：对于一句或者几句文本，生成一句短摘要，概括原文关键信息，且流畅可读，尽量达到人类作者撰写摘要的水平。
- 自动文摘分抽取式和生成式，前者是抽取原文句子构成文摘，后者是通过深度学习模型逐字生成文摘。本文主要研究生成文摘，并将问题抽象成对一个平均长度为28词的长句生成一个平均长度为8词的短句。

![i0TYEd.jpg](https://s1.ax1x.com/2018/10/20/i0TYEd.jpg)

# 预备知识

## 循环神经网络

- 循环神经网络（Recurrent Neural Network，RNN），是神经网络的一种变形，能够有效处理序列型数据，其所有的隐藏层共享参数，每一隐藏层不仅依赖当前时刻的输入还依赖上一时刻的隐藏层状态，数据流并不是如传统神经网络那样在网络层之间传播，而是作为状态在自身网络中循环传递。
- 不展开时形式如下：
  ![i0TtUA.jpg](https://s1.ax1x.com/2018/10/20/i0TtUA.jpg)
- 按时间步展开之后：
  ![i0TN4I.jpg](https://s1.ax1x.com/2018/10/20/i0TN4I.jpg)

## LSTM和GRU

- 循环神经网络能够有效捕捉序列数据的顺序信息且能够构造很深的神经网络而不产生大量需要学习的参数；但也因为参数共享，梯度通过时间步进行链式求导时相当于进行矩阵幂运算，若参数矩阵的特征值过小会造成梯度弥散，特征值过大会造成梯度爆炸，影响反向传播过程，即RNN的长期依赖问题。在处理长序列数据时长期依赖问题会导致序列的长期记忆信息丢失，虽然人们通过引入梯度截断和跳跃链接技术试图缓解此问题，但效果并不显著，直到长短期记忆神经网络和门控循环神经网络作为 RNN 的扩展形式的出现，有效地解决了这个问题。
- LSTM即长短期记忆神经网络。为了捕捉长期记忆信息，LSTM在其节点中引入了门控单元，作为网络参数的一部分参与训练。门控单元控制了当前隐藏层节点记忆（遗忘）过去信息，接受当前输入新记忆的程度。
  ![i0TaCt.jpg](https://s1.ax1x.com/2018/10/20/i0TaCt.jpg)
- GRU即门控神经网络，与LSTM不同的是，GRU 将遗忘门和输入门整合为重置门，遗忘门的门控值和输入门的门控值和为 1，因此 GRU 在 LSTM 的基础上简化了参数，从而使得模型能够更快得收敛。
  ![i0Td8P.jpg](https://s1.ax1x.com/2018/10/20/i0Td8P.jpg)
  
  ## 词嵌入
- 深度学习的一大好处就是能自动学习特征，在自然语言处理中，我们专门用word2vec之类的技术学习词语的特征表示，即词嵌入。
- 词向量（Word Vector）表示，又称词嵌入（Word Embedding），指以连续值向量形式表示（Distributed Representation）词语，而不是用离散的方式表示（One-hot Representation）。在传统的离散表示中，一个词用一个长度为 V 的向量表示，其中 V 是词典大小。向量中只有一个元素为 1，其余元素为 0，元素 1的位置代表这个词在词典中的下标。用离散的方式存储单词效率低下，且向量无法反映单词的语义语法特征，而词向量可以解决以上问题。词向量将向量的维
  度从 V 降低到 $\sqrt[k] V$（一般 k 取 4），每个元素的值不再是 1 和 0，而是连续值。词向量是建立神经网络语言模型 (Neural Network Language Model，NNLM)对语料进行监督学习得到的副产物，此模型的提出基于一个语言学上的假设：具有相近语义的词有相似的上下文，即 NNLM 模型能够在给定上下文环境的条件下求出相应中心词。
- 下图展示了word2vec中的skipgram模型：
  ![i0Twgf.jpg](https://s1.ax1x.com/2018/10/20/i0Twgf.jpg)
- 得到的词嵌入矩阵如下：
  ![i0T0v8.jpg](https://s1.ax1x.com/2018/10/20/i0T0v8.jpg)
- Mikolov等人在NNLM基础上提出了 Word2Vec 模型，此模型进行监督学习的输入输出分别为中心词与上下文（即 Skip Gram 模型) 或者上下文与中心词（即 CBOW 模型）。两种方式均能有效训练出高质量的词向量，但是 CBOW 模型是根据上下文求中心词，训练速度快，适合在大语料上训练；Skip Gram 模型能充分利用训练语料，其本身含义为“跳跃的语法模型”，不仅使用相邻词构成中心词的上下文环境，隔一个词的跳跃词也构成上下文环境，以图 2-1 为例，语料中的一个中心词的上下文包括四个词，假如语料中有 Wtotal 个词语，则 Skip Gram 模型能计算 4 · Wtotal 次损失并进行反向传播学习，对于语料的学习次数是 CBOW 模型的四倍，因此该模型适合于充分利用小语料训练词向量。
- Word2Vec 模型训练完成时，输入层与隐藏层之间的权重矩阵即词嵌入矩阵（Word Embedding Matrix）。离散表示单词的向量与矩阵相乘即得到词向量，这项操作实际上等效于在词嵌入矩阵中查找对应词向量（Embedding Look Up）。词向量能够有效表示词的语义关系，其本质是为机器学习模型提供了一种提取文本信息特征的方法，方便将单词数值化输入模型进行处理。传统的语言模型训练词向量在输出 Softmax 层开销太大，
- 而 Word2Vec 模型采用了分层 Softmax（Hierarchical Softmax）和噪声对比估计（Noise Contrastive Estimation）两种技术，大大加速了训练，使得在自然语言处理中使用大规模语料训练出的优质词向量成为可能。

## 注意力

- 在 NLP 任务中注意力机制最早应用于机器翻译，即引入一个权重矩阵代表编码器序列中各个元素对解码器生成的每一个词的贡献程度。在实际实现时注意力机制会生成一个注意力权重，对各个编码器元素的隐藏层状态进行注意力加权生成中间表示，而不是简单的采用最后一个元素的隐藏层状态。最简单的注意力即解码器对编码器的注意力，分为全局和局部注意力。全局注意力生成对整个编码器序列的注意力权重，而局部注意力先训练出一个注意力对齐位置，再对此位置附近取一个窗口，仅对窗口内的序列加权，使得注意力更加精确、集中。注意力机制带来的一个副产品即词与词之间的对齐Alignment）。在机器翻译中，对齐关系可以理解为词到翻译词的关联程度。在自动文摘中采用注意力机制，可以有效缓解长序列经过编码器编码成中间表示时的信息损失问题。
  ![i0TDKS.jpg](https://s1.ax1x.com/2018/10/20/i0TDKS.jpg)

## 序列到序列

- seq2seq模型，即使用一个RNN作为编码器，编码输入的序列数据得到中间语义表示，再利用另外一个RNN作为解码器，利用中间语义表示得到序列化的输出。广义的序列到序列以及端到端学习不一定使用RNN，也可以用CNN或者纯注意力机制。
  ![i0TrDg.jpg](https://s1.ax1x.com/2018/10/20/i0TrDg.jpg)
- 对于序列到序列模型的一些个人理解：
  - （信息论）如果不考虑编码器和解码器的具体实现形式，只认为编码器可以将序列数据转化为中间表示，解码器可以将中间表示转化为序列数据，则整个序列到序列模型相当于对文摘信息的一次编码解码，由于中间表示的维度远小于编码器的总维度，因此这种编码是有损编码。序列到序列模型要使得有损编码的结果是提取原文中的文摘信息，因此训练网络的目标是让损失的部分为文摘不需要的冗余信息。解码器相当于编码器的逆过程，从包含文摘信息的中间表示中还原出文摘的序列形式数据。
  - （学习与应用）将整个模型比作人类的大脑，则编码器的隐藏层状态相当于大脑内存储的知识，解码器则是利用这些知识解决问题，即编码器是一个学习过程而解码器是一个应用过程。这种比喻能够形象地解释后来对序列到序列模型的各种改进：时间步上的学习对应现实时间线上的学习，越早学习的知识在大脑中越容易忘记（靠前的时间步隐藏层信息传递到最终步时信息丢失严重），且大脑的容量一定（中间表示的信息存储容量一定），因此学习时我们选择性记忆和忘记信息（LSTM 和 GRU 的应用），并且在记忆的过程中划重点（注意力机制），甚至还可以借助向外界环境查询信息，不完全靠自己的记忆解决问题（记忆网络）。
  - （循环神经网络）完全从数据流和网络结构的角度看，整个序列到序列模型可以看作一个长的、限定了某些时间步输入输出的 RNN，以图 3-1 为例，模型就是一个具有 8 个时间步的 RNN，前 5 时间步没有输出，后 3 个时间步将输出传递到下一时间步指导状态变化。编码器和解码器联合训练即相当于对这个大型RNN 进行训练。这种只在部分节点进行输入输出的 RNN 从结构上就适合处理序列到序列任务。

## 序列损失

- 解码器每一步解码出来的实际上时词典的概率分布，取最大概率的词输出（或者做束搜索），损失是每一步的概率和这一步标准词语的01分布做交叉熵，求和再平均。实际上还要应用一个mask来解决句子长度不一定的问题。

# 基本模型

- 预处理：应用序列到序列模型需要对数据进行一定的预处理，除了常用的去停用词加UNK，以及padding之外，还需要设计解码的开始与结束符号，如下：
  ![i0TsbQ.jpg](https://s1.ax1x.com/2018/10/20/i0TsbQ.jpg)
- 训练好词嵌入之后，对输入做一个embedding lookup的操作得到特征
  ![i0T6Ej.jpg](https://s1.ax1x.com/2018/10/20/i0T6Ej.jpg)
- 特征输入编码器得到中间表示
  ![i0Tg5n.png](https://s1.ax1x.com/2018/10/20/i0Tg5n.png)
- 拿到中间表示和输出文摘(相当于label)，输入解码器进行解码
  ![i0TIrF.png](https://s1.ax1x.com/2018/10/20/i0TIrF.png)
- 加入注意力机制后完整的序列到序列模型结构如下：
  ![i0TRCq.jpg](https://s1.ax1x.com/2018/10/20/i0TRCq.jpg)

# 情感融合机制

- 情感机制主要是为文本补充了情感特征，通过查找情感词典的方式手动构造了一个六维的特征，未来希望能够用自动构造特征的方式来进行这方面工作。
- 先训练情感分类器，对原语料进行了筛选形成情感语料，在情感语料和普通语料上都测试了模型
  ![i0TW80.jpg](https://s1.ax1x.com/2018/10/20/i0TW80.jpg)
- 查找词典得到情感向量（即情感特征）
  ![i0Tf2V.jpg](https://s1.ax1x.com/2018/10/20/i0Tf2V.jpg)
- 将情感特征直接拼接在中间表示之后，输入解码器
  ![i0ThvT.jpg](https://s1.ax1x.com/2018/10/20/i0ThvT.jpg)

# 结果

- 结果由ROUGE-F1值形式记录，情感语料下各种方法对比
  ![i0T5KU.png](https://s1.ax1x.com/2018/10/20/i0T5KU.png)
- 普通语料下情感融合方案对比
  ![i0Tb5R.png](https://s1.ax1x.com/2018/10/20/i0Tb5R.png)
- 情感分类准确率，作为参考，之前训练的情感分类器准确率为74%
  ![i0Tob4.png](https://s1.ax1x.com/2018/10/20/i0Tob4.png)
- 因为是大语料小batch训练，只训练了十次迭代，各个迭代的测试集效果
  ![i0T7VJ.png](https://s1.ax1x.com/2018/10/20/i0T7VJ.png)

# 问题

- unk替换问题：在很多文献中都提到了使用指针开关技术解决生成文摘中的罕见词（unk)，即从原文中挑选词语替换掉文摘中的unk，但是由于ROUGE评测的是词的共现程度，因此只要替换了原文的词，哪怕位置、词语不对，都有可能是ROUGE值上升，造成评测结果偏高，本文设计了对比试验，发现即便是没有采用任何机制的随机替换都能提高ROUGE值
  ![i0THa9.png](https://s1.ax1x.com/2018/10/20/i0THa9.png)
- 语料重复问题：在检查语料时我们发现了大量短文本不同但短文摘相同的语料，重复的语料来自于对于相同事件的不同描述，或者一些功能性的文字，翻译过来例如“...... 周三...... 黄金价格上涨”和“......周四......黄金价格上涨”均生成相同的文摘“黄金价格上涨”。短文摘重复的语料会导致模型在解码端学到一些不应该固化的短语，而如果训练集和测试集中有重复的相同的文摘，则固化的文摘反而会导致生成文摘正确率虚高。对于这类文本，本文进行了四组实验：
  - 不去重：保留原文本，不去重，进行训练和测试
  - 去重：删除语料中所有重复文摘对应的的短文本-短文摘对。
  - 训练去重：部分去重，只将训练语料中的重复数据删除，即学习好的模型已经没有受到重复文本的影响。
  - 测试去重：部分去重，只将测试集中与训练集重复的部分删除，学习好的模型受到了重复文本的影响，但测试集中没有与重复文本中相对应的数据。
    重复的语料训练下 ROUGE-1 和 ROUGE-L 都突破了 30，远超正常训练的模型，去重之后便恢复到正常水平。两种部分去重的结果则分别表明：训练去重时，没有受重复语料影响的模型对于测试集中的重复数据没有明显的反应，与完全去重的普通模型近似；测试去重时，虽然模型受到了重复语料影响，但是测试集中没有重复的数据让其利用学习到的固化的文摘，因此结果也不会虚高，且由于学习到了不同短文对应相同文摘的模式，编码端实际上结构更加灵活，导致 ROUGE 指标高于训练去重的结果。
    ![i0TLP1.png](https://s1.ax1x.com/2018/10/20/i0TLP1.png)

# 实现环境

- 这里是github地址：-    [Abstract_Summarization_Tensorflow](https://github.com/thinkwee/Abstract_Summarization_Tensorflow)
- Ubuntu 16.04
- Tensorflow 1.6
- CUDA 9.0
- Cudnn 7.1.2
- Gigawords数据集，训练了部分数据，约30万
- GTX1066，训练时间3到4个小时

# 参考文献

![i0TX26.jpg](https://s1.ax1x.com/2018/10/20/i0TX26.jpg)
![i0TO8x.jpg](https://s1.ax1x.com/2018/10/20/i0TO8x.jpg)

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