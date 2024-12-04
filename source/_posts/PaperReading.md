---
title: Paper Reading 1
date: 2018-03-07 10:20:14
tags:
  - abstractive summarization
  - math
  - machine learning
  -    theory
  -    nlp
categories:
  - ML
author: Thinkwee
mathjax: true
---


*   Opening Work on Attention (Machine Translation)
    
*   Luong attention, global and local attention,
    
*   Opening Work on Attention (Automatic Text Summarization)
    
*   Generative Summary Techniques Collection: LVT, Switching Networks, Hierarchical Attention
    
*   Dialogue System, End-to-End Hierarchical RNN
    
*   Weibo summary, supplement micropoints
    
*   disan, directed transformer, attention mask
    
*   Attention Extractor
    
*   Generative Summary Based on Reinforcement Learning
    
*   w2v, negative sampling
  
<!--more-->

{% language_switch %}

{% lang_content en %}


Neural Machine Translation By Jointly Learning To Align And Translate
=====================================================================

*   Published in 2015.5 (ICLR2015), author Dzmitry Bahdanau.
*   Encoder-decoder model, translation task.
*   The bidirectional GRU serves as the encoder. The encoding hidden layer vectors are composed of bidirectional connections.
*   Different representations are generated for each word.
*   The weight is determined by the vector of the hidden layer of all steps and the vector of the hidden layer of the previous decoding step.
*   Generate weighted representations of the hidden layer vectors for all step encoding. ![i0oB79.png](https://s1.ax1x.com/2018/10/20/i0oB79.png) 

Effective Approaches to Attention-based Neural Machine Translation
==================================================================

*   Published in August 2015, the author (Minh-Thang Luong) used the RNN encoder-decoder model for the translation task.
    
*   During the decoding process, the attentional representation and the decoding hidden layer vector corresponding to the target word are concatenated and then passed through an activation function to generate the attention vector:
    
    $$
    h_t = tanh(W_c[c_t;h_t])
    $$
    
    Afterward, the attention vector is passed through softmax to generate a probability distribution.
    

Global Attention
----------------

*   The article first introduces the global attention model, which is an attention-weighted generation of representations for all encoding hidden layer information, leading to an unpredictable length of alignment vectors (alignment vectors weigh the input information, with the length being the same as the number of words in the input sentence). The model proposed by Dzmitry Bahdanau in the aforementioned text is the global attention model. The global attention model presented in this article is more generalized: it does not use bidirectional RNN concatenation of input vectors but employs a regular RNN instead; it calculates weights directly using the hidden layer vector at the current step, rather than the previous step, thus avoiding complex computations. ![i0oH9P.png](https://s1.ax1x.com/2018/10/20/i0oH9P.png) 
*   Afterward, two effective approaches were introduced, namely local attention and input-feeding.

Local Attention
---------------

*   Local Attention: Instead of using all input information, it first generates an alignment position for each output word, then only generates representations with attention weighted to the input information within the window around the alignment position. The article proposes two methods for generating alignment positions:
    
    *   Monotonic alignment: Simply setting the alignment position of the ith output word to i is obviously not advisable in abstracts.
        
    *   Predictive Alignment: Training Alignment Positions.
        
        $$
        p_t = S \cdots sigmoid(v_p^T tanh(W_ph_t)) \\
        $$
        
        $h_t$ is the hidden layer vector of the t-th generated word, and $W_p$ and $v_p$ are the weights that need to be trained. S is the length of the input word, and when multiplied by the sigmoid, it yields the value at any position in the input sentence
        
*   To maximize the weight of alignment positions, first generate a Gaussian distribution with alignment positions as the expectation and half-window length as the standard deviation, and then generate weights based on this.
    
    $$
    a_t(s) = align(h_t,h_s)exp(-\frac{(s-p_t)^2}{2\sigma ^2})
    $$
    
    ![i0ost1.png](https://s1.ax1x.com/2018/10/20/i0ost1.png)
    

Input-feeding
-------------

*   Input-feeding: When generating alignment, it still needs to rely on past alignments. The actual implementation involves using the attention vector from the previous step as the feed for the next decoding hidden layer. The benefit is that the model can fully understand the previous alignment decisions and creates a very deep network both horizontally and vertically.
*   Experimental results indicate that the use of the prediction alignment-based local attention model performs the best. ![i0oyfx.png](https://s1.ax1x.com/2018/10/20/i0oyfx.png) 

A Neural Attention Model for Abstractive Sentence Summarization
===============================================================

*   Published in September 2015, author Alexander M. Rush, Decoder-Encoder Model, Abstract Task.
*   Proposed an attention encoder using a standard NNLM decoder.
*   Not using RNN, directly using word vectors.
*   Utilize all input information and partial output information (yc) to construct attention weights.
*   Directly weighting the word vector matrix of the smoothed input sentence rather than the RNN hidden layer vector.
*   Model as shown in the figure: ![i0ocp6.png](https://s1.ax1x.com/2018/10/20/i0ocp6.png) 

Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond
=========================================================================

*   Published in August 2016, author Ramesh Nallapati. Encoder-decoder model, using RNN, attention, summarization task.
*   Based on the machine translation model by Dzmitry Bahdanau (bidirectional GRU encoding, unidirectional GRU decoding) for improvement.
*   Improved techniques include LVT (large vocabulary trick), feature-rich encoder, switching generator-pointer, and hierarchical attention.

LVT
---

*   Reduce the size of the softmax layer in the decoder to accelerate computation and convergence. The actual implementation is that the decoder's dictionary is limited to the input text within each mini-batch, and the most frequently occurring words from the decoder dictionary of the previous batch are added to the decoder dictionary of the subsequent batch (until a limit is reached).

Feature-rich Encoder
--------------------

*   Not using simple word vectors that only represent semantic distance, but constructing new word vectors by integrating various semantic features such as entity information, and separately forming vectors to concatenate them. ![i0og1K.png](https://s1.ax1x.com/2018/10/20/i0og1K.png) 

Switching Generator-pointer
---------------------------

*   Resolve the issues of rare words and additional words. The dictionary of the decoder is fixed, and how to deal with words outside the dictionary in the test text. The proposed solution is to add a switch to the decoder, where when the switch is on, it uses its own dictionary to generate the abstract normally, and when the switch is off, it generates a pointer to a word in the input text, copying it into the abstract. ![i0oRXD.png](https://s1.ax1x.com/2018/10/20/i0oRXD.png) ) Switching generator/pointer model When the switch is G, use the traditional method to generate the abstract. When the switch is P, copy words from the input to the abstract.

Hierarchical Attention
----------------------

*   Traditional attention refers to focusing on the positions of key words in sentences, and the hierarchical structure includes the upper level, that is, the positions of key sentences in the text. Two-layer bidirectional RNNs are used to capture attention at both the word level and the sentence level. The attention mechanism runs simultaneously at both levels, with the attention weights at the word level being reweighted and adjusted by the attention weights at the sentence level. ![i0ofne.png](https://s1.ax1x.com/2018/10/20/i0ofne.png) 

Recurrent Neural Network Regularization
=======================================

*   This paper introduces how to use dropout in recurrent neural networks to prevent overfitting
*   Dropout refers to randomly dropping certain nodes of some hidden layers in deep neural networks during each training, while not dropping nodes during testing but multiplying the node outputs by the dropout probability. This method can effectively solve the time-consuming and prone to overfitting problems in deep learning.
*   Two understandings of dropout exist: 1: It forces a neural unit to work together with other randomly selected neural units to achieve good results. It eliminates the weakened joint adaptability between neuron nodes, enhancing generalization ability. 2: It is equivalent to creating some noisy data, increasing the sparsity of the training data, and enhancing the discriminability of features.
*   It is not possible to directly discard hidden layer nodes in RNNs because doing so would lose the long-term dependency information required by RNNs, introducing a significant amount of noise that disrupts the learning process.
*   The author proposes hierarchical node dropping, that is, using a multi-layer RNN structure, even if a node in one layer at a certain time step is lost, the nodes at the same time step in other layers can still pass through, without disrupting the long-term dependency information of the sequence
*   Effect as shown: ![i0ov7j.png](https://s1.ax1x.com/2018/10/20/i0ov7j.png) 

Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models
========================================================================================

*   This paper introduces the construction of a non-goal-driven natural language dialogue system using a fully data-driven hierarchical RNN end-to-end model.
*   Training data consists of triples extracted from movie subtitles, where two speakers complete three expressions in the order of A-B-A.
*   The hierarchical structure is not simply an increase in the number of RNN hidden layers, but rather the construction of two RNNs at the word level and sentence level, as shown in the figure below. The three-part dialogue is contained within two word-level RNN end-to-end systems, and the sentences generated in the middle serve as hidden layer vectors for the higher-level sentence-level RNN.
*   The article employs the bootstrapping technique, using some pre-trained data as the initial values for the model. For example, word embedding matrices are initially trained in large corpora using w2v. In this article, the entire model is even pre-trained, with the principle being to pre-train the entire model using a two-stage dialogue large corpus of a QA system, with the third segment set to empty. In actual training, word embeddings are pre-trained first to complete the initialization of word embeddings, followed by pre-training the entire model for 4 epochs, and finally, the word embeddings are fixed and the entire model is pre-trained to the optimal value.
*   The system infers the third segment from the first two segments of a given three-segment dialogue. Evaluation uses two criteria: word perplexity measures grammatical accuracy, and word classification error measures semantic accuracy.
*   The paper finally summarized the reasons for the occurrence of a bad phenomenon. During the output of the maximum a posteriori probability, some usual answers often appear, such as "I don't know" or "I'm sorry." The author believes there are three reasons: 1: Insufficient data. Because dialogue systems have inherent ambiguity and multimodality, large-scale data is needed to train better results. 2: Punctuation and pronouns occupy a large proportion in dialogue, but the system finds it difficult to distinguish the meanings of punctuation and pronouns in different contextual environments. 3: Dialogues are generally very short, so a triple can provide too little information during inference, resulting in insufficient differentiation. Therefore, when designing natural language neural models, it is best to differentiate semantics and grammatical structures. The author also found that if the maximum a posteriori probability is not used and random output is employed, this problem does not occur, and the inferred sentences generally maintain the topic and appear with special words related to the topic.

News Event Summarization Complemented by Micropoints
====================================================

*   This is a paper from Peking University that uses data from Weibo. The work involves constructing some micropoints in Weibo posts with the same theme to supplement the abstracts extracted from traditional news. The experiment proves that this supplemented abstract can achieve better scores in ROUGE.
*   The focus of the article's exposition is on extracting micropoints rather than on how to integrate micropoints into the original abstract.
*   This team previously proposed two tools: a text clustering model CDW for extracting keywords from news articles; and a Snippet Growth Model for segmenting a blog post into segments (a few sentences each) that possess relatively complete meanings.
*   Micropoints generation main steps: filtering blog posts, categorizing blog posts by topic, segmenting blog posts into fragments, selecting some fragments from blog posts of the same topic to form micropoints, and filtering micropoints.
*   Screening blog posts considers two indicators: relevance and distinctiveness, which must be related to the original news abstract's theme without being too repetitive to cause redundancy. The author uses CDW to extract keywords from the original news from multiple perspectives, calculates the cosine similarity between the blog posts and these keywords to filter out multiple relevant blog posts. Additionally, the author utilizes Joint Topic Modeling (Joint topic modeling for event summarization across news and social media streams) to calculate the distinctiveness between the blog posts and the abstracts. The harmonic mean of the two calculated indicators is taken as the overall screening indicator.
*   Categorize the blog posts by topic: obtain p(topic|tweet) by using LDA with restricted use, then construct a vector v(t) = (p(topic 1 |t), p(topic 2 |t), ..., p(topic n |t)) for each blog post using this conditional probability, and finally use DBSCAN to complete the topic clustering.
*   Using the Snippet Growth Model proposed by the team before, the blog posts are divided into snippets, with the general method being to first take a sentence, then calculate the text similarity, distance measure, and influence measure between other sentences and this sentence to decide whether to add other sentences to the snippet where this sentence is located.
*   A pile of fragments categorized by topic has been obtained, and the next step is to select several fragments that best represent the topic within a single topic's fragments. The method is to pick the few fragments with the smallest average distance to other fragments of the same topic. Since the sentences are not very long, the author represents a fragment with a bag-of-words, where the bag contains all the words that make up all the sentences in the fragment, represented by word vectors. The distance is measured using KL divergence. If the newly selected fragments are too close to the already selected fragments, they are discarded to ensure that the selected fragments still maintain diversity.
*   The obtained fragments will form micropoints, but they need to be filtered before being supplemented into the abstract. The authors propose three indicators: information quantity, popularity, and conciseness. Information quantity refers to the information entropy gain of the abstract after supplementation; the higher the information quantity, the better. Popularity is measured by the number of comments on the original post, with more popular posts being less likely to be extreme. Popularity is preferred to prevent the supplementation of abstracts with extreme or morally incorrect posts. The higher the popularity, the better. Conciseness is described by the ratio of the length of the supplemented part to the length of the original abstract; the smaller the ratio, the more concise the supplementation, and it will not overshadow the original. At this point, the problem is reduced to a discrete optimization problem with constraints under a given conciseness requirement, where each fragment can bring benefits in terms of information quantity and popularity while consuming a certain amount of conciseness. The goal is to select fragments to maximize the benefits, which can be abstracted as a 0-1 knapsack problem and solved using dynamic programming. The authors also set thresholds and use a piecewise function: when popularity exceeds the threshold, the contribution of information gain to the benefit will be greater. This setting is to ensure that the abstract will not be supplemented with fragments where one side has a very high popularity or information gain while the other side is almost non-existent.

DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding
=================================================================================

*   Update, the author later released fast disan, it seems to have modified the calculation of attention, details to be supplemented
*   The author proposes a directed self-attention network that can also perform the encoding task in NLP problems without relying on RNN or CNN structures.
*   The authors believe that among the existing encoders, RNN can capture the sequence information well, but it is slow. The use of a pure attention mechanism (just like the attention weighting of a sequence of word vectors without using RNN in A Neural Attention Model for Abstractive Sentence Summarization) can be used to speed up the operation using existing distributed or parallel computing frameworks, but the sequence information is lost. Therefore, the authors propose a pure attention encoder structure that can capture sequential order information, which combines the advantages of both.
*   The author first proposed three attention concepts: multi-dimensional attention, self-attention, and directed attention.
*   Traditional attention assigns weights to each word in a sentence, with scalar values. In multi-dimensional attention, the weights are vectors, with dimensions matching those of the word vectors. The rationale for using multi-dimensional attention is that it applies attention weighting to each feature of every word. Word vectors inherently have polysemy, and the traditional attention mechanism that weights the entire word vector cannot effectively distinguish between the same word in different contextual environments. Multi-dimensional attention applies weighting to each component of the word vector, allowing for more attention weight to be given to features that can represent the current contextual environment. My understanding is that applying attention weighting to the components of the word vector is equivalent to having slightly different representations of the same word in different contextual environments, which can be used for distinction. The figure below illustrates the difference between traditional attention and multi-dimensional attention. ![i0ojBQ.png](https://s1.ax1x.com/2018/10/20/i0ojBQ.png) On the right is multi-dimensional attention, where the attention weights have become vectors, matching the dimensionality of the input word vectors.
*   The general attention weights are generated with encoding input and a decoding output as parameters, and the weights are related to the current output. Self-attention is unrelated to the decoding end, either replacing the decoding output with each word in the sentence or with the entire input sentence. The former, combined with multi-dimensions, forms token2token attention, while the latter, combined with multi-dimensions, forms source2token.
*   Directed attention involves adding a mask matrix when generating token2token attention, with matrix elements being 0 or negative infinity. The matrix can be upper triangular or lower triangular, representing masks for two directions, for example, from i to j is 0, and from j to i is negative infinity. This gives the attention between words in token2token a direction; attention in the incorrect direction is reduced to 0 after softmax, while attention in the correct direction is unaffected. The mask matrix also has a third type, a non-diagonal matrix, where the diagonal values are negative infinity. This way, a word in token2token does not generate attention for itself. Directed attention is as shown in the figure: ![i0ozAs.png](https://s1.ax1x.com/2018/10/20/i0ozAs.png) 
*   The final architecture of the self-attentional network utilizes the above three types of attention. Firstly, the combination of the upper and lower triangular masks with multi-dimensional token2token generates two self-attention vectors, similar to BLSTM, and then these vectors are connected, passing through a multi-dimensional source2token to produce the final encoded output. The authors tested that this encoding can achieve the best level in natural language prediction and sentiment analysis tasks and can also be used as part of other models for other tasks.

Neural Summarization by Extracting Sentences and Words
======================================================

*   This paper employs a fully data-driven model to accomplish extractive summarization. The model structure consists of a hierarchical text encoder and an attention-based extractor.
*   The difference from the generative attention mechanism summarization lies in: using CNN instead of w2v to construct word embeddings; attention is used to directly extract words rather than to weight and generate intermediate representations.
*   Because this paper uses data-driven extractive summarization, it requires a large amount of extractive summarization training data. Such training data is scarce, so the authors propose a method for generating extractive training data at the word and sentence levels: For sentence extraction, the authors' approach is to convert generative summarization into extractive summarization. First, obtain generative summarization, then compare each sentence in the original text with the generative summarization to decide whether it should be extracted. The comparison criteria include the position of the sentence in the document, the overlap of unigram and bigram grammar, the number of named entities appearing, etc.; for word extraction, the same approach is used to compare the degree of semantic overlap between the generative summarization and the words in the original text to decide whether the word should be extracted. For words that appear in the generative summarization but not in the original text, the authors' solution is to substitute with words that have a similar embedding distance to the original text words to form the training data.
*   During encoding, use CNN to form word embeddings, represent sentences as sequences of word embeddings, and then use RNN to encode at the document level (with one sentence as an input at each time step).
*   When performing sentence extraction, unlike generative models, the dependency of the extracted RNN output is on the previous extracted sentence multiplied by a confidence coefficient, which represents the probability of the previous sentence being extracted.
*   As with generative models, there are differences between train and infer, and the issues that arise during the initial infer phase will accumulate and grow over time. To address this problem, the authors employ a "curriculum learning strategy": initially setting the confidence level to 1 when the training cannot accurately predict, and then gradually restoring the confidence level to the value trained out as the training progresses.
*   Compared to sentence extraction, word extraction is more closely aligned with generative algorithms and can be regarded as a generative summary at the word level under dictionary constraints.
*   Extractions-based abstracts have advantages in handling sparse vocabulary and named entities, allowing the model to check the context and relative position of these words or entities in the sentence to reduce attention weights and minimize the impact of such words.
*   The problem to be addressed in the sampling method is to determine the number of samples. The authors select the three sentences with the highest sampling confidence as the abstract. Another issue is that the dictionary for each batch is generally different. The authors adopt a negative sampling solution.

A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION
=====================================================

*   Using reinforcement learning to optimize the current end-to-end generative summarization model ![i0TicT.png](https://s1.ax1x.com/2018/10/20/i0TicT.png) 
    
*   Addressing the issues of long text summarization and repetitive phrase generation
    
*   Enhanced learning requires external feedback for the model, here the authors use 人工 evaluation of the generated abstracts and provide feedback to the model, enabling it to produce more readable abstracts
    
*   The improvement of the model mainly focuses on two points: internal attention was added to both the encoding and decoding ends, where the encoding end is a previously proposed method, and this paper mainly introduces the internal attention mechanism at the decoding end; a new objective function is proposed, which combines cross-entropy loss with rewards from reinforcement learning
    
*   The inner attention at both ends of encoding and decoding addresses the repetition phrase issue from two aspects, as the repetition problem is more severe in long text summarization compared to short text.
    
*   The addition of inner attention at the encoding end is based on the belief that repetition arises from the uneven distribution of attention over the input long text across different decoding time steps, which does not fully utilize the long text. The distribution of attention may be similar across different decoding time steps, leading to the generation of repetitive segments. Therefore, the authors penalize input positions that have already received high attention weights in the model, ensuring that all parts of the input text are fully utilized. The method of introducing the penalty is to divide the attention weight of a certain encoding input position at a new decoding time step by the sum of attention weights from all previous time steps, so that if a large attention weight was produced in the past, the newly generated attention weight will be smaller.
    
*   The addition of internal attention at the decoding end is based on the belief that repetition also originates from the repetition of the hidden states within the decoding end itself. The authors argue that the information relied upon during decoding should not only include the hidden layer state of the decoding end from the previous time step, but also the hidden layer states from all past time steps, with attentional weighting given. Therefore, a similar internal attention mechanism and penalty mechanism are introduced at the decoding end.
    
*   In this end-to-end model, attention is not a means of communication between the encoding and decoding ends, but is independent at both ends, depending only on the state before and the current state of the encoding/decoding ends, thus it is intrinsic attention (self-attention).
    
*   In constructing the end-to-end model, the authors also adopted some other techniques proposed by predecessors, such as using copy pointers and switches to solve the sparse word problem, encoding and decoding the shared word embedding matrix, and also particularly proposed a small trick: based on observation, repeated three-word phrases generally do not appear in abstracts, so in the 束 search at the decoding end, if a repeated three-word phrase appears, it should be pruned.
    
*   Afterward, the author analyzed two reasons why static supervised learning often fails to achieve ideal results in abstract evaluation criteria: one is exposure bias, where the model is exposed to the correct output (ground truth) during training but lacks a correct output for correction during inference, thus if a word is misinterpreted during inference, the error accumulates increasingly; the other is that the generation of abstracts itself is not static, lacks a standard answer, and good abstracts have many possibilities (these possibilities are generally considered in abstract evaluation criteria), but the static learning method using the maximum likelihood objective function kills these possibilities.
    
*   Therefore, the authors introduced policy learning, a strategy search reinforcement learning method, for the abstracting task beyond supervised learning. In reinforcement learning, the model is not aimed at generating outputs most similar to the labels, but at maximizing a certain indicator. Here, the authors refer to a reinforcement learning algorithm from the image annotation task: the self-critical policy gradient training algorithm:
    
    $$
    L_{rl} = (r(y)-r(y^s))\sum _{t=1}^n log p(y_t^s | y_1^s,...,y_{t-1}^s,x)
    $$
    
    r is the 人工 evaluation reward function; the parameters of the two r functions are: the former is the baseline sentence obtained by maximizing the output probability, and the latter is the sentence obtained by sampling from the conditional probability distribution of each step; the goal is to minimize this L objective function. If the manually awarded sentence obtained from the sampling has more rewards than the baseline sentence, then this minimization of the objective function is equivalent to maximizing the conditional probability of the sampled sentence (after the calculation of the first two r functions, it becomes a negative sign)
    
*   Afterward, the author combines the two objective functions of supervised learning and reinforcement learning:
    

Distributed Representations of Words and Phrases and their Compositionality
===========================================================================

*   Described the negative sampling version of w2v.
    
*   Training with phrases as the basic unit rather than words can better represent some idiomatic phrases.
    
*   Using NCE (Noise Contrast Estimation) instead of hierarchical softmax, NCE approximates the maximization of the logarithmic probability of softmax, as in W2V, we only care about learning good representations, therefore, a simplified version of NCE, negative sampling, is used to replace the conditional probability of the output with the following formula:
    
    $$
    p(w_O | w_I) = \frac {exp(v_{w_O}^T v_{w_I})}{\sum _{w=1}^W exp(v_{w_O}^T v_{w_I})}
    $$
    
    $$ log \\sigma (v\_{w\_O}^T v\_{w\_I}) + \\sum\_{i=1}^k E\[w\_i \\sim P\_n(w)\] \[log \\sigma (v\_{w\_O}^T v\_{w\_I})\]
    
*   Each time, only the target label and k noise labels (i.e., non-target labels) are activated in the softmax output layer, i.e., for each word, there are k+1 samples, 1 positive sample, and k negative samples obtained by sampling, which are then classified using logistic regression. The above expression is the likelihood function of logistic regression, where Pn is the probability distribution of the noise.
    
*   Downsample common words because the vector representation of common words is easy to stabilize; even after several million training iterations, there is little change, so each word's training is skipped with a certain probability:
    
*   The skip-gram model trained in this way has good additive semantic compositionality (the component-wise addition of two vectors), i.e., Russia + river is close to the Volga River, because the vectors are logarithmically related to the probabilities of the output layer, and the sum of two vectors is related to the product of two contexts, which is equivalent to logical AND: high probability multiplied by high probability results in high probability, and the rest is low probability. Therefore, it has this simple arithmetic semantic compositionality.
    

{% endlang_content %}

{% lang_content zh %}

![i0o00J.jpg](https://s1.ax1x.com/2018/10/20/i0o00J.jpg)

# Neural Machine Translation By Jointly Learning To Align And Translate

- 发布于2015.5(ICLR2015)，作者Dzmitry Bahdanau。
- 编码器解码器模型，翻译任务。
- 其中双向GRU做编码器。编码隐藏层向量由双向连接而成。
- 生成每一个单词时有不同的表示。
- 权重由所有步编码隐藏层向量和前一步的解码隐藏层向量决定。
- 对所有步编码隐藏层向量加权生成表示。
  ![i0oB79.png](https://s1.ax1x.com/2018/10/20/i0oB79.png)

# Effective Approaches to Attention-based Neural Machine Translation

- 发布于2015.8，作者（Minh-Thang Luong）使用RNN编码器解码器模型，翻译任务。
- 其中解码时是将包含注意力的表示和目标单词对应的解码隐藏层向量连接再经激活函数生成注意力向量：
  
  $$
  h_t = tanh(W_c[c_t;h_t])
  $$
  
  之后注意力向量过softmax生成概率分布。

## 全局注意力

- 文章先介绍全局注意力模型,即对全部编码隐藏层信息进行注意力加权生成表示,这样会导致对齐向量长度不定(对齐向量对输入信息加权,长度和输入句子的单词数相同).上文中Dzmitry Bahdanau提出的模型即全局注意力模型。本文中的全局注意力模型更为一般化：未使用双向RNN拼接输入向量而是普通的RNN；直接用当前步解码隐藏层向量计算权重，而不是前一步，避免了复杂计算。
  ![i0oH9P.png](https://s1.ax1x.com/2018/10/20/i0oH9P.png)
- 之后引入了两种Effective Approaches，即局部注意力和input-feeding。

## 局部注意力

- 局部注意力：不使用全部输入信息，而是对每一个输出的单词先生成一个对齐位置，然后只对对齐位置附近窗内的输入信息注意力加权生成表示。文章给出了两种种生成对齐位置的方式：
  - 单调对齐：简单的将第i输出单词的对齐位置设为i,显然在文摘中这种方式不可取。
  - 预测对齐：训练对齐位置。
    
    $$
    p_t = S \cdots sigmoid(v_p^T tanh(W_ph_t)) \\
    $$
    
    其中$h_t$是第t个生成单词的隐藏层向量
    $W_p$和$v_p$都是需要训练的权重
    S是输入单词长度,与sigmoid相乘就得到输入句中任意位置
- 另外为了使得对齐位置的权重最大，先以对齐位置为期望、半窗长为标准差生成高斯分布，再以此为基础生成权重。
  
  $$
  a_t(s) = align(h_t,h_s)exp(-\frac{(s-p_t)^2}{2\sigma ^2})
  $$
  
  ![i0ost1.png](https://s1.ax1x.com/2018/10/20/i0ost1.png)

## Input-feeding

- Input-feeding：生成对齐时还需要依赖过去的对齐，实际实现是将上一步的注意力向量作为下一步解码隐藏层的feed，好处在于模型可以完全了解之前的对齐决策，而且在水平层次和垂直层次上创造了一个非常深的网络。
- 实验结果表明使用预测对齐的局部注意力模型表现最好。
  ![i0oyfx.png](https://s1.ax1x.com/2018/10/20/i0oyfx.png)

# A Neural Attention Model for Abstractive Sentence Summarization

- 发布于2015.9，作者Alexander M. Rush，解码器编码器模型，文摘任务。
- 提出了一种注意力编码器，使用普通的NNLM解码器。
- 未使用RNN，直接用词向量。
- 使用全部输入信息,局部输出信息(yc)构建注意力权重。
- 直接对平滑化的输入句子的词向量矩阵加权而不是RNN隐藏层向量。
- 模型如下图:
  ![i0ocp6.png](https://s1.ax1x.com/2018/10/20/i0ocp6.png)

# Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond

- 发布于2016.8，作者Ramesh Nallapati。编码器解码器模型，使用RNN，注意力，文摘任务。
- 基于Dzmitry Bahdanau的机器翻译模型（双向GRU编码，单向GRU解码）进行改进。
- 改进包括LVT（large vocabulary trick)、Feature-rich encoder、switching generator-pointer、分层注意力。

## LVT

- 减少解码器的softmax层大小，加速计算，加速收敛。实际实现是在每一个mini-batch中解码器的词典只限于本batch内的输入文本，而且把之前batch内解码词典中最频繁的单词加入之后batch的解码词典（直至达到一个上限）。

## Feature-rich Encoder

- 不使用简单的只表示语义距离的词向量，而是构建包含了实体信息等多种语义特征，分别构成向量并拼接起来形成新的词向量
  ![i0og1K.png](https://s1.ax1x.com/2018/10/20/i0og1K.png)

## Switching Generator-pointer

- 解决罕见词，额外词问题。解码器的词典是固定的，如果测试文本中包含词典外的单词该如何解决。其提供的解决方案是给解码器加上一个开关，开关打开时就普通的使用自己词典生成，开关关上时，就产生一个指针指向输入文本中的一个单词，并将其复制入文摘。
  ![i0oRXD.png](https://s1.ax1x.com/2018/10/20/i0oRXD.png))
  Switching generator/pointer model
  开关为G时就用传统方法生成文摘
  开关为P时就从输入中拷贝单词到文摘中

## 分层注意力

- 传统的注意力是指关注句子中的关键词位置，分层还包括上一层，即文本中的关键句位置。使用两层双向RNN分别在词层次和句层次捕捉注意力。注意力机制同时运行在两个层次之上，词层次的注意力权重会被句层次的注意力权重重新加权调整。
  ![i0ofne.png](https://s1.ax1x.com/2018/10/20/i0ofne.png)

# Recurrent Neural Network Regularization

- 本文介绍了如何在循环神经网络中使用dropout来防止过拟合
- Dropout是指在深度神经网络当中，在每次训练时随机丢掉某些隐藏层的某些节点，测试时不丢弃节点但是将节点输出乘以丢弃概率。这种方法可以有效解决深度学习费时且容易过拟合的问题。
- 对于dropout的理解有两种，1：它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。2：它相当于创造了一些噪声数据，增加了训练数据的稀疏性，增加了特征的区分度。
- 在RNN中不能直接按时间步丢弃隐藏层节点，因为这样会丢失RNN所需要的长期依赖信息，引入很大的噪声破坏了学习过程。
- 作者提出的是按层丢弃节点，即使用多层RNN结构，即便丢失了一层中的某一时间步的节点，其他层的相同时间步的节点也能传递过来，不破坏序列的长期依赖信息
- 效果如图:
  ![i0ov7j.png](https://s1.ax1x.com/2018/10/20/i0ov7j.png)

# Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models

- 本文介绍了使用完全数据驱动的分层RNN端到端模型来构建non-goal-driven的自然语言对话系统。
- 训练数据是从电影字幕中提取的triples，即两个对话人按A-B-A的顺序完成三次表达。
- 分层的结构不是简单的增加RNN隐藏层层数，而是分别在词水平和句水平构建两个RNN，如下图。
  ![i0TC90.png](https://s1.ax1x.com/2018/10/20/i0TC90.png)
  三段式对话包含在两个词水平RNN端到端系统中
  中间生成的句表示又作为更高层次即句水平RNN的隐藏层向量
- 文章中使用了bootstrapping的技巧，即用一些预先训练好的数据作为模型的初始值。例如用w2v在大语料库中训练好词嵌入矩阵的初始值。本文中甚至将整个模型都预训练好了，原理是使用QA系统的二段式对话大语料库预训练整个模型，第三段设为空。在实际训练中，先预训练词嵌入完成词嵌入的初始化，然后预训练整个模型4个epoch，最后固定词嵌入不变接着预训练整个模型到最佳值。
- 系统是给定三段式对话中的前两段推测出第三段。Evaluation使用两种标准：word perplexity衡量语法准确度，word classification error衡量语义准确度
- 论文最后总结了一种不好的现象出现的原因。在最大后验概率输出时经常出现一些通常的回答，例如”i don’t know”或者”i’m sorry”。作者认为原因有三种：1：数据太少。因为对话系统是有内在歧义性和多模态性的，大规模数据才能训练出较好的结果。2：标点符号和代词在对话中占有很大比例，但是系统很难区分不同上下文环境中标点符号和代词的意义。3：对话一般很简短，因此infer时一个triple可以提供的信息太少。区分度不够。因此设计自然语言神经模型时应尽量区分语义和语法结构。作者还发现，如果不用最大后验概率而是随机输出时不会出现此问题，且infer出来的句子一般能保持话题，出现一个与话题相关的特殊词。

# News Event Summarization Complemented by Micropoints

- 这是来自北大的一篇论文，使用微博上的数据，其工作是给定从传统新闻中提取出的文摘，在相同主题的微博博文中构造出一些micropoints来补充文摘，实验证明这种补充的文摘能在ROUGE中取得更好分数。
- 文章的阐述重点在于提取micropoints而不是如何将micropoints整合进原始文摘。
- 这个团队之前提出了两种工具：用于提取新闻关键词的一种文本聚类模型CDW；用于将一段博文分割成各自拥有相对完整含义的片段（几句话）的Snippet Growth Model。
- Micropoints生成的主要步骤：筛选博文、将博文按主题分类、将博文分割成片段、从同一主题的博文片段中挑选一些组成micropoints、对micropoints进行筛选。
- 筛选博文考虑两个指标：相关性和差异性，既要与原新闻文摘主题相关，又不能太重复而导致冗余。作者使用CDW提取原新闻多个角度的关键词，并将博文与这些关键词计算cos相似度，以筛选多个具有相关性的博文。另外作者利用Joint Topic Modeling(Joint topic modeling for event summarization across news and social media streams)计算博文和文摘的差异性。将计算出的两个指标取调和平均作为总体筛选的指标。
- 将博文按主题分类：受限使用LDA得到p(topic|tweet)，再利用此条件概率为每一个博文构造向量v(t) = (p(topic 1 |t), p(topic 2 |t), ..., p(topic n |t))，最后使用DBSCAN完成主题聚类。
- 使用团队之前提出的Snippet Growth Model将博文分成片段，大致方法是先取一个句子，然后计算其他句子与这个句子之间的文本相似度、距离度量、影响度量来决定是否将其他句子加入这个句子所在的片段当中。
- 现在已经得到了按主题分类的一堆片段，之后需要在一个主题的片段中挑一些最能代表本主题的片段出来。其方法是挑选到同主题其他片段平均距离最小的前几个片段。因为句子不太长，作者将一个片段用词袋表示，词袋中装了组成本片段中所有句子的所有词，用词向量表示。距离用KL散度衡量。如果新挑出来的片段与已挑片段距离太近则放弃，以保证挑选出来的片段依然具有差异性。
- 得到这些片段将组成micropoints，但是在补充进文摘之前还要进行筛选。作者提出了三个指标：信息量，流行度，简洁性。信息量即补充进文摘之后文摘获得信息熵增益，信息量越大越好；流行度用原博文评论数来衡量，越流行的博文越不容易极端，用流行度来防止极端、三观不正的博文被补充进文摘。流行度越大越好；简洁性用补充的部分和原文摘的长度比来描述，越小表示补充的越简洁，不会喧宾夺主。此时问题就化成在给定简洁性要求下，每一个片段能带来信息量和流行度的收益，同时耗费一定的简洁性，选择片段使收益最大，即离散的有限制条件的最优化问题，可以抽象为0-1背包问题，用动态规划解决。作者还设置了阈值，使用分段函数：当流行度大于阈值时，信息增益对于收益的贡献会更大。这样设置是为了保证不会因为出现流行度和信息增益一方很大而另一方几乎没有的片段被加入文摘。

# DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding

- 更新，作者后来又推出了fast disan，貌似是改了注意力的计算，细节待补充
- 作者提出了一种有向自注意网络，不依赖RNN或者CNN结构也能很好的完成NLP问题中的编码任务。
- 作者认为现有的编码器中，使用RNN能很好的捕捉序列的顺序信息但是慢；使用纯注意力机制（就如同A Neural Attention Model for Abstractive Sentence Summarization中不使用RNN而是直接对词向量序列进行注意力加权）虽然能利用现有的分布式或者并行式计算框架加速运算，却丢失了序列的顺序信息。因此作者提出了一种能捕捉序列顺序信息的纯注意力编码器结构，结合了两者的优点。
- 作者首先提出了三种注意力概念：多维度注意力，自注意力，有向注意力。
- 传统的注意力是对一个句子中各个单词加权，权值是标量。而多维度注意力中权值是向量，维度和词向量维度相同。使用多维度的理由在于这样是对每一个词的每一个特征进行注意力加权，词向量本身具有一词多义性，传统的对整个词向量进行加权的注意力机制对同一个词在不同上下文环境中的情况不能很好区分，多维度是对词向量的每一个分量加权，它可以给能表示当前上下文环境的特征更多的注意力权重。我的理解是对词向量的分量进行注意力加权，这样相当于同一个词在不同的上下文环境中有略微不同的表示，可以用来区分。下图是传统注意力与多维度注意力的区别。
  ![i0ojBQ.png](https://s1.ax1x.com/2018/10/20/i0ojBQ.png)
  右边是多维度注意力
  可以看到注意力权重变成了向量，与输入词向量维度数相同
- 一般的注意力权重是编码输入和一个解码输出作为参数生成，权重与当前输出有关。自注意力与解码端无关，要么用本句子中的每一个词替代解码输出，要么用整个输入句子替代解码输出。前者与多维度结合形成token2token注意力，后者与多维度结合形成source2token。
- 有向注意力即在生成token2token注意力时根据需要添加一个掩码矩阵，矩阵元素为0或者负无穷。矩阵可以为上三角或者下三角，代表两个方向的掩码，例如从i到j是0，从j到i是负无穷，则在token2token中词之间的注意力就有了方向，不正确方向的注意力过softmax之后会降到0，而正确方向的注意力不受影响。掩码矩阵还有第三种，无对角线矩阵，即对角线上的值为负无穷，这样token2token中一个单词对自己不会产生注意力。有向注意力如下图：
  ![i0ozAs.png](https://s1.ax1x.com/2018/10/20/i0ozAs.png)
- 最后有向自注意网络的构成利用了以上三种注意力，首先上下三角两种掩码搭配多维度token2token可以产生前向后向两个自注意力向量，类似blstm，然后将向量连接，过一个多维度source2token产生最终的编码输出。作者测试了这种编码能在自然语言推测和情感分析任务中达到最佳水平，也可以作为其他模型的一部分在其他任务中使用。

# Neural Summarization by Extracting Sentences and Words

- 本文是使用了完全数据驱动的模型来完成抽取式文摘。其模型结构由一个分层文本编码器和一个注意力机制抽取器构成。
- 与生成式注意力机制文摘不同的地方在于：使用CNN而不是w2v构建词嵌入；注意力用来直接抽取词而不是加权生成中间表示。
- 因为本文是用数据驱动的抽取式文摘，所以需要大量的抽取式文摘训练数据，这样的训练数据很少，作者提出了制造词水平和句水平的抽取式训练数据的方法：对于句抽取，作者的思路是将生成式文摘转换成抽取式文摘，首先获得生成式文摘，然后将原文中每一句与生成式文摘对比以决定是否应该抽取出来，对比的指标包括句子在文档中的位置，一元语法和二元语法重叠性，出现的命名实体数量等等；对于词抽取，同样也是对比生成式文摘和原文中词的词义重叠程度来决定该词是否应该抽取出来。对于那些生成式文摘中出现而原文中没有的词，作者的解决方案是用原文中词嵌入距离相近的词替代形成训练数据。
- 编码时，用CNN形成词嵌入，将句子表示为词嵌入序列，再用RNN形成文档层次上的编码（一个句子为一个时间步输入）。
- 句抽取时，与生成式不同，抽取的RNN输出依赖是上一个抽取生成的句子乘以一个置信系数，这个置信系数代表上一个句子有多大可能被抽取出来。
- 与生成式一样，train和infer存在差异，前期抽取infer出现的问题会在后期越滚越大。为了解决这个问题，作者使用了一种“课程学习策略”：训练刚开始置信水平不能准确预测，就设为1，之后随着训练进行逐渐将置信水平恢复成训练出来的值。
- 与句抽取相比，词抽取更加贴近生成式算法，可以看成是词典受限的词水平上的生成式文摘。
- 抽取式文摘在处理稀疏词汇和命名实体上有优势，可以让模型检查这些词汇或实体的上下文、句中相对位置等来降低注意力权重，减少这类词影响。
- 抽取式要解决的一个问题是决定抽取数量。作者取抽取置信水平最高的三句作为文摘。另一个问题是每一个batch的词典一般是不同的。作者采用了一种负采样的解决方案。

# A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION

- 使用强化学习来优化当前的端到端生成式文摘模型
  ![i0TicT.png](https://s1.ax1x.com/2018/10/20/i0TicT.png)
- 解决长文摘生成和重复短语问题
- 强化学习需要外界给予模型反馈，这里作者使用人工对生成的文摘进行评价并反馈给模型，使得模型可以生成可读性更好的文摘
- 模型的改进主要在两点：在编码端和解码端分别加入了内注意力，其中编码端是之前提出过的，本文主要引入解码端的内注意力机制；提出了一种新的目标函数，结合了交叉熵损失和来自强化学习的奖励
- 编码解码两端的内注意力是从两个方面解决重复短语问题，因为重复问题在长文本生成文摘中相比短文本更加严重。
- 编码端加入内注意力是认为重复来自于在解码各个时间步对输入长文本的注意力分布不均匀，没用充分利用长文本，在解码各个时间步可能注意力的分布都相似，导致生成重复语段。因此作者在模型中对过去已经获得高注意力权重的输入位置给予惩罚，保证输入文本的各个部分充分利用。引入惩罚的方式是在新的某一解码时间步，某一编码输入位置的注意力权重是本次产生的注意力权重除以之前所有时间步注意力权重之和，这样如果过去的产生了大的注意力权重，则新产生的注意力权重会变小。
- 解码端加入内注意力是认为重复还来源于解码端本身的隐藏状态重复。作者认为解码时依赖的解码端信息应该不止包含上一时间步的解码端隐藏层状态，而是过去所有时间步的隐藏层状态并给予注意力加权，因此在解码端引入了类似的内注意力机制和惩罚机制。
- 在这个端到端模型中注意力并不是沟通编码端和解码端的方式，而是独立在两端，仅依赖于编码/解码端之前的状态和当前状态产生，因此是内注意力（自注意力）。
- 在搭建端到端模型时作者还采用了一些其他前人提出过的技巧，例如使用复制指针和开关解决稀疏词问题，编码解码共享词嵌入矩阵，另外还特别提出了一个小技巧：基于观察，一般文摘中不会出现重复的三词短语，因此在解码端束搜索中若出现了重复的三词短语就剪枝。
- 之后作者分析了静态的监督学习在文摘评价标准中常常取不到理想结果的两个原因：一是exposure bias，即模型在训练时是接触到了正确的输出(ground truth)，但是在infer时是没有正确输出做矫正的，因此如果infer时错了一个词，之后错误会越积越大；二是文摘生成本身不是静态的，没有标准答案，而好的文摘有许多种可能（文摘评价标准中一般考虑了这些可能），但使用最大似然目标函数的静态的学习方法扼杀了这些可能。
- 因此作者在监督学习之外为文摘任务引入了policy learning，一种策略搜索强化学习方式。在强化学习中模型不是以生成与标签最相似的输出为目标，而是以最大化某一种指标为目标。在这里作者借鉴了图像标注任务中的一种强化学习算法：self-critical policy gradient training algorithm：
  
  $$
  L_{rl} = (r(y)-r(y^s))\sum _{t=1}^n log p(y_t^s | y_1^s,...,y_{t-1}^s,x)
  $$
  
  r是人工评价奖励函数
  两个r函数的参数：前者是最大化输出概率得到的基准句子，后者是根据每一步输出条件概率分布采样得到的句子
  目标是最小化这个L目标函数，假如采样的句子得到的人工奖励比基准句子多，则这个最小化目标函数等价于最大化采样句子的条件概率（前面的两个r函数计算之后为负号）
- 之后作者将监督学习和强化学习的两种目标函数结合起来：

# Distributed Representations of Words and Phrases and their Compositionality

- 介绍了w2v的负采样版本。
- 以短语为基本单位训练而不是单词，这样能够更好的表示一些idiomatic phrase。
- 用nce（噪声对比估计）替代分层softmax，nce近似最大化softmax的对数概率，在w2v中只关心学到好的表示，因此用简化版的nce，负采样，用下式替代输出的条件概率：
  
  $$
  p(w_O | w_I) = \frac {exp(v_{w_O}^T v_{w_I})}{\sum _{w=1}^W exp(v_{w_O}^T v_{w_I})}
  $$
  
  被替换成
  $$
  log \sigma (v_{w_O}^T v_{w_I}) + \sum _{i=1}^k E_{w_i \sim P_n(w)} [log \sigma (v_{w_O}^T v_{w_I})]
- 每次在softmax输出层只激活目标label和k个噪声label（即非目标label），即对每一个单词，有k+1个样本，1个正样本，k个采样得到的负样本，进行logistic回归分类，上式即logistics回归的似然函数，其中Pn是噪声的概率分布。
- 对常见词进行降采样，因为常见词的向量表示容易稳定，再训练几百万次也不会发生大的改变，因此每一次对词的训练以一定概率跳过：
- 这样训练出来的skip-gram模型具有很好的加法语义合成性（两个向量的逐分量相加），即俄国+河与伏尔加河相近，因为向量与输出层的概率对数相关，两个向量相加与两个上下文的乘积相关，这种乘积相当于逻辑与：高概率乘高概率为高概率，其余为低概率。因此具有这种简单的算术语义合成性。

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