---

title: Prompt - Task Reformulation in NLP
date: 2021-05-13 19:36:34
tags:

- pretrained language model
- pet
- machine learning
- few shot
- nlp
categories:
- NLP
html: true
mathjax: true

---


<img src="https://i.mji.rip/2025/07/16/ea5f8ed907cd529450d9e9793f9aa94a.png" width="500"/>


- Record of recent task reconstruction methods based on templates, a particularly interesting direction since the appearance of GPT-3. These methods typically design prompts for tasks, converting samples and tasks into natural language templates, which are then directly input into pre-trained language models to generate text, thereby indirectly completing the tasks. The construction of prompts standardizes the form of downstream tasks and pre-trained tasks (language models), achieving good results in few-shot learning. Key papers to read include the following nine:
  - Early work that converts questions into natural language and uses pre-trained language models for answers:
    - (Harvard) Commonsense Knowledge Mining from Pretrained Models
    - (Heidelberg) Argumentative Relation Classification as Plausibility Ranking
    - (NVIDIA) Zero-shot Text Classification With Generative Language Models
  - The PET approach, Pattern Exploiting Training:
    - (LMU) Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference
    - (LMU) It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners
    - (UNC) Improving and Simplifying Pattern Exploiting Training
  - Automatically constructing prompts, Automatically Searching Prompts:
    - (UCI, UCB) AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts
    - (Princeton, MIT) Making Pre-trained Language Models Better Few-shot Learners
    - (THU) GPT Understands, Too
<!--more-->

{% language_switch %}

{% lang_content en %}
# Commonsense Knowledge Mining from Pretrained Models

- The authors aim to mine commonsense knowledge from data with unknown distributions, whereas traditional supervised learning methods are easily influenced by the data distribution in the training set, leading to biased results.
- They convert relation triples into masked sentences and input them into BERT. The mutual information between the predicted results from BERT and the triples is used to rank the credibility of the triples.
- The task involves scoring a given triple to determine its likelihood of representing real-world knowledge. This is divided into two steps:
  - Converting the triple into a masked sentence: multiple templates were manually designed for each relationship, and a set of rules was designed to ensure grammatical correctness (singular/plural, inserting articles, modifying nouns, etc.). All combinations of these templates and rules form a series of candidate sentences, which are then input into a pre-trained unidirectional language model to compute the log-likelihood score of each sentence being grammatically correct.
  - Inputting the generated sentences into BERT for scoring: here, the authors use conditional pointwise mutual information (PMI), where the mutual information between the head and tail entities, conditioned on the relation r, serves as the score:
    
    $$
    PMI(tail, head | relation) = \log p(tail | head, relation) - \log p(tail | relation)
    $$

    In the language model, this effectively means masking the tail and predicting it. The first term in the equation masks only the tail, while the second term also masks the head (no prediction for the head). Additionally, if entities are composed of multiple words, a greedy approximation is used. Initially, all words are masked, and the highest-probability word is unmasked, followed by iterative predictions for the remaining words, where each time the highest-probability word is restored. The product of these probabilities gives the conditional probability of the entire word. The equation is asymmetric, so the authors also compute the head entity probability based on the relationship and tail entity, and average the two PMI values as the final result.
- The final results, although not as good as supervised learning, achieved the best results in unsupervised learning.
- This is an early attempt to use pre-trained models for Mask Predict, where the task is framed as a Cloze task. The patterns here are still manually designed (with a set of rules designed for each relation).

# Argumentative Relation Classification as Plausibility Ranking

- The task in this paper is Argumentative Relation Classification, i.e., text pair classification, where the goal is to distinguish whether a pair of texts supports or contradicts a given (or implicit) conclusion. In positive text pairs, both texts support the conclusion, while in negative pairs, one supports and the other contradicts the conclusion.
- For this interesting task, the authors propose a similarly interesting approach: using a Siamese network for ranking, where the ranking is based on the plausibility (credibility) of the constructed text. And what is this constructed text? Quite simple: the two sentences to be classified are connected by a conjunction to form the constructed text:
  - Positive example: Text A, and Text B
  - Negative example: Text A, however, Text B
- If Text A and Text B are contradictory, the credibility of the negative example is high. If Text A and Text B support each other, the credibility of the positive example is high.
- The next step involves using a pre-trained language model as the encoder in the Siamese network for ranking.
- The core idea here is to transform the task into natural language and use the language model, which has learned general knowledge about natural text, to perform the task and make predictions indirectly.
- Similar to the previous paper, the key here is to convert the task into natural language (a template) and cleverly use pre-trained language models to indirectly complete the task (by completing the constructed task).

# Zero-shot Text Classification With Generative Language Models

- The authors use GPT to transform the text classification task into a natural language question by combining the original text with the category, and indirectly determine the category through text generation.
- The main advantage, as highlighted in the title, is zero-shot learning, which allows the model to generalize to categories that do not exist in the training data.
- Specifically, the text classification problem is turned into a multiple-choice QA task, where the options are formulated into a question: "Which category does this text belong to? A; B; C; D..." and then the text to be classified is appended. The goal is to train the language model to directly generate the correct category as text.
- To minimize the gap between pre-training and fine-tuning, the authors introduce a pre-training task, called title prediction pretraining, where all candidate titles are concatenated with the main text and the correct title is generated.
- This is a very intuitive, indirect, and bold use of language models for classification tasks, where the language model directly generates category labels.
  ![gW98oV.png](https://z3.ax1x.com/2021/05/17/gW98oV.png)
- The final zero-shot results, while not as good as fine-tuning or state-of-the-art models, show a strong generalization ability, outperforming the random and majority baseline models. The key takeaway is how the language model is used creatively to solve the classification task by designing multiple-choice questions.

# Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference

- This paper formally introduces the concept of PET: Pattern-Exploiting Training.
- In the previous three papers, we see that many NLP tasks can be completed unsupervised or indirectly by providing natural language descriptions of the tasks through language models. However, these methods still fall short compared to supervised learning methods.
- PET offers a semi-supervised learning approach that successfully outperforms supervised models in low-resource settings.
- The principle of PET is explained in a single diagram:
  ![gWivmd.png](https://z3.ax1x.com/2021/05/17/gWivmd.png)
  - The authors introduce two concepts: pattern, which transforms the input text into a masked Cloze text based on the task, and verbalizer, which maps the predicted masked words from the language model to labels. Each pattern corresponds to a verbalizer, forming a PvP (pattern-verbalizer pair).
  - The PET process is divided into three steps:
    - First, use PvP to fine-tune the pre-trained language model on a small training set.
    - Second, for each task, multiple PvPs can be designed to create different models through fine-tuning, and then a soft label is assigned to unannotated data using these models.
    - Third, a classifier is trained on the labeled data to complete supervised learning.
- In the second step, there are two small details: ensemble learning with multiple classifiers (adding the predicted label distributions, which can be equally weighted or weighted based on zero-shot performance in the training set) and using soft labels (probability distributions) when labeling the data, with softmax applied at temperature T=2. These two techniques help better leverage the knowledge from the language model, one through ensemble robustness and the other through knowledge distillation.
- The authors also introduce iPET, a traditional semi-supervised learning approach that iterates between labeling and training, using increasing amounts of data and different generations of models to improve performance.
- The advantage of this semi-supervised framework is that the final operation is still supervised learning, achieving high accuracy, while reducing the uncertainty introduced by the language model through knowledge distillation (soft labels).

# It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

- The original PET team has another paper where the motivation is that small models can also achieve results comparable to large models like GPT-3 in few-shot learning when using PET, promoting environmental sustainability.
- In this paper, the authors extend the prediction of masked words in PvP to multiple masks, inserting a fixed maximum number of masks during training and then performing post-processing during prediction.
- They provide more extensive experimental results, which, while still in preprint form (not yet published in conferences...), later won the NAACL 2021 Outstanding Paper Award.

# Improving and Simplifying Pattern Exploiting Training

- [![gIih5j.png](https://z3.ax1x.com/2021/05/19/gIih5j.png)](https://imgtu.com/i/gIih5j)

- This paper improves PET by further simplifying the design of the pattern-verbalizer pair and reducing the number of patterns needed to achieve few-shot learning in a broad set of tasks.
- The simplification helps lower the entry barrier for researchers and developers by making it easier to implement this framework with minimal effort.
  
# AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts
- From the work introduced above, it can be seen that constructing effective text to trigger language models to generate results is crucial, which means constructing the prompt. Currently, all prompts are manually constructed, but later a series of works emerged attempting to automatically construct prompts.

- This work cannot really be considered as prompts; a more accurate term would be "trigger words sequence," because it essentially applies a method for generating adversarial text samples to the task of constructing prompts.

- Specifically, it draws on two papers: *HotFlip: White-box adversarial examples for text classification* and *Universal Adversarial Triggers for Attacking and Analyzing NLP*. The idea is to concatenate a sequence of trigger words into the sample, which can lead the model to make incorrect predictions. The search for trigger words in the model primarily uses the HotFlip method:
  
  - Initialize the trigger word $$\mathbf{e}_{adv}$$ (e.g., words like the, a, an), then pass the model forward to obtain the gradient of the loss with respect to the trigger word embedding $$\nabla_{\mathbf{e}_{adv}} \mathcal{L}$$. Note that the label used for loss calculation should be the incorrect label that the model is intended to be fooled into predicting (i.e., the label after fooling the model).
  
  - We aim to replace the $i$-th trigger word with a word $$\mathbf{e}_{i}$$ such that the loss is minimized most significantly after replacement, meaning the model is most likely to predict the wrong label. Therefore, the word we are looking for is $$ \underset{\mathbf{e}_{i}^{\prime} \in \mathcal{V}}{\arg \min } \mathcal{L}(\mathbf{e}_{i}^{\prime}) $$, where a first-order Taylor expansion is used for approximation. We need to compute the gradient of the loss with respect to the token. Since token embedding lookup is not differentiable, we need to compute the gradient of the embedding of a specific token.

  $$
  \mathcal{L}(\mathbf{e}_{i}^{\prime})    =  \mathcal{L}(\mathbf{e}_{adv_{i}}) + \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{adv_{i}}\right]^{\top} \nabla_{\mathbf{e}_{adv_{i}}} \mathcal{L} 
  $$

  $$
  \propto \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{adv_{i}}\right]^{\top} \nabla_{\mathbf{e}_{adv_{i}}} \mathcal{L} 
  $$

  - This results in the first trigger word in the first round of iteration. Then, through beam search, the remaining trigger words are generated, iterating multiple times to eventually obtain a sequence of trigger words that can be used to attack the model.

- The above describes the HotFlip method for text adversarial attacks. Its essence is to generate trigger words and append them to the sample to make the model predict an incorrect label. The idea of autoprompt is to generate trigger words to make the model predict a specified label.
  
  [![ghFDuF.md.png](https://z3.ax1x.com/2021/05/18/ghFDuF.md.png)](https://imgtu.com/i/ghFDuF)


- Now it becomes simpler. The authors first used the HotFlip method to generate trigger words for each task in the training set, then used a template to transform the sample into a sentence. As shown in the figure, the sentence is concatenated with the trigger word sequence ([T]) and the mask position ([P]) that the PLM is to predict. The model then predicts the word, and the label is obtained through post-processing. The specific post-processing operation involves summing the probabilities of the predicted words for each label, and finally normalizing these sums to get the probability for each label.

- The above only explains the automatic prompt construction method in PvP. As for the verbalizer, i.e., the mapping from predicted words to labels, the authors also propose an automatic search method:

  - After encoding the PLM and obtaining the embedding of the mask token containing contextual information, this embedding is used as a feature input, and the label is used as the output to train a logistic classifier. Then, the PLM-encoded embedding of each word is fed into this classifier to obtain a score for each word on each label. The top k words with the highest scores are chosen for each label as the mapped word set. Essentially, this compares the mask token's encoded embedding (needed for predicting the label) with the embeddings of each word, selecting the top k closest words, but using a logistic classifier to apply class-related feature weighting. This is not merely based on the semantic similarity of the PLM encoding but is a very clever method.

# Making Pre-trained Language Models Better Few-shot Learners

- The title of this paper is essentially the title of GPT-3 with "better" added, emphasizing how to better use prompts for few-shot learning.
- A training framework is proposed: prompt-based fine-tuning + automatic prompt generation + dynamically and selectively integrating task descriptions into prompts, all of which are strongly task-agnostic. Let's now go through these three improvements in detail.
- [![g58yOe.png](https://z3.ax1x.com/2021/05/19/g58yOe.png)](https://imgtu.com/i/g58yOe)

- The image above clearly demonstrates the first improvement: prompt-based fine-tuning. As we can see, compared to previous prompt-based methods, in addition to the input and prompt, the input is also concatenated with a description for each label.
- As for automatic prompt generation, it is divided into two parts:
  - How to automatically generate the mapping from target words to labels given a template. Here, the author iterates over the results of a pre-trained language model (PLM). First, for each class, all the training samples of this class are identified, and the probability distribution of the masked word is inferred using the PLM. The top-k words are selected by accumulating the probability distributions of all samples to get the word-to-label mapping for that category. Since the model parameters change during the fine-tuning process, the result may shift, so the mapping needs to be re-ranked and adjusted after each training round.
  - Given a category and its target word, how to generate a template. The author uses the T5 model because its mask span seq2seq pretraining task aligns well with the template generation task. This can be explained in a single diagram:
    ![g5YjfI.png](https://z3.ax1x.com/2021/05/19/g5YjfI.png)
    The generated prompt takes into account both the context of the training samples and the semantic context of the label word. The author uses a wide beam width to beam search a set of prompt candidates (100+), then fine-tunes each sample on a small training set, selecting the one with the highest performance on the validation set (or using top-k ensemble) as the final prompt.
  - Dynamic selective integration of tasks, which is more complicated. After obtaining the prompt, the question is how to construct the input sample. As shown in the first image, for each category, a sample is randomly selected and converted into a prompt to serve as the description for that category. All category descriptions are concatenated with the input sample (the one to be trained). During sampling, sentence-BERT is used to obtain the semantic embeddings of each sample, and only the top 50% of samples with the highest semantic similarity to the input sample are selected.
- The design of this prompt is somewhat similar to a semantic similarity task, where the input is "x is a mask example? y is positive; z is negative." This essentially compares the semantic similarity between x and yz, propagating the label through this comparison.

# GPT Understands, Too
- [![g52dDH.png](https://z3.ax1x.com/2021/05/19/g52dDH.png)](https://imgtu.com/i/g52dDH)
- This paper introduces P-tuning, which is not about finding discrete prompts (specific texts), but rather continuous ones (embeddings).
- Let's review the entire prompt-based method. It essentially transforms data and tasks into a form suitable for language model tasks, bringing them closer to pretraining objectives and enabling better utilization of the pretrained model's knowledge. In practice, this involves adding some prompt-generated templates to the input, and the output becomes target words related to category labels. The author reflects on whether these prompt-generated templates necessarily need to be human-understandable text. After all, what the model actually processes are embeddings. So, when searching for prompts, why not directly optimize the embeddings instead? Therefore, the author proposes using some unused symbols from word tables (such as the "unused" tokens in BERT) as pseudo-template tokens. These tokens are fixed, and rather than searching for new tokens, we directly optimize the embeddings corresponding to these tokens.
- To make these pseudo-tokens resemble natural language more closely, rather than just being independent symbols, the author also uses a bidirectional LSTM for encoding, which serves as the prompt encoder. However, the motivation for this approach isn't fully clear. Why not directly model the relationship within the PLM itself?
- From this perspective, the approach is essentially about concatenating a few embeddings to the input and optimizing them. The output and post-processing adopt the PET (Prompt-based Elicitation of Task) form, which feels like adding a layer for fine-tuning (hence the name **P**rompt fine**tuning**?). In my view, both layer-based fine-tuning and P-tuning introduce a small number of parameters to adapt the PLM to downstream tasks, but P-tuning changes the format of the downstream task to better align with pretraining objectives, making the fine-tuning structural priors more reasonable. It also offers a higher-level summary of prompt-based work.

{% endlang_content %}

{% lang_content zh %}

# Commonsense Knowledge Mining from Pretrained Models

- 作者想要做到挖掘未知分布数据中的常识，而传统的监督学习方法容易受到训练集中的数据分布影响，导致结果有偏差
- 将关系三元组转换为masked sentence送给BERT，通过BERT的预测结果计算互信息来对三元组的可信度排序
- 任务，给定一个三元组为其打分，确定这个三元组代表了真实世界知识的可能性，作者将其分为两步：
  - 将三元组转化为mask过后的句子：对每个关系手工设计了多个模板，同时还设计了一系列规则来确保语法正确性（单复数、插入冠词、改动名词等等），这样所有模板和规则的组合得到了一系列候选句子，然后通过预训练单向语言模型来计算每个句子是正常句子的得分log-likelihood
  - 将生成的句子输入BERT打分：这里作者用条件点互信息计算，即在关系r的条件下，头尾实体之间的互信息大小作为分数：
    
    $$
    PMI(tail,head|relation) = \log p(tail|head, relation) - \log p(tail|realtion) \\
    $$
    
    放在语言模型中，实际上就是将tail mask掉然后预测，只不过上式右边第一项是只mask tail,第二项则还mask掉了head（只mask,不预测）。另外可能出现实体由多个词组成的情况，这里作者采用了一种贪心近似的方法，先把词全部mask掉然后预测，拿到概率最高的词unmask，再反复迭代预测剩下的词，每次还原概率最高的词，之后累乘这一系列概率就可以得到整个词的条件概率。上式并不是对称的，因此作者还反过来计算了基于关系和尾实体的头实体概率，最后平均两个PMI值作为结果。
- 最终结果虽然比不上监督学习，但是在无监督学习中取得了最佳效果
- 这是较早尝试利用预训练模型的Mask Predict，将任务设计为完形填空来完成，可以看到这里的Pattern还是手工设计（针对每个关系设计一系列规则）。

# Argumentative Relation Classification as Plausibility Ranking

- 这篇论文做的任务为Argumentative relation classification，即文本对分类，给定（或者不显式给出）结论，区分一对文本是支持还是反对。正例文本对里，两个文本都支持结论；负例文本对里，一个支持结论而另一个不支持，互相反驳。
- 对于这个很有意思的任务，作者采用了一个同样很有意思的做法：使用孪生网络做ranking，rank的是一个构造文本的plausibility，即可信度。而这个构造文本是什么？很简单，将要判别的两个句子用一个连接词连接起来，得到构造文本的正负例：
  - 正例：文本A，而且，文本B
  - 负例：文本A，然而，文本B
- 假如文本A和文本B是反对的关系，那么显然负例这么一段文本的可信度高；为文本A和文本B互相支持，那么正例构造文本的可信度高。
- 接下来就用预训练语言模型作为孪生网络的编码器，然后做ranking。
- 本质思想是构造了文本和任务，将任务用正常的自然语言表示，这样就可以利用学习到正常文本知识的语言模型来做学习和预测。
- 和上一篇论文一样，核心都是将任务转为自然语言（模板），巧用预训练语言模型间接的完成任务（完成构造任务）

# Zero-shot Text Classification With Generative Language Models

- 作者使用GPT，将文本分类问题转化为给定包含原文本和类别的自然语言，通过文本生成间接判断类别
- 这样做的一个好处即标题提到的zero-shot，可以泛化到训练集中不存在的类别
- 具体而言，将文本分类问题转为一个选择QA任务，即所有的选项拼成了问题：该文本属于下面哪一类？A;B;C;D.....，之后再拼接上待分类文本，目标是训练语言模型，直接生成正确的类别的文本。
- 另外为了减少预训练和finetune之间的gap，作者还加入了一个前置的预训练任务，叫title prediction pretraining，即将所有候选标题和正文拼接起来，然后生成正确的标题。
- 这是一篇非常直观、间接且大胆的利用语言模型分类任务的工作，直接让语言模型生成类别文字。
  [![gW98oV.png](https://z3.ax1x.com/2021/05/17/gW98oV.png)](https://imgtu.com/i/gW98oV)
- 最终的zero-shot结果，虽然依然比不上finetune和sota，但是相比random和majority两个baseline可以比较出模型还是学到了相当强的泛化能力。最主要的还是把语言模型玩出了花，提供了这么一种直接设计多项选择疑问句来完成分类任务的思路。

# Exploiting Cloze Questions for Few Shot Text Classification and NaturalLanguage Inference

- 该论文正式引入了PET的概念：Pattern-Exploiting Training。
- 在上面三篇论文中我们可以看到，很多NLP任务可以通过提供自然语言任务描述的方式，通过语言模型来无监督的或者间接的完成。但是这类方法终究还是比不过监督学习方法。
- PET提供了一种半监督学习方式，在低资源场景下成功超过了监督学习模型的结果。
- 一张图就能说明PET的原理：
  [![gWivmd.png](https://z3.ax1x.com/2021/05/17/gWivmd.png)](https://imgtu.com/i/gWivmd)
  - 作者引入了两个名词，pattern负责把输入文本根据任务改造成一个带mask的完形填空文本，verbalizer负责把语言模型预测的mask词映射到label上。这样一个pattern对应一个verbalizer，称为PvP。。。（pattern verbalizer pair）
  - 整个PET过程分三步：
    - 第一步用PvP，在小训练集上微调预训练语言模型
    - 第二步，每一个任务可以设计多个PvP，这样得到多个第一步训练出的语言模型，集成，在大量未标注数据上打标软标签
    - 第三步，用一个分类器在打标后的数据上完成监督学习
- 第二步中有两个小细节：多分类器集成，即多个预测标签分布相加，这里可以等权重相加，也可以根据PvP直接在训练集上zero-shot的表现作为先验权重（实验结果这样做好些）；打标时打的是软标签即概率分布，softmax时取T=2做了温度处理。这两个处理都是为了能够更好的学习到语言模型的知识，一个在于集成更加鲁棒，另一个则相当于知识蒸馏。
- 另外作者还提出了iPET，其实就是传统的半监督学习，训练打标之间迭代，用越来越多的数据训练出不同代模型然后集成。
- 这样的半监督框架好处在于，最终实际操作依然是监督学习，准确率较高，而语言模型带来的不确定性在知识蒸馏（软化标签）的时候降低了。

# It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners

- 还是PET原版人马，又水了一篇，换了个动机，说用PET的话，小模型也能在few-shot上取得与GPT-3这样的大模型接近的结果，环保
- 将PvP中要预测的词从单个mask扩展为多个mask，训练的时候插入固定最大数量的mask，预测时再做后处理
- 给了更丰富的实验结果（不过好像还是arxiv挂着，没中会议。。。）（更新：惊了，拿到了NAACL 2021 杰出论文）

# Improving and Simplifying Pattern Exploiting Training

- [![gIih5j.png](https://z3.ax1x.com/2021/05/19/gIih5j.png)](https://imgtu.com/i/gIih5j)
- PET依然需要大量领域未标注数据来做半监督学习，本文提出了ADAPET，不用未标注数据也能取得更好效果
- 作者通过修改任务目标来达成这一目的。当我们使用PET时，浪费了两类信息：
  - mask位置上预测的词，仅仅在与类别label有映射关系的target word vocab上做softmax计算交叉熵，其余词没有计算损失
  - 仅仅预测了mask位置，其他所有位置的embedding没有计算损失
- 因此作者就想充分利用这两个信息，修改任务目标
  - 将损失从交叉熵改为两个二元交叉熵，一个依然是在label相关target词上算损失，另一部分损失则负责优化降低其他所有不相关词的概率
  - 将mask替换为正确或者错误的target word，然后对输入剩下部分做MLM,要是target word对的话MLM就应该预测对，反之就应该预测错
  - 分别对应图中左右两类损失
- ADAPET增加了目标函数，对参数做了更充分的训练，对比PET结果也确实不错，不使用未标注数据还在很多任务上超过了PET

# AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts

- 由上面介绍的工作可以发现，构建有效的文本来触发语言模型得到结果至关重要，即构建prompt。目前看到的都是手工构建的，后来也出现了一批工作尝试自动构建prompts

- 这个工作其实不能算是prompts，更准确的说法是trigger words sequence，因为它其实是把文本对抗样本生成的一套方法拿到了prompt构建当中。

- 具体而言，其借鉴了HotFlip: White-box adversarial examples for text classification 和 Universal Adversarial Triggers for Attacking and Analyzing NLP两篇论文，即在样本中拼接一系列触发词，即可使得模型的预测结果错误，而模型的触发词搜索主要使用的是hotflip方法：
  
  - 初始化触发词 $$\mathbf{e}_{a d v}$$（比如the，a，an等），前向过一遍模型得到损失关于触发词embedding的梯度 $$\nabla_{\mathbf{e}_{a d v}} \mathcal{L}$$ ，注意这里用于计算损失所用的label应该是想要攻击得到的错误label，即fool model之后的label
  - 我们希望替换第i个触发词为词 $$\mathbf{e}_{i}$$，使得替换之后损失下降的最多，模型最容易预测出错误的标签，所以我们要找的词是 $$ \underset{\mathbf{e}_{i}^{\prime} \in \mathcal{V}}{\arg \min } \mathcal{L}(\mathbf{e}_{i}^{\prime}) $$。这里通过泰勒一阶展开来近似，需要求到损失关于token的导数，由于token embedding lookup不可导，所以才需要求到某个token的embedding的导数
  
  $$
  \mathcal{L}(\mathbf{e}_{i}^{\prime})    =  \mathcal{L}(\mathbf{e}_{a d v_{i}}) + \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{a d v_{i}}\right]^{\top} \nabla_{\mathbf{e}_{a d v_{i}}} \mathcal{L} 
  $$
  
  $$
  \propto \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{a d v_{i}}\right]^{\top} \nabla_{\mathbf{e}_{a d v_{i}}} \mathcal{L} 
  $$
  
  - 这样就得到了第一轮迭代中的第一个触发词，之后通过beam search得到剩下的触发词，并迭代多次，最终得到可以用于攻击模型的触发词序列。

- 以上是文本对抗攻击中的hotflip方法，其本质就是生成一些触发词，拼接到样本上，使得模型预测出错的label。autoprompt的思想就是生成触发词，使得模型预测出指定label。
  [![ghFDuF.md.png](https://z3.ax1x.com/2021/05/18/ghFDuF.md.png)](https://imgtu.com/i/ghFDuF)

- 接下来就简单了。作者首先在训练集上用hotflip方法为每个任务生成了触发词，然后用模板将样本变为一个句子，如图所示，句子拼接上触发词序列（[T]）和PLM要预测的mask位置([P])，让模型预测出词之后再后处理得到label。具体的后处理操作是，将每个label对应的预测词集合得到的概率累加，最后归一化，作为标签的概率。

- 上面只说了PvP中的prompt自动构建方法，而verbalizer，即预测词到标签的映射作者也给出了一个自动搜索的方法：
  
  - 将PLM编码之后包含上下文信息的mask token的embedding作为特征输入，标签作为输出来训练一个logistic分类器，之后将所有词的PLM编码之后的embedding依次输入这个分类器，得到每个词在每个标签上的评分，根据评分top k来为每个标签类别选择词作为映射集合。这么做实际上是将预测标签所需的mask token编码embedding和每个词的编码embedding比较，取最相近的top k，只不过利用logistic分类器做了一个类别相关的特征加权，不仅仅是取PLM编码之后的语义相似度，非常巧妙。

# Making Pre-trained Language Models Better Few-shot Learners

- 这篇论文标题就是GPT3的标题加了个better，强调如何更好的利用prompt做few shot learning。
- 提出了一个训练体系：基于prompt的微调+prompt自动生成+动态选择性融入任务说明到prompt中，且这一切都是strong task-agnostic。接下来分别说这三点改进。
- [![g58yOe.png](https://z3.ax1x.com/2021/05/19/g58yOe.png)](https://imgtu.com/i/g58yOe)
- 上图清晰的展示了第一点改进：基于prompt的微调。可以看到，和以往prompt方法相比，除了输入、prompt之外，输入还拼接上了每个label的说明
- 至于prompt自动生成，分为两部分：
  - 如何在给定模板的情况下，自动生成目标词到标签的映射。这里作者也是用PLM的结果不断迭代。首先对每个类，找出这个类的所有训练样本，通过PLM推断得到mask词的概率分布，累加所有样本的概率分布取topk就得到了词到该类别标签的映射。由于接下来训练微调时模型参数变化，结果可能有改变，所以需要每轮训练后重新rerank调整一下映射关系。
  - 给定类别和这个类别的目标词，如何生成模板。作者采用了T5模型，因为其mask span seq2seq预训练的目标和模板生成任务很符合。一张图就可以解释清楚：
    [![g5YjfI.png](https://z3.ax1x.com/2021/05/19/g5YjfI.png)](https://imgtu.com/i/g5YjfI)
    这样生成的prompt考虑了训练样本上下文和标签词的语境。作者使用wide beam width来beam search出一堆prompt候选（100+），然后在一个小训练集上微调每个样本，取验证集最高的（或者topk集成）作为最终prompt
  - 动态选择性融入任务，这里做的比较麻烦，即得到prompt后如何构造输入样本，也是如第一张图所示，对每个类别，采样一个样本转化为prompt当做这个类别的说明，将所有类别说明和输入样本（待训练样本）拼接。采样时，使用sentence-BERT得到每个样本的语义embedding，然后只取和输入样本语义相似度前50%的样本进行采样。
- 这种prompt的设计有点像是在做语义相似度任务，输入x，已知y为正例，z为负例，构造了输入为“x是mask例？y为正例；z为负例”，相当于比较x与yz的语义相似度，做一个标签的传播

# GPT Understands, Too

- [![g52dDH.png](https://z3.ax1x.com/2021/05/19/g52dDH.png)](https://imgtu.com/i/g52dDH)
- 本文提出了P-tuning，即不是找离散的prompt（具体文本），而是找连续的（embedding）
- 回顾一下整个prompt based methods，都是把数据和任务转化为语言模型任务的形式，使其更加贴近预训练目标，能够更好的利用预训练模型的知识。实际操作时，就是把输入添加一些prompt generated templates，输出变成与类别label相关的target words，作者反思，这些prompt generated templates 本质上就是一些词，一定要是人类能够理解的文本吗？这些文本输入到模型的实际上是embedding，那么搜索prompt的时候为什么不直接优化embedding呢？所以作者提出就用几个词表中没用的符号（例如BERT中的unused）来作为pseudo template token，固定这些token，不去搜索新的token，而是直接优化token对应的embedding。
- 为了让这些pseudo token更像是自然语言，而不是独立的几个符号，作者还用了双向LSTM来做编码，即prompt encoder，这里感觉动机阐释的不是很清楚，为什么不能放在PLM里直接建模之间关系？
- 这么看来整体就相当于输入拼接上几个embedding然后去优化，只不过输出和后处理采用了PET的形式，很像自己加了某个层去微调（所以叫**P**rompt fine**tuning**？）。我感觉加层微调和P-tuning都是引入少量参数把PLM用到自己的下游任务上，只不过P-tuning转换了下游任务形式，使其跟贴近预训练目标，算是微调结构先验更合理吧，同时也算是从另一个高度总结了prompt一类的工作。

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
        data-lang="en"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>