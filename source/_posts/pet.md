---
title: Prompt Based Task Reformulation in NLP调研
date: 2021-05-13 19:36:34
tags:
  - pretrained language model
  - pet
  - machine learning
  -	few shot
  -	nlp
categories:
  - 自然语言处理
html: true
mathjax: true
---
***
-	记录近年基于模板来完成任务重构的方法，这是一个比较有意思的方向，尤其是GPT3出现之后。 这类方法一般针对任务设计prompt，将样本和任务一起转换为自然语言形式的template，直接输入预训练语言模型预测出文本，间接的完成任务。prompt的构建一方面统一了下游任务和预训练任务的形式（语言模型）在few shot learning上能取得较好结果。主要阅读以下9篇论文：
	-	早期的将问题转为自然语言并使用预训练语言模型解答的：
		-	(Harvard)Commonsense Knowledge Mining from Pretrained Models
		-	(Heidelberg)Argumentative Relation Classification as Plausibility Ranking
		-	(NVIDIA)Zero-shot Text Classification With Generative Language Models
	-	PET方向，Pattern Exploiting Training
		-	(LMU)Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference
		-	(LMU)It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners
		-	(UNC)Improving and Simplifying Pattern Exploiting Training
	-	自动构建prompt，Automatically Searching Prompts
		-	(UCI,UCB)AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts
		-	(Princeton, MIT)Making Pre-trained Language Models Better Few-shot Learners
		-	(THU)GPT Understands, Too


<!--more-->
# Commonsense Knowledge Mining from Pretrained Models
-	作者想要做到挖掘未知分布数据中的常识，而传统的监督学习方法容易受到训练集中的数据分布影响，导致结果有偏差
-	将关系三元组转换为masked sentence送给BERT，通过BERT的预测结果计算互信息来对三元组的可信度排序
-	任务，给定一个三元组为其打分，确定这个三元组代表了真实世界知识的可能性，作者将其分为两步：
	-	将三元组转化为mask过后的句子：对每个关系手工设计了多个模板，同时还设计了一系列规则来确保语法正确性（单复数、插入冠词、改动名词等等），这样所有模板和规则的组合得到了一系列候选句子，然后通过预训练单向语言模型来计算每个句子是正常句子的得分log-likelihood
	-	将生成的句子输入BERT打分：这里作者用条件点互信息计算，即在关系r的条件下，头尾实体之间的互信息大小作为分数：
		$$
		PMI(tail,head|relation) = \log p(tail|head, relation) - \log p(tail|realtion) \\
		$$
		放在语言模型中，实际上就是将tail mask掉然后预测，只不过上式右边第一项是只mask tail,第二项则还mask掉了head（只mask,不预测）。另外可能出现实体由多个词组成的情况，这里作者采用了一种贪心近似的方法，先把词全部mask掉然后预测，拿到概率最高的词unmask，再反复迭代预测剩下的词，每次还原概率最高的词，之后累乘这一系列概率就可以得到整个词的条件概率。上式并不是对称的，因此作者还反过来计算了基于关系和尾实体的头实体概率，最后平均两个PMI值作为结果。
-	最终结果虽然比不上监督学习，但是在无监督学习中取得了最佳效果
-	这是较早尝试利用预训练模型的Mask Predict，将任务设计为完形填空来完成，可以看到这里的Pattern还是手工设计（针对每个关系设计一系列规则）。

# Argumentative Relation Classification as Plausibility Ranking
-	这篇论文做的任务为Argumentative relation classification，即文本对分类，给定（或者不显式给出）结论，区分一对文本是支持还是反对。正例文本对里，两个文本都支持结论；负例文本对里，一个支持结论而另一个不支持，互相反驳。
-	对于这个很有意思的任务，作者采用了一个同样很有意思的做法：使用孪生网络做ranking，rank的是一个构造文本的plausibility，即可信度。而这个构造文本是什么？很简单，将要判别的两个句子用一个连接词连接起来，得到构造文本的正负例：
	-	正例：文本A，而且，文本B
	-	负例：文本A，然而，文本B
-	假如文本A和文本B是反对的关系，那么显然负例这么一段文本的可信度高；为文本A和文本B互相支持，那么正例构造文本的可信度高。
-	接下来就用预训练语言模型作为孪生网络的编码器，然后做ranking。
-	本质思想是构造了文本和任务，将任务用正常的自然语言表示，这样就可以利用学习到正常文本知识的语言模型来做学习和预测。
-	和上一篇论文一样，核心都是将任务转为自然语言（模板），巧用预训练语言模型间接的完成任务（完成构造任务）

# Zero-shot Text Classification With Generative Language Models
-	作者使用GPT，将文本分类问题转化为给定包含原文本和类别的自然语言，通过文本生成间接判断类别
-	这样做的一个好处即标题提到的zero-shot，可以泛化到训练集中不存在的类别
-	具体而言，将文本分类问题转为一个选择QA任务，即所有的选项拼成了问题：该文本属于下面哪一类？A;B;C;D.....，之后再拼接上待分类文本，目标是训练语言模型，直接生成正确的类别的文本。
-	另外为了减少预训练和finetune之间的gap，作者还加入了一个前置的预训练任务，叫title prediction pretraining，即将所有候选标题和正文拼接起来，然后生成正确的标题。
-	这是一篇非常直观、间接且大胆的利用语言模型分类任务的工作，直接让语言模型生成类别文字。
	[![gW98oV.png](https://z3.ax1x.com/2021/05/17/gW98oV.png)](https://imgtu.com/i/gW98oV)
-	最终的zero-shot结果，虽然依然比不上finetune和sota，但是相比random和majority两个baseline可以比较出模型还是学到了相当强的泛化能力。最主要的还是把语言模型玩出了花，提供了这么一种直接设计多项选择疑问句来完成分类任务的思路。

# Exploiting Cloze Questions for Few Shot Text Classification and NaturalLanguage Inference
-	该论文正式引入了PET的概念：Pattern-Exploiting Training。
-	在上面三篇论文中我们可以看到，很多NLP任务可以通过提供自然语言任务描述的方式，通过语言模型来无监督的或者间接的完成。但是这类方法终究还是比不过监督学习方法。
-	PET提供了一种半监督学习方式，在低资源场景下成功超过了监督学习模型的结果。
-	一张图就能说明PET的原理：
	[![gWivmd.png](https://z3.ax1x.com/2021/05/17/gWivmd.png)](https://imgtu.com/i/gWivmd)
	-	作者引入了两个名词，pattern负责把输入文本根据任务改造成一个带mask的完形填空文本，verbalizer负责把语言模型预测的mask词映射到label上。这样一个pattern对应一个verbalizer，称为PvP。。。（pattern verbalizer pair）
	-	整个PET过程分三步：
		-	第一步用PvP，在小训练集上微调预训练语言模型
		-	第二步，每一个任务可以设计多个PvP，这样得到多个第一步训练出的语言模型，集成，在大量未标注数据上打标软标签
		-	第三步，用一个分类器在打标后的数据上完成监督学习
-	第二步中有两个小细节：多分类器集成，即多个预测标签分布相加，这里可以等权重相加，也可以根据PvP直接在训练集上zero-shot的表现作为先验权重（实验结果这样做好些）；打标时打的是软标签即概率分布，softmax时取T=2做了温度处理。这两个处理都是为了能够更好的学习到语言模型的知识，一个在于集成更加鲁棒，另一个则相当于知识蒸馏。
-	另外作者还提出了iPET，其实就是传统的半监督学习，训练打标之间迭代，用越来越多的数据训练出不同代模型然后集成。
-	这样的半监督框架好处在于，最终实际操作依然是监督学习，准确率较高，而语言模型带来的不确定性在知识蒸馏（软化标签）的时候降低了。

# It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners
-	还是PET原版人马，又水了一篇，换了个动机，说用PET的话，小模型也能在few-shot上取得与GPT-3这样的大模型接近的结果，环保
-	将PvP中要预测的词从单个mask扩展为多个mask，训练的时候插入固定最大数量的mask，预测时再做后处理
-	给了更丰富的实验结果（不过好像还是arxiv挂着，没中会议。。。）（更新：惊了，拿到了NAACL 2021 杰出论文）

# Improving and Simplifying Pattern Exploiting Training
-	[![gIih5j.png](https://z3.ax1x.com/2021/05/19/gIih5j.png)](https://imgtu.com/i/gIih5j)
-	PET依然需要大量领域未标注数据来做半监督学习，本文提出了ADAPET，不用未标注数据也能取得更好效果
-	作者通过修改任务目标来达成这一目的。当我们使用PET时，浪费了两类信息：
	-	mask位置上预测的词，仅仅在与类别label有映射关系的target word vocab上做softmax计算交叉熵，其余词没有计算损失
	-	仅仅预测了mask位置，其他所有位置的embedding没有计算损失
-	因此作者就想充分利用这两个信息，修改任务目标
	-	将损失从交叉熵改为两个二元交叉熵，一个依然是在label相关target词上算损失，另一部分损失则负责优化降低其他所有不相关词的概率
	-	将mask替换为正确或者错误的target word，然后对输入剩下部分做MLM,要是target word对的话MLM就应该预测对，反之就应该预测错
	-	分别对应图中左右两类损失
-	ADAPET增加了目标函数，对参数做了更充分的训练，对比PET结果也确实不错，不使用未标注数据还在很多任务上超过了PET

# AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts
-	由上面介绍的工作可以发现，构建有效的文本来触发语言模型得到结果至关重要，即构建prompt。目前看到的都是手工构建的，后来也出现了一批工作尝试自动构建prompts
-	这个工作其实不能算是prompts，更准确的说法是trigger words sequence，因为它其实是把文本对抗样本生成的一套方法拿到了prompt构建当中。
-	具体而言，其借鉴了HotFlip: White-box adversarial examples for text classification 和 Universal Adversarial Triggers for Attacking and Analyzing NLP两篇论文，即在样本中拼接一系列触发词，即可使得模型的预测结果错误，而模型的触发词搜索主要使用的是hotflip方法：
	-	初始化触发词 $$\mathbf{e}_{a d v}$$（比如the，a，an等），前向过一遍模型得到损失关于触发词embedding的梯度 $$\nabla_{\mathbf{e}_{a d v}} \mathcal{L}$$ ，注意这里用于计算损失所用的label应该是想要攻击得到的错误label，即fool model之后的label
	-	我们希望替换第i个触发词为词 $$\mathbf{e}_{i}$$，使得替换之后损失下降的最多，模型最容易预测出错误的标签，所以我们要找的词是 $$ \underset{\mathbf{e}_{i}^{\prime} \in \mathcal{V}}{\arg \min } \mathcal{L}(\mathbf{e}_{i}^{\prime}) $$。这里通过泰勒一阶展开来近似，需要求到损失关于token的导数，由于token embedding lookup不可导，所以才需要求到某个token的embedding的导数

	$$
	\mathcal{L}(\mathbf{e}_{i}^{\prime})	=  \mathcal{L}(\mathbf{e}_{a d v_{i}}) + \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{a d v_{i}}\right]^{\top} \nabla_{\mathbf{e}_{a d v_{i}}} \mathcal{L} 
	$$
	$$
	\propto \left[\mathbf{e}_{i}^{\prime}-\mathbf{e}_{a d v_{i}}\right]^{\top} \nabla_{\mathbf{e}_{a d v_{i}}} \mathcal{L} 
	$$
	-	这样就得到了第一轮迭代中的第一个触发词，之后通过beam search得到剩下的触发词，并迭代多次，最终得到可以用于攻击模型的触发词序列。
-	以上是文本对抗攻击中的hotflip方法，其本质就是生成一些触发词，拼接到样本上，使得模型预测出错的label。autoprompt的思想就是生成触发词，使得模型预测出指定label。
	[![ghFDuF.md.png](https://z3.ax1x.com/2021/05/18/ghFDuF.md.png)](https://imgtu.com/i/ghFDuF)
-	接下来就简单了。作者首先在训练集上用hotflip方法为每个任务生成了触发词，然后用模板将样本变为一个句子，如图所示，句子拼接上触发词序列（[T]）和PLM要预测的mask位置([P])，让模型预测出词之后再后处理得到label。具体的后处理操作是，将每个label对应的预测词集合得到的概率累加，最后归一化，作为标签的概率。
-	上面只说了PvP中的prompt自动构建方法，而verbalizer，即预测词到标签的映射作者也给出了一个自动搜索的方法：
	-	将PLM编码之后包含上下文信息的mask token的embedding作为特征输入，标签作为输出来训练一个logistic分类器，之后将所有词的PLM编码之后的embedding依次输入这个分类器，得到每个词在每个标签上的评分，根据评分top k来为每个标签类别选择词作为映射集合。这么做实际上是将预测标签所需的mask token编码embedding和每个词的编码embedding比较，取最相近的top k，只不过利用logistic分类器做了一个类别相关的特征加权，不仅仅是取PLM编码之后的语义相似度，非常巧妙。

# Making Pre-trained Language Models Better Few-shot Learners
-	这篇论文标题就是GPT3的标题加了个better，强调如何更好的利用prompt做few shot learning。
-	提出了一个训练体系：基于prompt的微调+prompt自动生成+动态选择性融入任务说明到prompt中，且这一切都是strong task-agnostic。接下来分别说这三点改进。
-	[![g58yOe.png](https://z3.ax1x.com/2021/05/19/g58yOe.png)](https://imgtu.com/i/g58yOe)
-	上图清晰的展示了第一点改进：基于prompt的微调。可以看到，和以往prompt方法相比，除了输入、prompt之外，输入还拼接上了每个label的说明
-	至于prompt自动生成，分为两部分：
	-	如何在给定模板的情况下，自动生成目标词到标签的映射。这里作者也是用PLM的结果不断迭代。首先对每个类，找出这个类的所有训练样本，通过PLM推断得到mask词的概率分布，累加所有样本的概率分布取topk就得到了词到该类别标签的映射。由于接下来训练微调时模型参数变化，结果可能有改变，所以需要每轮训练后重新rerank调整一下映射关系。
	-	给定类别和这个类别的目标词，如何生成模板。作者采用了T5模型，因为其mask span seq2seq预训练的目标和模板生成任务很符合。一张图就可以解释清楚：
	[![g5YjfI.png](https://z3.ax1x.com/2021/05/19/g5YjfI.png)](https://imgtu.com/i/g5YjfI)
	这样生成的prompt考虑了训练样本上下文和标签词的语境。作者使用wide beam width来beam search出一堆prompt候选（100+），然后在一个小训练集上微调每个样本，取验证集最高的（或者topk集成）作为最终prompt
	-	动态选择性融入任务，这里做的比较麻烦，即得到prompt后如何构造输入样本，也是如第一张图所示，对每个类别，采样一个样本转化为prompt当做这个类别的说明，将所有类别说明和输入样本（待训练样本）拼接。采样时，使用sentence-BERT得到每个样本的语义embedding，然后只取和输入样本语义相似度前50%的样本进行采样。
-	这种prompt的设计有点像是在做语义相似度任务，输入x，已知y为正例，z为负例，构造了输入为“x是mask例？y为正例；z为负例”，相当于比较x与yz的语义相似度，做一个标签的传播

# GPT Understands, Too
-	[![g52dDH.png](https://z3.ax1x.com/2021/05/19/g52dDH.png)](https://imgtu.com/i/g52dDH)
-	本文提出了P-tuning，即不是找离散的prompt（具体文本），而是找连续的（embedding）
-	回顾一下整个prompt based methods，都是把数据和任务转化为语言模型任务的形式，使其更加贴近预训练目标，能够更好的利用预训练模型的知识。实际操作时，就是把输入添加一些prompt generated templates，输出变成与类别label相关的target words，作者反思，这些prompt generated templates 本质上就是一些词，一定要是人类能够理解的文本吗？这些文本输入到模型的实际上是embedding，那么搜索prompt的时候为什么不直接优化embedding呢？所以作者提出就用几个词表中没用的符号（例如BERT中的unused）来作为pseudo template token，固定这些token，不去搜索新的token，而是直接优化token对应的embedding。
-	为了让这些pseudo token更像是自然语言，而不是独立的几个符号，作者还用了双向LSTM来做编码，即prompt encoder，这里感觉动机阐释的不是很清楚，为什么不能放在PLM里直接建模之间关系？
-	这么看来整体就相当于输入拼接上几个embedding然后去优化，只不过输出和后处理采用了PET的形式，很像自己加了某个层去微调（所以叫**P**rompt fine**tuning**？）。我感觉加层微调和P-tuning都是引入少量参数把PLM用到自己的下游任务上，只不过P-tuning转换了下游任务形式，使其跟贴近预训练目标，算是微调结构先验更合理吧，同时也算是从另一个高度总结了prompt一类的工作。