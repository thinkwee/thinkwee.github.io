---
title: Edit-based Text Generation
date: 2021-05-11 15:45:26
tags:
  - seq2seq
  - math
  - machine learning
  -	theory
  -	nlp
categories:
  - 自然语言处理
html: true
mathjax: true
---
***
-	记录近年来关于编辑式seq2seq的方法，这类方法对于输入输出同语种且较小更改的任务（纠错、简化、摘要）有着高效率（部分自回归或非自回归解码）和less data hungry（输出词表小）的优势。
-	主要阅读五篇论文，按照其在arxiv上发表时间排序：
	-	(LevT, Facebook) Levenshtein Transformer
	-	(华为) EditNTS: An Neural Programmer-Interpreter Model for Sentence Simplification through Explicit Editing
	-	(LaserTagger, Google) Encode, Tag, Realize: High-Precision Text Editing
	-	(PIE，印度理工) Parallel Iterative Edit Models for Local Sequence Transduction
	-	(Google) Felix: Flexible Text Editing Through Tagging and Insertion
  
<!--more-->

# Levenshtein Transformer
-	贡献点
	-	提出了基于插入和删除操作的编辑式transformer，效率提升5倍
	-	针对插入和删除互补性，设计了一种dual policy式的强化学习训练方法
	-	统一了文本生成和文本完善（refinement）两个任务

## 定义问题
-	作者将文本生成和文本完善统一为一个马尔科夫过程，由元组(Y,A,$\xi$,R,$y_0$)定义（序列，动作集合，环境，奖励函数，初始序列）
	-	Y即文本序列，大小为$V^N$
	-	$\xi$是环境，在解码的每一步，agent接收一个序列y作为输入，采取动作a，然后得到奖励r
	-	奖励函数定义为当前序列和ground truth序列之间的距离，显然本文就使用了Levenshtein距离（编辑距离）
	-	agent要学习的策略即一个从序列y到动作概率分布P(A)的映射
	-	值得注意的是，$y_0$可以是空序列，此时即文本生成，也可以是一个生成好的序列，此时的任务即文本完善
-	动作
	-	删除操作：对序列中的每一个token做二分类，决定是否应该删除。对于序列中第一个和最后一个token直接保留，不做删除操作的判断，避免序列边界被破坏
	-	插入操作：插入操作分为两步：首先预测insert position,对每两个相邻token位置做一个多分类（insert position的数量），值得注意的是输入包含了这两个token；之后再针对每个预测位置做一个词典V大小的分类，生成具体的词插入。
-	整个流程分为了三步：输入序列$y_0$，并行的对序列所有位置做删除操作，得到$y_1$；对$y_1$，对序列所有位置预测insert position，根据预测结果添加placeholder得到序列$y_2$；对$y_2$中的所有placeholder预测具体的词得到最终结果，完成一轮迭代。
-	三步操作共享一个transformer，同时为了降低复杂度（一轮自回归文本生成变成了三轮非自回归操作），将删除和insert position预测操作的分类器接在中间的transformer block上，只对文本生成的分类器接在最后一层transformer block上，因为前两个任务比较简单，不需要复杂的特征提取。

## 训练
-	训练的过程为imitation learning，这里有较多强化学习概念，但实际上就是对数据做扰动然后让模型学习还原，且依然是MLE。我尝试在NLP任务中用其具体操作来解释
	-	训练依然是teacher forcing，即直接对输入去学习预测oracle（ground truth），不存在自回归。
	-	对于Levenshtein Transformer，训练时我们需要三部分：
		-	训练时的输入：输入即需要做编辑操作的文本（state distribution fed during training)，这些句子应该是一些原始文本做了一些扰动操作（roll-in policy）得到的。很容易理解，比如我故意删除了一些token，然后让模型去学习插入。因此该部分文本构建分为两步：确定原始文本，执行扰动操作。其中原始文本可以是空文本，也可以是ground truth文本。在设计扰动操作时，引入了作者所谓的dual policy，实际上就是让模型自己的操作也作为扰动操作

		| 操作                                 	| 扰动类型 	| 让模型学习 	| dual policy 	|
		|--------------------------------------	|----------	|------------	|-------------	|
		| 对oracle文本随机丢词作为输入         	| 删除     	| 插入       	|             	|
		| 直接将oracle输入文本作为输入         	| 插入     	| 删除       	|             	|
		| 使用模型执行删除动作后的文本作为输入 	| 删除     	| 插入       	| √           	|
		| 使用模型执行插入动作后的文本作为输入 	| 插入     	| 删除       	| √           	|

		这里作者使用概率来选择各个操作构造数据，示意图如下：
		[![gaoEPx.png](https://z3.ax1x.com/2021/05/11/gaoEPx.png)](https://imgtu.com/i/gaoEPx)

		-	训练时的输出：即模型学习到与扰动相反的操作之后，需要还原的原始文本，当确定了输入输出之后，ground truth的动作（expert policy）就确定了。一般来讲自然是要还原到oracle，但是作者考虑到直接学习太难，因此以一定概率模型只需要还原出一个低噪声版本即可。这个低噪声版本由序列蒸馏得到，具体做法是用统一数据集训练一个正常的transformer，然后用这个transformer对每条数据做推断，使用beam search找出非top 1的句子作为该条数据的低噪声版本给模型学习还原。
		-	训练时的ground truth action：在确定了输入和输出之后，利用动态规划得到编辑距离消耗最小的操作作为ground truth action(expert policy)让模型学习

## 推断
-	推断时，直接对输入文本进行多轮编辑（每一轮包括一次删除、预测位置、插入），直接贪心的选择概率最大的操作。
-	多轮编辑直到：
	-	出现循环（即重复出现的编辑后文本）
	-	预设最大轮次
-	添加placeholder过多会导致生成的文本过短，可以根据实际情况对placeholder添加惩罚项
-	用两个编辑操作能够对模型的推断带来一定的可解释性

## 实验
-	指标结果主要包含六项，三类语对的翻译BLEU值以及摘要Gigaword的ROUGE-1,2，L值。在英日翻译和Gigaword全指标上，原始Transformer取得最好结果，比LevT好一个点左右，在剩下两个翻译数据集上LevT使用序列蒸馏的结果最好，比原始Transformer好一个点左右。
-	对于LevT，序列蒸馏来训练比用原始oracle序列普遍要好
-	在推断速度上，使用序列蒸馏的LevT效率最高，推断单句基本在90ms左右，而原始Transformer在翻译数据上要200-300ms，在摘要数据上也要116ms。作者还给出了推断迭代次数，这里可以看到LevT平均只要2次迭代，即两次删除+两次插入，注意这是对整句同时进行操作，而Transformer的平均迭代次数就是平均推断文本长度。短摘要数据集平均迭代10.1次，即平均长度10.1，而翻译数据集上则达到了20+。这样也能看出LevT的另一个优势，就是推断速度对于推断文本长度不是那么敏感，其主要和对整句的处理迭代次数相关，而迭代次数随长度的增长速度小于对数复杂度，远小于传统Transformer的平方复杂度。Transformer的推断延迟和长度基本成正比，而LevT都在90ms左右，两次迭代。
-	消融表明了两种操作的设计、参数共享、dual policy都能提升最终指标。
-	对删除和预测insert position直接用第一层结果early exit就能取得较好效果，相比取用第六层，BLEU降低0.4，换来加速比从3.5倍提升到5倍
-	同时作者发现，在翻译任务上训练的LevT可以zero-shot直接用于文本完善，效果比重新训练一个文本完善LevT最小只差了0.3个点，在Ro-En上甚至表现更好，且都优于传统Transformer。这里对于文本完善的适应性也符合直觉，完善文本自然是进行较小的编辑修改，完全重新生成的假设空间更大，难以达到理想目标。在文本完善数据集上微调之后，效果更是全面优于所有模型。

# EditNTS
-	提出了一个句子简化模型，显式地执行三类操作：插入、删除、保留
-	与LevT不同的是，EditNTS依然是自回归的，并没有同时对整句操作，并且在预测操作时引入了更多的信息
-	模型包含两部分：基于LSTM的seq2seq作为programmer，预测每个token位置的操作，产生编辑操作序列；以及一个额外的interpreter来执行操作生成编辑后的句子，这个interpreter还包含一个简单的LSTM来汇总编辑后句子的上下文信息
	[![gdTVa9.png](https://z3.ax1x.com/2021/05/12/gdTVa9.png)](https://imgtu.com/i/gdTVa9)
-	值得注意的有三点：
	-	预测出删除和保留后，programmer就移动到下一个单词，而预测出插入操作时，programmer不移动单词，接着在这个单词上预测下一个操作，来满足插入多个词的场景
	-	interpreter除了根据programmer预测结果完成编辑外，还用了一个LSTM来汇总当前时间步的编辑后句子信息
	-	programmer在预测编辑操作时，用到了四部分信息：
		-	encoder的context，来自encoder lstm last layer attention-weighted output
		-	当前操作的词，来自decoder lstm hidden state
		-	已经编辑好的句子的context，来自interpreter lstm attention-weighted output
		-	已经产生的编辑操作序列，来自attention-weighted edit label output，这里只用了简单的attention，没有用lstm
-	encoder输入还额外引入了pos embedding
-	标签构造，类似LevT，需要在句子简化数据集上构建出ground truth编辑操作，依然是基于编辑距离的动态规划得到操作序列，当多个操作序列存在时，优先选择插入序列多的，作者尝试过其他优先方式（优先删除、随机、引入替换操作），效果均不如直接优先插入。
-	使用100维的glove向量来初始化词向量和编辑label embedding，用30维向量初始化pos embedding，使用编辑标签逆频次作为损失权重来平衡各个标签的损失占比。
-	结果并不是很优秀，在指标上没有比非编辑方法好，在人类评估上好一些但存在主观性。消融也不是很显著。

# LaserTagger
-	依然是保留、删除、插入三种操作，不过用上了BERT，且google的工程能力使得加速比达到了100-200倍
-	贡献点
	-	提出了模型，并提出了从数据中生成标签词典的方法
	-	提出了基于BERT的和用BERT初始化seq2seq encoder的两个版本，前者加速快，后者效果好

## Text Editing as Tagging
-	LaserTagger将文本编辑问题转换为序列标注问题，主要包含三部分
	-	标注操作：将编辑转化标注，其标签并不是保留删除插入三种，而是分为两部分：base tag B，有保留和删除两种；added phrase P，有V种，代表插入的片段（空白、词或者短语），从训练数据中构建一个P vocabulary得到，P和B组合得到标签，因此标签总共有2V种。可以根据下游人物添加任务相关的标签。
	-	added phrase vocabulary构建：构建词典是一个组合优化问题，一方面我们希望这个词典尽可能小，另一方面我们希望这个词袋能够覆盖最多的编辑情况，完美的解决该问题是NP-hard的，作者采用了另外一种方式。对每个训练文本对，通过动态规划求出最长公共子序列，用目标序列减去最长公共子序列就得到了需要添加的phrase，之后对phrase按照出现频次排序，取top k，在公开数据集上top 500就能覆盖85%的数据。
	-	构建标注序列：直接见伪算法，这里不需要动态规划，而是采取了一种贪心的方式，逐字匹配，找最短phrase插入
		[![gdxc0x.png](https://z3.ax1x.com/2021/05/12/gdxc0x.png)](https://imgtu.com/i/gdxc0x)
		这样可能会出现不能编辑的情况，这种情况就从训练集中筛去，作者认为这样也可以看成是一种降噪。
	-	根据预测结果编辑句子：就是直接操作，可能根据下游任务预测出特别的标签进行任务相关的操作。作者认为将编辑操作模块化出来比直接端到端要更灵活。

## 实验
-	模型有两种，直接用BERT做标注或者用一个seq2seq，encoder用BERT初始化，前者的话就是在BERT最后一层是再加一层transformer block做标注。
-	四项任务，句子融合，分句与转述，摘要，语法纠错。均超过baseline，其中句子融合取得新的SOTA。但是LaserTagger最亮眼的表现在于其效率，只用BERT encoder能达到最高200倍的加速比（毕竟把文本生成变成了一个小词典的序列标注），同时对于小样本学习非常友好。
-	作者还分析了seq2seq和基于编辑的模型常出现的问题及其原因、两者的优势等等
	-	想象词：seq2seq会在subword粒度上组合生成不存在的词，LaserTagger不存在这种情况
	-	过早结束：seq2seq可能很早生成EOS导致句子不完整或者过短，而LaserTagger理论上存在这种可能，实际上作者没有见到过，因为这意味着模型要预测出一堆删除操作，而这在训练数据中是几乎没有的。
	-	重复短语：在分句任务中，seq2seq经常会重复生成短语，而LaserTagger会选择不分句，或者lazy split
	-	Hallucination：即生成的文本与源文本无关或者反常识反事实，两类模型都会出现这类问题，seq2seq可能隐藏的更深（看起来流畅，实际上不合理）
	-	共指消解：两类模型都存在这种问题，seq2seq容易将代词替换为错误名词，LaserTagger容易保留代词不管
	-	错误删除：LaserTagger看似文本生成较为可控，但任然存在一些情况，即模型删除了部分词，剩下句子语义流畅，但含义错误
	-	lazy split：在分句任务中，LasterTagger可能只分句，而不对分开的句子做任何后处理。

# PIE
-	也是BERT+序列标注，同时从数据中构造phrase词典
-	编辑操作包括复制（保留）、插入、删除、词形变换（用于语法纠错）
-	构造phrase词典、构造ground truth编辑操作和LaserTagger基本一致
-	将BERT左右扩展了两部分来获得替换或者增加所需的信息：
	-	输入层，M为mask标识符的embedding，p为positional embedding，X为word embedding
	-	h即原始BERT信息，包含词和位置信息
	-	r即替换信息，只不过把当前位置的词mask掉，用M替代，且计算注意力时不查询当前位置的h
	-	a即插入信息，只不过把当前位置的词mask掉，用M替代，且p替换为相邻位置p的平均
	[![gwZarT.png](https://z3.ax1x.com/2021/05/12/gwZarT.png)](https://imgtu.com/i/gwZarT)
	-	之后利用三类信息来分别计算不同操作的概率，并归一化，CARDT分别代表复制（保留）、插入、替换、删除、词形变换
	[![gwZ5IH.png](https://z3.ax1x.com/2021/05/12/gwZ5IH.png)](https://imgtu.com/i/gwZ5IH)
	-	上式第一项是每一项编辑操作的得分；第二项是保留当前词的得分，删除和替换不得分，其余的则用当前词embedding $\phi(x_i)$来参数化得分，这里$\phi$的意义不明，原文是Embedding of w, represented by $\phi(w)$ is obtained by summing up in-dividual output embeddings of tokens in w；第三项是新词带来的影响，只有替换和插入有。
-	推断时也是多轮推断（输出变输入），直到出现重复句子
-	这里编辑操作设计的动机和解释不是很清楚，最终结果也很一般，ensemble情况下才与Transformer + Pre-training + LM+ Spellcheck + Ensemble Decoding 的工作持平，而且从消融来看，预训练模型是带来性能提升的主要原因。

# Felix
-	将编辑操作分为两个非自回归的部分，首先用一个基于Transformer的指针网络做tagging和reordering，并插入placeholder；然后用一个MLM对placeholder做预测
-	Felix的设计考虑三点需求
	-	灵活的编辑，适合多种文本生成任务
	-	充分利用BERT等预训练模型
	-	高效推断
	[![gwKovR.png](https://z3.ax1x.com/2021/05/12/gwKovR.png)](https://imgtu.com/i/gwKovR)
-	下图可以清晰的表示出Felix的标签设计，其中*y^t*是tagging模型需要预测出的编辑操作序列，*y^m*是根据编辑操作序列，添加相关special token(REPL,MASK)之后的中间状态序列，中间状态序列直接喂给MLM来预测需要插入的词，最终得到Pred里的结果
-	[![gwMHzj.md.png](https://z3.ax1x.com/2021/05/12/gwMHzj.md.png)](https://imgtu.com/i/gwMHzj)
-	这里需要注意两点：
	-	MLM虽然只完成插入词的预测，但也需要完整的编辑操作信息，而不是输入MLM时直接把预测删除操作位置的词删掉，这里作者的做法是用REPL括起来要删除的span，给MLM也提供了删除编辑的信息。
	-	图中给出了Mask和Infill两种形式，其实就是对中间状态序列的MASK设计不同，涉及到多token插入的任务交给谁：Mask方式里是把任务交给了tagging模型，直接预测出插入多个词的编辑操作，例如$DEL^{INS\_2}$，即插入两个，对应的就在中间序列中生成两个MASK让模型预测；Infill方式里把这个任务交给了MLM，统一对每个插入位置准备四个MASK让模型预测，多了的部分就预测PAD
-	为了更灵活的建模，tagging部分的模型还需要做reordering，不然交互两个词的位置就得先删再添加，增加了复杂度。这里是通过指针网络注意力来为每个词确定其指向的后一个词，第一个词由特殊标识CLS指示。在推断时，使用了受控的beam search避免指向顺序产生循环。
-	主要与LaserTagger对比，在性能，小样本上的表现要优于LaserTagger,且没有了phrase vocabulary的限制。作者也就phrase vocabulary和reordering做了实验，结果很多可见原论文。


# 总结
-	总结各个模型
	
	| 模型 	| 编辑操作 	| 加速 	| 插入多个词 	| 构建ground truth编辑序列 	| 编辑失败 	| 测试任务 	|
|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|:-:	|
| LevT 	| 插入删除 	| 多轮非自回归，5倍 	| 先预测placeholder数量，再替换 	| 基于编辑距离动态规划 	| 退化为文本生成 	| 翻译、摘要 	|
| EditNTS 	| 插入删除保留 	| 自回归，未提及加速 	| 插入时原地停留，直到插入完毕 	| 基于编辑距离动态规划 	| 退化为文本生成 	| 文本简化 	|
| LaserTagger 	| 插入删除保留 	| 一轮序列标注，100+倍 	| 直接插入phrase，根据训练数据构建phrase词典 	| 贪心匹配 	| 不编辑 	| 句子融合，分句与转述，摘要，语法纠错 	|
| PIE 	| 插入删除替换复制词形变换 	| 序列标注，2倍 	| 直接插入phrase，根据训练数据构建phrase词典 	| 贪心匹配 	| 未说明 	| 语法纠错 	|
| Felix 	| 删除插入保留重排序 	| 标注+MLM，快于LaserTagger 	| 设置多个Mask让MLM预测 	| 简单比较 	| 较差的MLM预测结果 	| 句子融合、机器翻译后处理、摘要、文本简化 	|