---
title: BERT解剖综述
date: 2020-03-02 16:40:55
categories: 自然语言处理
tags:
  - bert
  - deep learning
  -	natural language processing
mathjax: true
html: true
---

翻译A Primer in BERTology: What we know about how BERT works一文，系统的介绍了近年来对于BERT可解释性以及扩展性方面的研究。
原论文[arxiv pdf](https://arxiv.org/pdf/2002.12327.pdf)

<!--more-->

# BERT embeddings
-	BERT作为一个NLU编码器，其生成的embedding是上下文相关的，关于embedding的研究有
	-	BERT embedding形成了明显的聚类，与word sense相关
	-	也有人发现，相同单词的embedding随着Position不同有差别，且该现象貌似与NSP任务有关
	-	有人研究了同一单词在不同层的representation，发现高层的表示更加与上下文相关，且越高层，embedding在高维空间中越密集（occupy a narrow cone in the vector space），在各向同性的情况下，两个随机单词之间的cosine距离比想象中的更为接近

# Syntactic knowledge
-	BERT的表示是层次而非线性的，其捕捉到的更像是句法树的信息而不是词序信息。
-	BERT编码了pos,chunk等信息，但是并没有捕捉完整的句法信息（有些远距离的依赖被忽略）
-	句法信息没有直接编码在self attention weight当中，而是需要转换
-	BERT在完形填空任务中考虑了主谓一致（subject-predicate agreement）
-	BERT不能理解否定，对于malformed input不敏感
-	其预测不会因为词序颠倒、句子切分、主谓语移除而改变
-	即BERT编码了句法信息，但是没有利用上

# Semantic knowledge
-	通过在MLM任务中设置探针，一些研究表明BERT能编码一些语义角色信息
-	BERT编码了实体类型、关系、语义角色和proto-roles
-	由于wordpiece的预处理，BERT在数字编码、推理上表现的并不好

# World knowledge
-	在某些关系中，BERT比基于知识库的方法更好，只要有好的模板句，BERT可以用于抽取知识
-	但是BERT不能利用这些知识进行推理
-	另外有研究发现BERT的知识是通过刻板的字符组合猜出来的，并不符合事实it would predict that a person with an Italian-sounding name is Italian, even when it is factually incorrect.

# Self-attention heads
-	研究发现attention heads可以分成几类
	-	attend to自己、前后单词、句子结尾
	-	attend to前后单词、CLS、SEP,或者在整个序列上都有分布
	-	或者是以下5种
	![3Wlqsg.png](https://s2.ax1x.com/2020/03/02/3Wlqsg.png)
-	attention weight的含义：计算该词的下一层表示时，其他的单词如何加权
-	self attention并没有直接编码语言学信息，因为大部分的head都是heterogeneous或者vertical的，与参数量过多有关
-	少数的head编码了词的句法角色
-	单一的head无法捕捉完整的句法树信息
-	即便一些head能够捕捉语义关系，它们也不是带来相关任务上的提升的必须条件

# layers
-	底层包含了最多的线性词序关系，越高层，词序信息越弱，知识信息越强
-	BERT的中间层包含最强的句法信息，甚至可以设法还原句法树
-	BERT中间层的迁移表现和性能最好
![3WJDg0.png](https://s2.ax1x.com/2020/03/02/3WJDg0.png)
-	但是该结论存在冲突，有些人发现底层做chunking更好，高层做parsing更好，有些人则发现中间层做tagging和chunking都是最好的
-	fine-tune时，底层不变对性能影响不大；最后一层在微调时变化最大
-	语义信息在各个层中都存在

# pre-training
-	原始的任务是MLM和NSP，有研究提出了更好的训练目标
	-	NSP移除影响不大，尤其在多语言版本中
	-	NSP可以扩展为预测前后两句，也可以将前后翻转的句子作为negative sample，而不是从其他的文档中随便找一句作为negative sample
	-	dynamic mask可以改善性能
	-	Beyond-sentence MLM，将句子替换为任意字符串
	-	Permutation language modeling，即XLNET当中的打乱单词顺序，再从左往右预测，结合了非回归和自回归的特点，既考虑了上下文，又不会有mask导致pre-training和fine-tune目标不一致
	-	Span boundary objective，只用span边界的词来预测span
	-	Phrase masking and named entity masking
	-	Continual learning，持续学习
	-	Conditional MLM，将segmentation embedding替换为label embedding以适应序列标注任务
	-	replacing the MASK token with [UNK] token
-	另外一条改进路线是数据集，有些研究试图将结构化数据融入BERT的pre-training，更为常见的融入常识信息的是加入entity embedding或者semantic role information，例如E-BERT、ERNIE、SemBERT
-	关于是否需要预训练，预训练使得模型更鲁棒，但是依然看任务，有些任务从头训练和预训练差别不大

# Model architecture
-	层数比head数更重要
-	大的batch能加速模型的收敛，with a batch size of 32k BERT’s training time can be significantly reduced with no degradation in performance
-	A robustly optimized BERT pretraining approach 公开了一些最优的参数设定
-	因为高层的一些self attention weight和底层很像，所以可以先训练浅层，在把浅层参数复制到深层，能带来25%的训练效率提升

# fine-tuning
-	有人认为fine-tuning是告诉BERT该忽略哪些信息
-	一些fine-tune的建议
	-	考虑多层的加权输出，而不仅仅使用最后一层去predict
	-	Two-stage fine-tuning
	-	Adversarial token perturbations
-	可以插入adapter modules来加速fine-tune
-	初始化很重要，但没有论文针对初始化系统的试验过

# Overparametrization
-	BERT没有很好的利用其庞大的参数，大部分head可以被裁剪掉
-	一层的head大部分相似，甚至可以将一层的head数裁剪为一个
-	有些层和head会降低模型的性能
-	在主谓一致、主语检测任务上，大的BERT模型表现反而不如小的
-	同一层使用同一个MLP、attention dropout可能导致了这样的head冗余现象

# Compression
-	主要两种方式：量化与知识蒸馏
-	还有一些方式例如progressive model replacing、对embedding矩阵做分解、化多层为循环单层等等。

# Multilingual BERT
-	多语言版本的BERT在许多任务的零次学习迁移上表现非常好，但是在语言生成任务上表现不佳
-	一些改进mBERT的手段
	-	fine-tune时固定低层
	-	translation language modeling 
	-	improving word alignment in fine-tuning
	-	combine 5 pre-training tasks (monolingual and cross-lingual MLM, translation language modeling, cross-lingual word recovery and paraphrase classification）