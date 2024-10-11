---
title: CLSciSumm summary
date: 2020-03-27 14:04:23
categories: NLP
tags:
  - machine learning
  - workshop
  - natural language processing
mathjax: true
html: true
---

总结一下实验室参加的CLSciSumm Workshop，主要是方法，实验分析论文里很详细
论文：

- [2016](http://ceur-ws.org/Vol-1610/paper18.pdf)
- [2017](http://ceur-ws.org/Vol-2002/cistclscisumm2017.pdf)
- [2018](http://ceur-ws.org/Vol-2132/paper8.pdf)
- [2019](http://ceur-ws.org/Vol-2414/paper20.pdf)
  
  <!--more-->

# Task

- task 1a:给定论文CP和被引论文RP，给定CP当中的citation，找出RP中被这个citation引用的cited text span(CTS)，即content linking，本质上是一个句子对相似度计算任务
- task 1b:在找出CTS之后，判断这个text span属于RP中的哪一个facet，即文本分类任务
- task 2:生成RP的摘要，属于自动文摘任务

# 2016

## task 1a

- 将输入的text span pair转换成特征向量，送入分类器判断两个text span是否link
- 特征包括
  - RP当中的高频词，用wordnet和word vector扩充
  - RP和CP一起训练LDA，得到主题特征
  - CTS和citance的词共现频率
  - idf：共现词在RP句子集当中的idf
  - 两个text span的jaccard 相似度
  - 考虑上下文的相似度，即当前句的前后两句与待匹配句的相似度相乘再开方
  - word2vec得到的两个text span相似度，这里两个句子当中的词相似度最大值，再针对两句长度做对称归一化得到
  - doc2vec，直接得到句向量，做相似度计算
- 分类器：SVM和人为赋予权重直接打分
- 可以看到这个任务的数据集严重不平衡，实际上不匹配的样本数量是匹配的样本数量的125倍，作者尝试切分负样本，分别训练了125个SVM并通过投票得到结果，但是效果非常差，因此采取了直接人为赋予权重打分的方法
- jaccard距离的效果最好，以此为主要打分特征，根据其余特征的单独实验效果赋予权重作为补充特征

## task 1b

- 基于规则的
  - facet指 Hypothesis, Implication, Aim, Results and Method之类的论文部分，假如句子中包含这些词就直接分类
  - 计算每个facet下的高频词，并扩充。设定阈值，当句子高频词的数量超过阈值之后，就将对应facet加入候选集，选择coverage最高的facet作为结果
- SVM
  - 提取四种特征：段落位置、文档位置比例、段落位置比例、RCTS位置
- 投票
  - 结合上面的所有结果投票
- 融合
  - 每一个1a的run都能得到一次CTS的结果，对每一次结果计算一个1b的run，取最好的

## task 2

- 提取特征：
  - hLDA：层次主题特征。hLDA的特征有两种利用方式：共享同一条路径的句子具有相似的主题分布，所以可以先进行句子的聚类，另外在评测任务当中我们还可以将facet作为聚类结果；另一种方式就是计算每个词的hlda得分，得分由两部分组成：词所在层的层（被分配主题）权重*词在所在层上的概率 + 当前主题节点内该词的概率。通过经验发现，若以三层hlda建模，高层是最抽象的词，底层是最具体的词，中间层次的词的抽象程度比较容易出现在摘要句当中，因此中间层赋予较高权重。
  - 句子长度：对gold summary的句子长度做一次高斯建模
  - 句子位置
  - 1a特征：假如是1a中提取到的CTS，那么就基于一个弱分数，因为1a的结果包含错误
  - RST特征：基于修辞结构理论的特征
- 加权特征进行打分，做了一些细节操作
  - 将结果句当中的第一人称改为第三人称
  - 为每一个facet或者每一个hlda聚类结果单独抽取句子
  - 移除重复度高的句子

# 2017

## task 1a

- 在2016的基础上添加了
  - 利用Wordnet计算相同pos的词之间的相似度，共6种：cn, lin, lch, res, wup and path similarit。利用与word2vec特征同样的处理方法将词相似度转成句相似度
  - 用CNN训练，计算两个句子的相似度，再将CNN的结果作为打分特征的一种

## task 1b

- 基本同2016年一样，除了一些SVM训练细节稍有不同

## task 2

- 相比前一年主要引入了行列式点过程采样来保证摘要结果质量与多样性的均衡
- 在训练hlda时，不仅仅将RP作为单篇文档的语料，也引入了该RP相关的所有citation
- 特征
  - hlda的特征增加了句子的主题分布
  - 增加了标题相似度作为特征
- 引入了行列式点过程抽样，将句子作为待抽样的点，给定每个点的质量（得分）以及点之间的相似度，采样出一个子集（摘要），使得子集的质量高且子集内元素相似度低：
  ![Gic2ef.png](https://s1.ax1x.com/2020/03/27/Gic2ef.png)

# 2018

## task 1a

- 相比前一年，作者采用了WMD距离作为特征向量之间的相似度度量，将其应用于task 1a的similarity feature以及task2中DPPs
- 改进了LDA特征的利用。之前只是将隐主题作为词典，现在还计算了两个句子的LDA分布相似度，不仅考虑了属于同一主题的词的数量，还考虑了内部的主题分布

## task 1b

- 尝试了很多机器学习方法，包括SVM、DT、KNN、RF、GB，但只有RF取得了和规则打分方法相当的效果

## task 2

- 依然是规则打分+DPPs，只不过在构建DPPs所需的L矩阵时，采用了WMD计算相似度

# 2019

## task 1a

- 在计算LDA相似度时，采用jaccard距离，因为LDA的主题分布通常很稀疏
- 使用CNN时，采用了Word2Vec_H特征：
  - 先用SVD对两个句子的embedding矩阵进行降维
  - 再计算词级别的相似度矩阵，$L_{ij}$是句子a的i号词和句子b的j号词的降维之后的词向量计算cosine距离
  - 将L矩阵作为CNN的输入

## task 1b

- 添加了CNN作为分类方法，但是依然比不过传统特征打分

## task 2

- 将WMD距离经过反比例映射、线性映射、和指数映射压缩入[0,1]区间内
- 构建L矩阵时除了QS分解，还采取了Gram分解，这样不需要显式计算特征与相似度，只需要输入每一句的特征向量即可。尝试了word2vec与LSA构建特征向量
