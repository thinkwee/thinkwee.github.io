---
title: 论文阅读笔记2018下半年
date: 2018-07-03 15:18:52
tags:
  - abstractive summarization
  - math
  - machine learning
  -	theory
  -	nlp
categories:
  - 机器学习
author: Thinkwee
mathjax: true
html: true
---

-	分散注意力机制
-	外部信息注意力
-	指针复制网络PGNet
-	基于RNN的抽取式摘要
-	Transformer
-	选择门机制

<!--more--> 

![i0o47d.jpg](https://s1.ax1x.com/2018/10/20/i0o47d.jpg)

# Distraction-Based Neural Networks for Document Summarization
-	不仅仅使用注意力机制，还使用注意力分散机制，来更好地捕捉文档的整体含义。实验证明这种机制对于输入为长文本时尤其有效。
![i0oh0H.png](https://s1.ax1x.com/2018/10/20/i0oh0H.png)
-	在编码器和解码器之间引入控制层，实现注意力集中和注意力分散，用两层GRU构成：
	$$
	s_t = GRU _1 (s_t^{temp},c_t) \\
	s_t^{temp} = GRU _2 (s_{t-1},e(y_{t-1})) \\
	$$
-	这个控制层捕捉$s_t^{'}$和$c_t$之间的联系，前者编码了当前及之前的输出信息，后者编码经过了注意力集中和注意力分散处理的当前输入，而$e(y_{t-1})$是上一次输入的embedding。
-	三种注意力分散模型
	-	M1：计算c_t用于控制层，在输入上做分散，其中c_t是普通的注意力编码出来的上下文c_t^{temp}，减去了历史上下文得到，类似coverage机制
	$$
	c_t = tanh (W_c c_t^{temp} - U_c \sum _{j=1}^{t-1} c_j) \\
	c_t^{temp} = \sum _{i=1}^{T_x} \alpha _{t,i} h_i \\
	$$
	-	M2：在注意力权重上做分散，类似的，也是减去历史注意力，再做归一化
	$$
	\alpha _{t,i}^{temp} = v_{\alpha}^T tanh(W_a s_t^{temp} + U_a h_i - b_a \sum _{j=1}^{t-1}\alpha _{j,i}) \\
	\alpha _{t,i} = \frac {exp(\alpha _{t,i}^{temp})}{\sum _{j=1}^{T_x} exp(\alpha _{t,j}^{temp})} \\
	$$
	-	M3：在解码端做分散，计算当前的$c_t$，$s_t$，$\alpha _t$和历史的$c_t$，$s_t$，$\alpha _t$之间的距离，和输出概率一起作为解码时束搜索所依赖的得分。
	$$
	d_{\alpha , t} = \min KL(\alpha _t , \alpha _i) \\
	d_{c , t} = \max cosine(c _t , c _i) \\
	d_{s , t} = \max cosine(s _t , s _i) \\
	$$

# Document Modeling with External Attention for Sentence Extraction
-	构造了一个抽取式文摘模型，由分层文档编码器和基于外部信息注意力的抽取器组成。
在文摘任务中，外部信息是图片配字和文档标题。
-	通过隐性的估计每个句子与文档的局部和全局关联性，显性的考虑外部信息，来决定每句话是否应该加入文摘。

![i0oLjS.png](https://s1.ax1x.com/2018/10/20/i0oLjS.png)

-	句子级编码器：如图所示，使用CNN编码，每个句子用大小为2和4的卷积核各三个，卷积出来的向量做maxpooling最后生成一个值，因此最后生成的向量为6维。
-	文档级编码器：将一个文档的句子6维向量依次输入LSTM进行编码。
-	句子抽取器：由带注意力机制的LSTM构成，与一般的生成式seq2seq不同，句子的编码不仅作为seq2seq中的编码输入，也作为解码输入，且一个是逆序一个是正序。抽取器依赖编码端输入$s_t$，解码端的上一时间步状态$h_t$，以及进行了注意力加权的外部信息$h_t^{'}$：

![i0oIAA.png](https://s1.ax1x.com/2018/10/20/i0oIAA.png)

# Get To The Point: Summarization with Pointer-Generator Networks
-	介绍了两种机制，Pointer-Generator解决OOV问题，coverage解决重复词问题
-	Pointer-Generator:通过context，当前timestep的decoder状态及输入学习到指针概率
	$$
	p_{gen} = \sigma (w_h^T h_t + w_s^T s_t + w_x^T x_t +b_{ptr}) \\
	P(w) = p_{gen} P_{vocab}(w) + (1-p_{gen}) \sum _{i:w_i = w} a_i^t \\
	$$
-	指针概率指示是否应该正常生成，还是从输入里按照当前的注意力分布采样一个词汇复制过来，在上式中，如果当前的label是OOV，则左边部分为0，最大化右边使得注意力分布能够指示该复制的词的位置；如果label是生成的新词（原文中没有），则右边部分为0，最大化左边即正常的用decoder生成词。综合起来学习正确的指针概率。

![i0ootI.png](https://s1.ax1x.com/2018/10/20/i0ootI.png)

-	Coverage:使用coverage机制来修改注意力，使得在之前timestep获得了较多注意力的词语之后获得较少注意力
-	普通注意力计算

	$$
	e_i^t = v^T tanh(W_h h_i + W_s s_t + b_{attn}) \\
	a^t = softmax(e^t) \\
	$$

-	维护一个coverage向量，表示每个词在此之前获得了多少注意力:

	$$
	c^t = \sum _{t^{temp} = 0}^t-1 a^{t^{temp}}
	$$

-	然后用其修正注意力的生成，使得注意力生成考虑了之前的注意力累积

	$$
	e_i^t =v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_{attn})
	$$

-	并在损失函数里加上coverage损失

	$$
	covloss_t = \sum _i \min (a_i^t , c_i^t)
	$$

-	使用min的含义是，我们只惩罚每一个attention和coverage分布重叠的部分，也就是coverage大的，如果attention也大，那covloss就大；coverage小的，不管attention如何，covloss都小

# SummaRuNNer A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents

![i0oTht.png](https://s1.ax1x.com/2018/10/20/i0oTht.png)

-	用RNN做抽取式文摘,可以用可视化展示模型决策过程，并且使用了一种端到端的训练方法
-	将抽取视为句子分类任务，对每个句子按原文顺序依次访问，决定是否加入文摘，且这个决策考虑了之前决策的结果。
-	用一层双向GRU在词级别上编码，再用一层双向GRU在句子级别上编码，两层输出的编码都经过了正反拼接和均值pooling

	$$
	d = tanh(W_d \frac {1}{N_d} \sum _{j=1}^{N^d} [h_j^f,h_j^b]+b)
	$$

-	其中d是整篇文档的编码，$h_j^f$和$h_j^b$代表句子经过GRU的正反向编码
-	之后根据整篇文档的编码、句子的编码以及文摘在当前句子位置的动态表示来训练一个神经网络做二分类，决定每个句子是否应该加入文摘：

![i0ob1f.png](https://s1.ax1x.com/2018/10/20/i0ob1f.png)

-	其中sj为到j位置为止已经产生的文摘的表示，用每个句子的二分类概率对之前句子的编码加权求和得到：

	$$
	s_j = \sum _{i=1}^{j-1} h_i P(y_i = 1 | h_i,s_i,d)
	$$

-	第一行：参数为当前句子编码，表示当前句子的内容
-	第二行：参数为文档编码和句子编码，表示当前句子对文档的显著性
-	第三行：参数为句子编码和文摘动态编码，表示当前句对已产生文摘的冗余。（We squash the summary representation using the tanh operation so that the magnitude of summary remains the same for all time-steps.）
-	第四行和第五行：考虑了句子在文档中的相对位置和绝对位置。（The absolute position denotes the actual sentence number, whereas the relative position refers to a quantized representation that divides each document into a fixed number of segments and computes the segment ID of a given sentence.）
-	最后对整个模型做最大似然估计:

	$$
	l(W,b) = -\sum _{d=1}^N \sum _{j=1}^{N_d} (y_j^d log P(y_j^d = 1 | h_j^d,s_j^d,d_d)+(1-y_j^d)log(1-P(y_j^d=1|h_j^d,s_j^d,d_d)))
	$$

-	作者将这种抽取式方法应用在生成式文摘语料上，也就是如何用生成式的文摘为原文中每个句子打上二分类的label。作者认为label为1的句子子集应该和生成式文摘ROUGE值最大，但是找出所有的子集太费时，就用了一种贪心的方法：一句一句将句子加入子集，如果剩下的句子没有一个能使当前子集ROUGE值上升，就不加了。这样将生成式文摘语料转换为抽取式文摘语料。
-	还有一种方式，直接在生成式文摘语料上做训练，将上面提到的动态文摘表示，取它最后一句也就是包含了整个文档的文摘表示s，输入一个解码器，解码出来生成式文摘。因为文摘表示是解码器的唯一输入，训练解码器的同时也能学习到好的文摘表示，从而完成抽取式文摘的任务。
-	因为在生成二分类概率时包含了几个部分，将它们归一化可以得到各个部分做出的贡献，从而可视化决策过程：

![i0oqc8.png](https://s1.ax1x.com/2018/10/20/i0oqc8.png)

# Attention Is All You Need
-	抛弃了RNN和CNN做seq2seq任务，直接用multi head attention组成网络块叠加，加入BN层和残差连接构造深层网络

![i0oXng.png](https://s1.ax1x.com/2018/10/20/i0oXng.png)

-	完全使用attention的一个好处就是快。
-	为了使用残差，所有的子模块（multi-head attention和全连接）都统一输出维度为512
-	编码端：6个块，每个块包含attention和全连接两个子模块，均使用了残差和bn。
-	解码端：也是6个块，不同的是加了一个attention用于处理编码端的输出，而且与解码端输入相连的attention使用了mask，保证了方向性，即第i个位置的输出只与之前位置的输出有关。
-	编码与解码的6个块都是堆叠的(stack)，
-	Attention的通用模型是指将一个query和一系列键值对映射到输出的一种机制，其中输出是值的加权和，而每个值的权重将对应的键和query输入一个兼容性函数计算得到，传统的attention键和值相同，都是输入每个位置上的隐藏层状态，query就是当前输出，兼容性函数就是各种attention计算方法。图中指向attention的三个箭头分别代表key,value,query。

![i0TSNn.png](https://s1.ax1x.com/2018/10/20/i0TSNn.png)

-	Multi-head attention由多个scaled dot-product attention并行组成。
-	Scaled dot-product attention如图所示，query和key先做点积，再放缩，如果是解码器输入的attention还要加上mask，之后过softmax函数与value做点积得到attention权重。实际计算时为了加速都是一系列query,key,value一起计算，所以Q,K,V都是矩阵。做放缩是为了防止点积attention在k的维度过大时处于softmax的两端，梯度小。

	$$
	Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt {d_k}}) V
	$$

-	Multi-head attention就是有h个scaled dot-product attention作用于V,K,Q的h个线性投影上，学习到不同的特征，最后拼接并进行线性变换。作者认为这种multi-head的设计能使模型学习到不同位置的表示子空间的信息。

	$$
	MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^o \\
	where \ \  head_i = Attention(QW_i^Q,KW_i^K,VW_i^V) \\
	$$

-	论文中取h=8个head，为了保证维度一致，单个q,k,v的维度取512/8=64
-	这种multi-head attention用在了模型的三个地方：
	-	编码解码之间，其中key,value来自编码输出，query来自解码块中masked multi-head attention的输出。也就是传统的attention位置
	-	编码端块与块之间的自注意力
	-	解码端块与块之间的自注意力
-	在每个块里还有一个全连接层，这个层包含两个线性变换，中间插入了ReLU激活，且每个输入位置都有相同的参数，但不同的块的全连接层参数不同

	$$
	FFN(x) =\max (0,xW_1+b_1)W_2 +b_2 \\
	$$

-	完全使用注意力的话会抛弃了序列的顺序信息，为了利用这部分信息，加入了三角函数位置编码来利用相对位置信息：

	$$
	PE_{(pos,2i)} = sin(pos/10000 ^{2i/d_{model}}) \\
	PE_{(pos,2i+1)} = cos(pos/10000 ^{2i/d_{model}}) \\
	$$

# A Joint Selective Mechanism for Abstractive Sentence Summarization

![ivQCE8.png](https://s1.ax1x.com/2018/11/15/ivQCE8.png)

-	文摘不同于翻译，端到端框架应该对损失（信息压缩）建模，而不是和翻译一样单纯对对齐建模
-	作者针对损失建模，做了两点改进：
	-	在编码完成之后添加了一个门限用于裁剪编码信息
	-	添加了一个选择损失，同时关注输入和输出，辅助门限工作
-	选择门限同时考虑了编码之后的隐藏层状态和原始词嵌入，并作用于隐藏层状态之上，对隐层向量做裁剪，之后再经过注意力加权生成上下文。作者认为这个过程相当于让网络观察rnn处理前后的词嵌入，能够知道输入中的哪个单词对于产生文摘很重要：
$$
g_i = \sigma (W_g h_i + U_g u_i) \\
h_i^{'} = h_i \cdot g_i \\
$$
-	而选择损失函数则是在解码端构造了一个回顾门限，考虑了编码端的隐层和原始输入，解码端的隐层和原始输入，解码端每一个位置的回顾门限是对编码端所有位置的回顾门限求平均：
$$
r_{i,t} = \sigma (W_r h_i + U_r u_i + V_r s_{t-1} + Q_r w_{t-1}) \\
r_i = \frac 1m \sum _{t=2}^{m+1} r_{i,t} \\
$$
-	作者认为回顾门限的作用相当于让网络阅读产生的文摘，并回顾输入文本，使其知道学会如何挑选文摘。
-	之后用选择门限和回顾门限的欧氏距离作为选择损失，加入总损失中:
$$
d(g,r) = \frac 1n \sum _{i=1}^n |r_i - g_i | \\
L = -p(y|x,\theta) + \lambda d(g,r) \\
$$
-	作者并没有说明为什么将回顾门限和选择门限之间的欧式距离作为损失函数，也没有说明选择门限和注意力的区别，感觉就像是考虑了原始输入embedding的一种注意力机制，且在传统注意力加权之前先对隐层每一时间步做了裁剪。选出来的可视化特例也很精巧，恰恰说明了这个选择机制能识别句子中的转折，因而改变了选择的词，这还是和之前选择门限提出的原论文对比。原论文Selective Encoding for Abstractive Sentence Summarization 也没有说出这种设计的动机。