---

title: Note for Variational Auto-Encoder
date: 2019-03-20 09:53:31
categories: ML
tags:

- vae
- math
- mcmc
mathjax: true
html: true

---


*   Variational Autoencoder Learning Notes
    
*   Reference Article:
    
    *   [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
    *   [Daniel Daza, The Variational Autoencoder](https://dfdazac.github.io/01-vae.html)
    *   [Sūshén's VAE series](https://spaces.ac.cn/tag/vae/)
*   On VAE, the original paper and the two blogs above have already explained it very clearly. I am just repeating and paraphrasing, just to go through it myself. If anyone reads this blog, I recommend reading these three reference sources first
  
<!--more-->

{% language_switch %}

{% lang_content en %}
Directly view the network structure
===================================

*   Variational autoencoders use variational inference, but because the parameter estimation part employs the gradient descent method of neural networks, their network structure can be directly depicted—we actually refer to them as autoencoders, and this is also because their structure has many similarities with autoencoders. Even if one does not start from a Bayesian perspective, VAE can be directly regarded as a type of special autoencoder.
*   Taking the mnist experiment from the original paper as an example, we directly examine the network structure of the VAE, and then generalize the model and explain the details:
    *   An encoder and a decoder, the goal is to reconstruct the error and obtain useful encoding, just like an autoencoder
    *   However, variational autoencoders do not directly encode the input, but rather assume that the encoding follows a multivariate normal distribution; the encoder encodes the mean and variance of this multivariate normal distribution
    *   That is, VAE assumes that the encoding is simple, following a normal distribution, while what I train and utilize is the decoding, this decoder can decode and reconstruct the input from samples of the normal distribution, or in other words, generate the output
*   mnist input is 28\*28, batch\_size is 128, assuming the hidden layer dim is 200, and the parameter dim is 10, then the entire network is:

1. Input $x$ \[128,28\*28\], pass through a linear+ReLu, obtain encoder hidden layer $h$ \[128,200\]
2. Through two linear transformations, obtain normal distribution parameters $\mu _e$ \[128,10\], $\log \sigma _e$ \[128,10\]
3. From a standard multivariate normal distribution, sample $\epsilon \sim N(0,I)$ , to obtain $\epsilon$ \[128,10\]
4. Combine the parameters obtained through the network $\mu _e$ , $\log \sigma _e$ to the sampling values of the standard multidimensional normal distribution, $z = \mu _e + \sigma _e \bigodot \epsilon$ \[128,10\], $z$ is the encoding
5. encoder part is completed, next is the decoder part: encoding $z$ through overlinear + ReLu to obtain the decoder hidden layer state h\[128,200\]
6. Decoding the hidden state h through linear + sigmoid yields $\mu _d$ \[128, 28\*28\], i.e., the output after decoding
7. Decoding output $\mu _d$ and input $x$ to calculate the Bernoulli cross-entropy loss
8. In addition, a regularity-like loss term should be added $\frac 12 \sum _{i=1}^{10} (\sigma _{ei}^2 + \mu _{ei}^2 -log(\sigma _{ei}^2) - 1)$

Direct Network Analysis
=======================

*   From the network, the biggest difference between VAE and AE is that it does not directly encode the input but introduces the concept of probability into the network structure, constructing an encoding that satisfies a multi-dimensional normal distribution for each input
*   One advantage of this approach is that the encoding can perform interpolation, achieving a continuous change in generated images: the original AE has a fixed encoding for each specific input, while the network in VAE is only responsible for generating fixed mean and variance, i.e., a fixed normal distribution. The actual encoding is just a sampling from this fixed normal distribution, still uncertain. During the training process, VAE trains an area rather than a point, so the obtained encoding has continuity, and similar images can be generated near the center point of the area. Moreover, interpolation can be performed between the two encoding areas for the two types of image inputs, realizing a smooth transition between the two generated images.
*   Step 1, 2, 3, and 4 involve the encoder sampling an encoding from a distribution of $N(\mu _e,\sigma _e ^2)$ , however, there is a reparameterization trick, namely:
    *   The encoder neural network should originally be fitted to a distribution satisfying $N(\mu _e,\sigma _e ^2)$ , and then samples taken from the distribution
    *   However, the values obtained from the sampling cannot be backpropagated
    *   Therefore, the neural network is modified to only fit the parameters of the distribution, then samples from a simple standard multivariate normal distribution, and the sampled values are processed by the fitting parameters, i.e., $z = \mu _e + \sigma _e \bigodot \epsilon$ , to achieve the effect of $z$ as if directly sampled from $N(\mu _e,\sigma _e ^2)$ , while the neural network merely fits the parameters, allowing for backpropagation, and the sampling is equivalent to performing some weighted operations with specified weights, participating in the network training
*   Step 5, 6, and 7 are ordinary decoding, as the output is a 28\*28 black and white image, so it is directly decoded into a 28\*28 binary vector, and compared with the input to calculate cross-entropy
*   The key is 8, how is this regular term obtained?

Regular term
============

*   This regular term actually represents the KL divergence between the encoded normal distribution and the standard normal distribution, i.e., (where K is the dimension of the multivariate normal distribution, which is 10 in the above example):
    
    $$
    KL(N(\mu _e,\sigma _e ^2)||N(0,I)) =  \frac 12 \sum _{i=1}^{K} (\sigma _{ei}^2 + \mu _{ei}^2 -log(\sigma _{ei}^2) - 1)
    $$
    
*   That is to say, we hope the normal distribution we encode is close to the standard normal distribution, why?
    
*   There are many different interpretations here:
    
    *   The first type: We hope that for different classes of inputs, the encoding can be encoded into the same large area, that is, while the regions within are compact, the distance between regions should not be too far, and it is best to reflect the distance in terms of image features, for example, taking mnist as an example, the images of 4 and 9 are relatively similar, while the difference with the image of 0 is large, then the distance between their encoding regions can reflect the similarity; or, during the process from 0 to 8, the intermediate states will resemble 9, then the encoding region of 9 should be between the encoding regions of 0 and 8. However, in reality, the encoder network may learn such an encoding method: for different classes of inputs, the difference $\mu$ is large, it separates the encoding regions of different class inputs (more accurately, non-similar inputs, here in unsupervised learning, there are no classes) quite far apart. The neural network does this for a reason: to make it easier for the decoder to distinguish between different inputs during decoding. This goes against our original intention of encoding them into continuous regions for easy interpolation. Therefore, we strongly hope that the learned encoding distribution is approximately a standard normal distribution, so that they are all in a large area, of course, not too similar, otherwise everyone is the same, the decoder's burden is too heavy, and it cannot decode the differences, which is the role of the reconstruction loss mentioned earlier.
    *   The second type: The effect of VAE is equivalent to adding Gaussian noise to the standard autoencoder, making the decoder robust to noise. The size of the KL divergence represents the strength of the noise: a smaller KL divergence indicates that the noise is closer to the standard Gaussian noise, i.e., stronger; a larger KL divergence indicates a weaker noise strength, here understood as the noise being assimilated, not that the variance has decreased, because the noise should be unrelated to the input signal and always maintain Gaussian noise or other specified distributions. If the noise becomes increasingly distant from the specified distribution and more related to the input, its role as noise diminishes accordingly.
    *   The third: This is the most rigorous understanding, where the KL divergence is obtained from the perspective of variational inference, and the entire model is derived from Bayesian framework reasoning. The network structure exists because the author uses neural networks to fit the parameters, and the specification of the hyperparameters and distributions is a special case of this framework in the mnist generation task, after all, the original text refers to it as autoencoder variational Bayesian (a method), not variational autoencoder (a structure). Next, let's look at how the entire model is derived from the perspective of the original paper, and naturally obtain this KL divergence regularization term.

Variational Autoencoder Bayesian
================================

*   The entire decoder section can be regarded as a generative model, with its probability graph being: ![AKu5FA.png](https://s2.ax1x.com/2019/03/20/AKu5FA.png) 
    
*   $z$ is the encoding, $\theta$ is the decoder parameters we hope to obtain, controlling the decoder to decode (generate) from the encoding ( $x$ )
    
*   The problem now returns to the inference of probabilistic graphical models: Given the observed variable x, how to obtain the parameter $\theta$ ?
    
*   The author's approach is not a complete copy of variational inference; in VAE, the $q$ distribution is also used to approximate the posterior distribution $p(z|x)$ . The log-likelihood of the observables is decomposed into ELBO and KL(q||p(z|x)). The difference is that in variational inference, q is obtained using the EM method, while in VAE, q is fitted using a neural network (the input of the neural network is $z$ , and therefore $q$ itself is also a posterior distribution $q(z|x)$ .
    
    $$
    \log p(x|\theta) = \log p(x,z|\theta) - \log p(z|x,\theta) \\
    $$
    
    $$
    = \log \frac{p(x,z|\theta)}{q(z|x,\phi)} - \log \frac{p(z|x,\theta)}{q(z|x,\phi)} \\
    $$
    
    $$
    = \log p(x,z|\theta) - \log q(z|x,\phi) - \log \frac{p(z|x,\theta)}{q(z|x,\phi)} \\
    $$
    
    $$
    = [ \int _z q(z|x,\phi) \log p(x,z|\theta)dz - \int _z q(z|x,\phi) \log q(z|x,\phi)dz ] + [- \int _z \log \frac{p(z|x,\theta)}{q(z|x,\phi)} q(z|x,\phi) dz ]\\
    $$
    
*   Note that we actually aim to obtain the parameters $\theta$ and $\phi$ that maximize the logarithmic likelihood of the observations, while the latent variable $z$ can be obtained with the model given the input.
    
*   It can be seen that, under the condition of the measurement, i.e., the log-likelihood, the larger the value in the previous brackets, i.e., the ELBO, the closer the subsequent KL divergence, i.e., the posterior distribution $q(z|x,\phi)$ and the posterior true distribution $p(z|x,\theta)$ . This posterior distribution, i.e., given $x$ , to obtain $z$ , is actually the encoder, so the smaller the KL divergence, the better the encoder's performance. Therefore, we should maximize the ELBO. The ELBO can be rewritten as:
    
    $$
    ELBO = \int _z q(z|x,\phi) \log p(x,z|\theta)dz - \int _z q(z|x,\phi) \log q(z|x,\phi)dz \\
    $$
    
    $$
    = E_{q(z|x,\phi)}[\log p(x,z|\theta)-\log q(z|x,\phi)] \\
    $$
    
    $$
    = E_{q(z|x,\phi)}[\log p(x|z,\theta)]-KL(q(z|x,\phi)||(p(z|\theta))) \\
    $$
    
*   Another KL divergence has appeared! This KL divergence is the KL divergence between the posterior distribution of the latent variables encoded by the encoder and the prior distribution of the latent variables. The first part, $p(x|z,\theta)$ , which calculates the distribution of the observable variables from the known latent variables, is actually the decoder. Therefore, $\phi$ and $\theta$ correspond to the parameters of the encoder and decoder, respectively, which are actually the parameters of the neural network. The former is called the variational parameter, and the latter is called the generative parameter.
    
*   We aim to maximize this ELBO, and the VAE directly uses it as the objective function of the network structure, performing gradient descent and taking derivatives with respect to $\theta$ and $\phi$ . In this case, the first part $E_{q(z|x,\phi)}[\log p(x|z,\theta)]$ calculates the expectation using the Monte Carlo method, i.e., sampling multiple $z$ from $z \sim q(z|x,\phi)$ , and then calculating the mean to find the expectation, where the reparameterization technique mentioned above is applied.
    
*   At this point, the entire probabilistic graphical model, including the inference part, becomes ![AKuhod.png](https://s2.ax1x.com/2019/03/20/AKuhod.png) 
    
*   Process: Obtain the observation x -> Obtain samples of z through reparameterization -> Input the samples of z into the target function (ELBO) for differentiation -> Gradient descent, update parameters $\theta$ and $\phi$
    

Return to mnist
===============

*   In the MNIST experiment, the authors set the prior of the latent variables, the q distribution, the base distribution in reparameterization $\epsilon$ , and the posterior distribution of the observations to be:
    
    $$
    p(z) = N(z|0,I) \\
    $$
    
    $$
    q(z|x,\phi) = N(z|\mu _e , diag(\sigma _e)) \\
    $$
    
    $$
    \epsilon \sim N(0,I) \\
    $$
    
    $$
    p(x|z,\theta) = \prod _{i=1}^D \mu _{d_i}^{x_i} (1-\mu _{d_i})^{1-x_i} \\
    $$
    
*   The model parameters $\phi = [\mu_e , \sigma _e]$ , $\theta=\mu _d$ are obtained through neural network learning
    
*   The first part of the ELBO objective function, the expectation part, has already been completed through reparameterization, the internal
    
    $$
    \log p(x|z,\theta) = \sum _{i=1}^D x_i \log \mu _{d_i} + (1-x_i) \log (1- \mu _{d_i}) \\
    $$
    
*   Bernoulli cross-entropy, where in network design the sigmoid function is added to the last layer, is to ensure that the output $mu_d$ satisfies the probability.
    
*   The latter part of the ELBO objective function, i.e., the KL divergence between the posterior q distribution of the latent variables and the prior p distribution, becomes the regularization term mentioned above, making the approximate distribution closer to the prior distribution
    
*   The entire model considers both the reconstruction loss and the prior information
    
*   Therefore, the ELBO can be written as:
    
    $$
    ELBO = - reconstruction loss - regularization term
    $$
    

Effect
======

*   Reconstruction Effect on the MNIST Dataset ![AKufdH.png](https://s2.ax1x.com/2019/03/20/AKufdH.png) 
*   The effect obtained from variance disturbance ![AKuWee.png](https://s2.ax1x.com/2019/03/20/AKuWee.png) 
*   The effect of mean perturbation ![AKu2LD.png](https://s2.ax1x.com/2019/03/20/AKu2LD.png) 
*   Interpolation results for 4 and 9 ![AKuIJI.png](https://s2.ax1x.com/2019/03/20/AKuIJI.png) 


{% endlang_content %}

{% lang_content zh %}

# 直接看网络结构

- 变分自编码器用了变分推断，但是因为其中的参数估计部分用的是神经网络的梯度下降方法，因此可以直接画出其网络结构——实际上我们称其为自编码器，也是因为其结构上和自编码器有许多相通之处，如果不从贝叶斯的角度出发，甚至可以将VAE直接看成一类特殊的自编码器。
- 以原论文的mnist实验为例，我们直接看VAE的网络结构，之后再一般化模型并解释细节：
  - 整体和自编码器一样，一个encoder和一个decoder,目标是重构误差，获取有用的编码
  - 然而变分自编码器不对输入直接编码，而是假定编码服从多维正态分布，encoder编码的是这个多维正态分布的均值和方差
  - 也就是VAE假定编码很简单，就是服从正态分布，而我要训练出来并利用的是解码，这个解码器能从正态分布的采样中解码还原出输入，或者说，生成输出
- mnist的输入为28\*28，batch_size为128，假定隐层dim为200，参数dim为10，则整个网络为：
1. 输入$x$[128,28\*28]，过一个linear+ReLu，得到编码器隐层$h$[128,200]
2. $h$分别过两个linear，得到正态分布的参数$\mu _e$[128,10]，$\log \sigma _e$[128,10]
3. 从一个标准多维正态分布中采样$\epsilon \sim N(0,I)$，得到$\epsilon$[128,10]
4. 组合通过网络得到的参数$\mu _e$，$\log \sigma _e$到标准多维正态分布的采样值中，$z = \mu _e + \sigma _e \bigodot \epsilon$[128,10]，$z$即编码
5. encoder部分就此完成，接下来是decoder部分：编码$z$过linear+ReLu得到解码隐层状态h[128,200]
6. 解码隐层状态h经过linear+sigmoid得到$\mu _d$[128,28\*28]，即解码后的输出
7. 解码后输出$\mu _d$与输入$x$计算伯努利交叉熵损失
8. 此外还要加上一个类似正则项的损失$\frac 12 \sum _{i=1}^{10} (\sigma _{ei}^2 + \mu _{ei}^2 -log(\sigma _{ei}^2) - 1)$

# 直接对网络分析

- 从网络可以看到，VAE相比AE最大的区别就是不直接对输入编码，而是将概率的思想引入了网络结构，对每一个输入单独构建一个满足多维正态分布的编码
- 这么做的一个好处是，编码可以做插值，实现生成图像连续的变化：原始的AE对于每一个确定输入有确定的编码，而VAE中的网络只负责生成确定的均值和方差，也就是确定的正态分布，实际的编码只是在这个确定的正态分布中采样得到，依然是不确定的。在训练过程中VAE是对一片区域进行训练，而不是点训练，因此得到的编码具有连续性，在区域中心点附近也能生成相似的图像，甚至可以在两类图像输入确定的两个编码区域之间插值，实现两种生成图像的平滑过渡。
- 步骤1，2，3，4实现的是编码器从一个$N(\mu _e,\sigma _e ^2)$的分布中采样得到编码，然而这里存在一个reparameterization的技巧，即：
  - 本来应该是让编码器神经网络拟合一个满足$N(\mu _e,\sigma _e ^2)$的分布，再从分布中采样
  - 但是采样得到的值无法进行反向传播
  - 因此改成神经网络只拟合分布的参数，然后从一个简单的标准多维正态分布中采样，采样值经过拟合参数处理，即$z = \mu _e + \sigma _e \bigodot \epsilon$，来达到$z$仿佛直接从$N(\mu _e,\sigma _e ^2)$采样得到的效果，而神经网络仅仅拟合参数，可以进行反向传播，采样在其中相当于做了一些指定权值的加权，参与进网络的训练
- 步骤5，6，7则是普通的解码，因为输出是28\*28的黑白图像，因此直接解码成28\*28的二值向量，与输入比对计算交叉熵
- 关键是8，这个正则项如何得到？

# 正则项

- 这个正则项实际上是编码得到的正态分布和标准正态分布之间的KL散度，即（其中K是多维正态分布的维度，在上例中是10）：
  
  $$
  KL(N(\mu _e,\sigma _e ^2)||N(0,I)) =  \frac 12 \sum _{i=1}^{K} (\sigma _{ei}^2 + \mu _{ei}^2 -log(\sigma _{ei}^2) - 1)
  $$
- 也就是说，我们希望编码的正态分布接近标准正态分布，为什么？
- 这里就有很多种说法了：
  - 第一种：我们希望的是对于不同类的输入，编码能编码到同一个大区域，即不同区域内部紧凑的同时，区域之间的距离也不应该太远，最好能体现图像特征上的距离，比如以mnist为例，4和9的图像比较近似，和0的图像差异比较大，则他们的编码区域之间的距离能反映相似的关系；或者说从0变到8的过程中中间状态会像9，那么9的编码区域能在0和8的编码区域之间最好。然而实际上是，编码器网络可能会学到这样的编码方法：对于不同类的输入，$\mu$相差很大，它将不同类输入（准确的说是不近似的输入，这里是无监督学习，没有类别）的编码区域隔得很开。神经网络这么做是有道理的：让解码器方便区别不同的输入进行解码。这就与我们希望它编码成连续区域方便插值的初衷相悖，因此我们强制希望所学到的编码分布都近似标准正态分布，这样都在一个大区域中，当然也不能太近似，不然大家都一样，解码器负担太大，根本解码不出来区别，这就是前面的重构损失的作用。
  - 第二种：VAE的效果相当于在标准的自编码器中加入了高斯噪声，使得decoder对噪声具有鲁棒性。KL散度的大小代表了噪声的强弱：KL散度小，噪声越贴近标准高斯噪声，即强度大；KL散度大，噪声强度就小，这里理解为噪声被同化了，而不是说方差变小了，因为噪声应该与输入信号无关，一直保持高斯噪声或者其他指定的分布，如果噪声变得和指定分布越来越远，和输入越来越相关，那其作为噪声的作用也就越来越小了。
  - 第三种：也是最严谨的一种理解，这个KL散度是从变分推断的角度出发得到的，整个模型也是从贝叶斯框架推理得到的。其之所以有网络结构是因为作者用了神经网络来拟合参数，神经网络的超参、分布的指定也是该框架在mnist生成任务中的一种特例，毕竟原文叫自编码变分贝叶斯（一种方法），而不是变分自编码网络（一种结构）。接下来我们从原论文的角度来看看整个模型如何推导出来，并自然而然得到这个KL散度正则项。

# 变分自编码贝叶斯

- 整个解码器部分我们可以看成一个生成模型，其概率图为：
  ![AKu5FA.png](https://s2.ax1x.com/2019/03/20/AKu5FA.png)
- $z$即编码，$\theta$是我们希望得到的解码器参数，控制解码器从编码中解码出（生成出）$x$
- 现在问题回归到概率图模型的推断：已知观测变量x，怎么得到参数$\theta$？
- 作者采取的思路并不是完全照搬[变分推断](https://thinkwee.top/2018/08/28/inference-algorithm/#more)，在VAE中也采用了$q$分布来近似后验分布$p(z|x)$，并将观测量的对数似然拆分成ELBO和KL(q||p(z|x))，不同的是变分推断中用EM的方式得到q，而在VAE中用神经网络的方式拟合q（神经网络输入为$z$，因此$q$本身也是后验分布$q(z|x)$。完整写下来：
  
  $$
  \log p(x|\theta) = \log p(x,z|\theta) - \log p(z|x,\theta) \\
  $$
  
  $$
  = \log \frac{p(x,z|\theta)}{q(z|x,\phi)} - \log \frac{p(z|x,\theta)}{q(z|x,\phi)} \\
  $$
  
  $$
  = \log p(x,z|\theta) - \log q(z|x,\phi) - \log \frac{p(z|x,\theta)}{q(z|x,\phi)} \\
  $$
  
  $$
  = [ \int _z q(z|x,\phi) \log p(x,z|\theta)dz - \int _z q(z|x,\phi) \log q(z|x,\phi)dz ] + [- \int _z \log \frac{p(z|x,\theta)}{q(z|x,\phi)} q(z|x,\phi) dz ]\\
  $$
-  注意，我们实际希望得到的是使得观测量对数似然最大的参数$\theta$和$\phi$，而隐变量$z$可以在输入确定的情况下随模型得到。
- 可以看到，在观测量即对数似然确定的情况下，前一个中括号内即ELBO值越大，则后面的KL散度，即后验$q(z|x,\phi)$分布和后验真实分布$p(z|x,\theta)$越相近。这个后验分布，即已知$x$得到$z$实际上就是编码器，因此这个KL散度越小则编码器效果越好，既然如此我们就应该最大化ELBO，ELBO可以改写成：
  
  $$
  ELBO = \int _z q(z|x,\phi) \log p(x,z|\theta)dz - \int _z q(z|x,\phi) \log q(z|x,\phi)dz \\
  $$
  
  $$
  = E_{q(z|x,\phi)}[\log p(x,z|\theta)-\log q(z|x,\phi)] \\
  $$
  
  $$
  = E_{q(z|x,\phi)}[\log p(x|z,\theta)]-KL(q(z|x,\phi)||(p(z|\theta))) \\
  $$
- 又出现了一个KL散度！这个KL散度是编码器编码出的隐变量后验分布和隐变量先验分布之间的KL散度。而前半部分，$p(x|z,\theta)$，已知隐变量求出观测量的分布，实际上就是解码器。因此$\phi$和$\theta$分别对应编码器和解码器的参数，实际上即神经网络的参数。前者称为variational parameter，后者称为generative parameter
- 我们要使得这个ELBO最大，VAE就直接将其作为网络结构的目标函数，做梯度下降，分别对$\theta$和$\phi$求导。在这里前半部分$E_{q(z|x,\phi)}[\log p(x|z,\theta)]$求期望，用的是蒙特卡罗方法，即从$z \sim q(z|x,\phi)$中采样多个$z$，再求均值求期望，这里用到了上面说到的reparameterization技巧。
- 此时整个概率图模型，加上推断部分，变成了
  [![AKuhod.png](https://s2.ax1x.com/2019/03/20/AKuhod.png)](https://imgchr.com/i/AKuhod)
- 流程：得到观测量x->通过reparameterization得到z的样本->将z的样本带入目标函数（ELBO）求导->梯度下降，更新参数$\theta$和$\phi$

# 回到mnist

- 在mnist实验中，作者设置隐变量的先验、q分布、reparameterization中的基础分布$\epsilon$、观测量的后验分布分别为：
  
  $$
  p(z) = N(z|0,I) \\
  $$
  
  $$
  q(z|x,\phi) = N(z|\mu _e , diag(\sigma _e)) \\
  $$
  
  $$
  \epsilon \sim N(0,I) \\
  $$
  
  $$
  p(x|z,\theta) = \prod _{i=1}^D \mu _{d_i}^{x_i} (1-\mu _{d_i})^{1-x_i} \\
  $$
- 其中模型参数$\phi = [\mu_e , \sigma _e]$,$\theta=\mu _d$通过神经网络学习得到
- 而目标函数ELBO的前半部分，求期望部分已经通过reparameterization完成，内部的
  
  $$
  \log p(x|z,\theta) = \sum _{i=1}^D x_i \log \mu _{d_i} + (1-x_i) \log (1- \mu _{d_i}) \\
  $$
- 即伯努利交叉熵，在网络设计是最后一层增加sigmoid函数也就是为了输出的$mu_d$满足为概率。
- 目标函数ELBO的后半部分，即隐变量的后验q分布和先验p分布之间的KL散度，就成为了上面所说的正则项,使得近似分布靠近先验分布
- 整个模型既考虑了重构损失，也考虑了先验信息
- [因此ELBO可以写成](https://dfdazac.github.io/01-vae.html)：
  
  $$
  ELBO = -重构误差损失-正则惩罚
  $$

# 效果

- 在mnist数据集上的重构效果
  [![AKufdH.png](https://s2.ax1x.com/2019/03/20/AKufdH.png)](https://imgchr.com/i/AKufdH)
- 对方差扰动得到的效果
  [![AKuWee.png](https://s2.ax1x.com/2019/03/20/AKuWee.png)](https://imgchr.com/i/AKuWee)
- 对均值扰动得到的效果
  [![AKu2LD.png](https://s2.ax1x.com/2019/03/20/AKu2LD.png)](https://imgchr.com/i/AKu2LD)
- 对4和9进行插值的结果
  [![AKuIJI.png](https://s2.ax1x.com/2019/03/20/AKuIJI.png)](https://imgchr.com/i/AKuIJI)


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