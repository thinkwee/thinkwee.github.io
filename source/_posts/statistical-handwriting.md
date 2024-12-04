---

title: Statistical Learning - A hand-write note
date: 2018-08-09 10:03:46
categories: ML
tags:

- code
- machine learning
- statistical learning
- math

mathjax: true
html: true 

---

***
The ten major algorithms of statistical learning methods have been simplified and handwritten out (although I think the book itself is already quite concise). Now there is only the process of the algorithms themselves; in the future, if I have any new understandings, I will supplement them. The writing is ugly, even I can't bear to look at it, so I post it purely as a backup

<!--more-->

{% language_switch %}

{% lang_content en %}

Abstract
========

![i0HLjK.jpg](https://s1.ax1x.com/2018/10/20/i0HLjK.jpg)

![i0Hb1x.jpg](https://s1.ax1x.com/2018/10/20/i0Hb1x.jpg)

![i0HoN9.jpg](https://s1.ax1x.com/2018/10/20/i0HoN9.jpg)

Perceptron
==========

![i0HH91.jpg](https://s1.ax1x.com/2018/10/20/i0HH91.jpg)

![i0HThR.jpg](https://s1.ax1x.com/2018/10/20/i0HThR.jpg)

k-Nearest Neighbors
===================

![i0Hqc6.jpg](https://s1.ax1x.com/2018/10/20/i0Hqc6.jpg)

![i0HXnO.jpg](https://s1.ax1x.com/2018/10/20/i0HXnO.jpg)

Naive Bayes
===========

![i0HjBD.jpg](https://s1.ax1x.com/2018/10/20/i0HjBD.jpg)

Decision Tree
=============

*   GBDT is written in the improvement method, and it is also possible to expand and look at random forests, which is an bootstrap method that utilizes decision trees. ![i0HzAH.jpg](https://s1.ax1x.com/2018/10/20/i0HzAH.jpg) 

![i0HvHe.jpg](https://s1.ax1x.com/2018/10/20/i0HvHe.jpg)

![i0bP3t.jpg](https://s1.ax1x.com/2018/10/20/i0bP3t.jpg)

![i0bSNd.jpg](https://s1.ax1x.com/2018/10/20/i0bSNd.jpg)

Logistic regression, maximum entropy
====================================

*   The mutual derivation between maximum entropy and logistic regression is to be supplemented ![i0bigP.jpg](https://s1.ax1x.com/2018/10/20/i0bigP.jpg) 

![i0bp4A.jpg](https://s1.ax1x.com/2018/10/20/i0bp4A.jpg)

Support Vector Machine
======================

![i0bFjf.jpg](https://s1.ax1x.com/2018/10/20/i0bFjf.jpg)

![i0bAu8.jpg](https://s1.ax1x.com/2018/10/20/i0bAu8.jpg)

![i0bEDS.jpg](https://s1.ax1x.com/2018/10/20/i0bEDS.jpg)

![i0beEQ.jpg](https://s1.ax1x.com/2018/10/20/i0beEQ.jpg)

![i0bVHg.jpg](https://s1.ax1x.com/2018/10/20/i0bVHg.jpg)

![i0bmNj.jpg](https://s1.ax1x.com/2018/10/20/i0bmNj.jpg)

Enhancement Methods
===================

*   XGBoost ![i0bn4s.jpg](https://s1.ax1x.com/2018/10/20/i0bn4s.jpg) 

![i0bKCn.jpg](https://s1.ax1x.com/2018/10/20/i0bKCn.jpg)

Expectation-Maximization (EM) algorithm
=======================================

![i0bM3q.jpg](https://s1.ax1x.com/2018/10/20/i0bM3q.jpg)

![i0bQg0.jpg](https://s1.ax1x.com/2018/10/20/i0bQg0.jpg)

*   When performing inference on Gaussian mixture models using the EM algorithm, the parameters to be inferred include the mean, variance, and proportion coefficient of k Gaussian models. The latent variable represents the probability that the jth observation sample belongs to the kth Gaussian model, referred to as responsibility, while $n_k$ is the sum of the responsibility of the kth Gaussian model over all samples, divided by $N$ , which is used to update the GMM proportion coefficient with its mean. The mean is updated using responsibility-weighted samples, and the variance is updated in a similar manner.
*   After updating the parameters, recalculate responsibility with these parameters, recalculate the E-step, and then proceed to the M-step, thereby completing the iteration.

Hidden Markov
=============

![i0b3uT.jpg](https://s1.ax1x.com/2018/10/20/i0b3uT.jpg)

![i0blvV.jpg](https://s1.ax1x.com/2018/10/20/i0blvV.jpg)

![i0b8DU.jpg](https://s1.ax1x.com/2018/10/20/i0b8DU.jpg)

![i0bGbF.jpg](https://s1.ax1x.com/2018/10/20/i0bGbF.jpg)

Conditional Random Field
========================

*   Address the solutions for three types of problems, as conditional random fields are the conditional extension of the hidden Markov model, and the algorithms are similar ![i0bYE4.jpg](https://s1.ax1x.com/2018/10/20/i0bYE4.jpg) 



{% endlang_content %}

{% lang_content zh %}


# 概论

![i0HLjK.jpg](https://s1.ax1x.com/2018/10/20/i0HLjK.jpg)

![i0Hb1x.jpg](https://s1.ax1x.com/2018/10/20/i0Hb1x.jpg)

![i0HoN9.jpg](https://s1.ax1x.com/2018/10/20/i0HoN9.jpg)

# 感知机

![i0HH91.jpg](https://s1.ax1x.com/2018/10/20/i0HH91.jpg)

![i0HThR.jpg](https://s1.ax1x.com/2018/10/20/i0HThR.jpg)

# k近邻

![i0Hqc6.jpg](https://s1.ax1x.com/2018/10/20/i0Hqc6.jpg)

![i0HXnO.jpg](https://s1.ax1x.com/2018/10/20/i0HXnO.jpg)

# 朴素贝叶斯

![i0HjBD.jpg](https://s1.ax1x.com/2018/10/20/i0HjBD.jpg)

# 决策树

- GBDT写在了提升方法里，另外可以扩展看看随机森林，是一个自举的方法，利用了决策树。
  ![i0HzAH.jpg](https://s1.ax1x.com/2018/10/20/i0HzAH.jpg)

![i0HvHe.jpg](https://s1.ax1x.com/2018/10/20/i0HvHe.jpg)

![i0bP3t.jpg](https://s1.ax1x.com/2018/10/20/i0bP3t.jpg)

![i0bSNd.jpg](https://s1.ax1x.com/2018/10/20/i0bSNd.jpg)

# 逻辑斯蒂回归、最大熵

- 待补充最大熵和逻辑斯蒂回归之间的相互推导
  ![i0bigP.jpg](https://s1.ax1x.com/2018/10/20/i0bigP.jpg)

![i0bp4A.jpg](https://s1.ax1x.com/2018/10/20/i0bp4A.jpg)

# 支持向量机

![i0bFjf.jpg](https://s1.ax1x.com/2018/10/20/i0bFjf.jpg)

![i0bAu8.jpg](https://s1.ax1x.com/2018/10/20/i0bAu8.jpg)

![i0bEDS.jpg](https://s1.ax1x.com/2018/10/20/i0bEDS.jpg)

![i0beEQ.jpg](https://s1.ax1x.com/2018/10/20/i0beEQ.jpg)

![i0bVHg.jpg](https://s1.ax1x.com/2018/10/20/i0bVHg.jpg)

![i0bmNj.jpg](https://s1.ax1x.com/2018/10/20/i0bmNj.jpg)

# 提升方法

- 待补充XGBoost
  ![i0bn4s.jpg](https://s1.ax1x.com/2018/10/20/i0bn4s.jpg)

![i0bKCn.jpg](https://s1.ax1x.com/2018/10/20/i0bKCn.jpg)

# EM算法

![i0bM3q.jpg](https://s1.ax1x.com/2018/10/20/i0bM3q.jpg)

![i0bQg0.jpg](https://s1.ax1x.com/2018/10/20/i0bQg0.jpg)

- 用EM算法做高斯混合模型的推断时，需要推断的参数包括k个高斯模型的均值、方差、比例系数，隐变量代表第j个观测样本来自第k个高斯模型的可能，叫做responsibility，而$n_k$则是对第k个高斯模型在所有样本上的responsibility的总和，除以$N$即以其均值来更新GMM比例系数，用responsibility加权样本来更新均值，方差同理。
- 在更新完参数之后，再用这些参数重新计算responsibility，重新计算E步骤，再继续做M步骤，从而完成迭代。

# 隐马尔可夫

![i0b3uT.jpg](https://s1.ax1x.com/2018/10/20/i0b3uT.jpg)

![i0blvV.jpg](https://s1.ax1x.com/2018/10/20/i0blvV.jpg)

![i0b8DU.jpg](https://s1.ax1x.com/2018/10/20/i0b8DU.jpg)

![i0bGbF.jpg](https://s1.ax1x.com/2018/10/20/i0bGbF.jpg)

# 条件随机场

- 待补充三种问题的解法，因为条件随机场是隐马尔可夫模型的条件化扩展，算法也类似
  ![i0bYE4.jpg](https://s1.ax1x.com/2018/10/20/i0bYE4.jpg)

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