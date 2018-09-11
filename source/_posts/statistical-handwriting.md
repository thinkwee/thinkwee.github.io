---
title: 统计学习方法手写版笔记
date: 2018-08-09 10:03:46
categories: 机器学习
tags:
- code
- machine learning
- statistical learning
- math

mathjax: true
html: true
photo: http://ojtdnrpmt.bkt.clouddn.com/blog/180809/8mFd81eem2.PNG
---
***
把统计学习方法十大算法精简了一些手写了出来（虽然我觉得书本身已经很精简了）
现在只有算法本身的流程，以后如果有什么新的理解再补充
字太丑，自己都看不下去，发上来纯粹做个备份

<!--more-->
# 概论
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/1lg8jI0Bfe.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/dg79KfJ6LG.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/BiH0Ae7hf3.jpg?imageslim)
# 感知机
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/5mH2Lc8EKA.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/GEebi0EGfA.jpg?imageslim)
# k近邻
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/lBbBE0iAbd.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/diCeKIIKjd.jpg?imageslim)
# 朴素贝叶斯
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/LDadChj6HF.jpg?imageslim)
# 决策树
-	GBDT写在了提升方法里，另外可以扩展看看随机森林，是一个自举的方法，利用了决策树。
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/6Hee6Cgh7I.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/D4agk9eK5e.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/mF6BD8AFm8.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/Kb0chj4hBI.jpg?imageslim)
# 逻辑斯蒂回归、最大熵
-	待补充最大熵和逻辑斯蒂回归之间的相互推导
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/ADGijmf5lh.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/j2eFgF2JL9.jpg?imageslim)
# 支持向量机
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/g2H4e4kfKL.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/2G6eFkEAF2.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/iBafKl6LLc.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/0eckEhF2ID.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/9A8mffeIe5.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/lk4KHEA8f4.jpg?imageslim)
# 提升方法
-	待补充XGBoost
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/AAFA9K0D8g.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/a39G37jjCH.jpg?imageslim)
# EM算法
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/4EaGEHG3fC.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/fi8EcEa2Li.jpg?imageslim)

-	用EM算法做高斯混合模型的推断时，需要推断的参数包括k个高斯模型的均值、方差、比例系数，隐变量代表第j个观测样本来自第k个高斯模型的可能，叫做responsibility，而$n_k$则是对第k个高斯模型在所有样本上的responsibility的总和，除以$$N$即以其均值来更新GMM比例系数，用responsibility加权样本来更新均值，方差同理。
-	在更新完参数之后，再用这些参数重新计算responsibility，重新计算E步骤，再继续做M步骤，从而完成迭代。

# 隐马尔可夫
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/B8a4fcL3F8.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/KfH43akdkI.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/ggL7E50ckI.jpg?imageslim)

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/E9EFhCgL5E.jpg?imageslim)
# 条件随机场
-	待补充三种问题的解法，因为条件随机场是隐马尔可夫模型的条件化扩展，算法也类似
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/180809/Bk5Bj6dfEg.jpg?imageslim)
