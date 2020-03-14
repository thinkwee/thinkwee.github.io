---
title: 支持向量机一些细节补充
date: 2020-02-13 22:04:46
categories: 自然语言处理
tags:
  - svm
  - machine learning
  -	math
mathjax: true
html: true
---

一些推导关键步骤

<!--more-->

# 目标函数
-	最原始的想法自然是考虑几何间隔
$$
\begin{array}{cc}{\max _{\vec{w}, b} \gamma} \\ {s . t . \quad \tilde{y}_{i}\left(\frac{\overrightarrow{\mathbf{w}}}{\|\overrightarrow{\mathbf{w}}\|_{2}} \cdot \overrightarrow{\mathbf{x}}_{i}+\frac{b}{\|\overrightarrow{\mathbf{w}}\|_{2}}\right) \geq \gamma, i=1,2, \cdots, N}\end{array}
$$
-	但是几何间隔可以用函数间隔表示，且函数间隔可以缩放而不影响分类超平面的选择，因此才有令函数间隔等于1，再取倒数把max换成min，化简了目标函数
$$
\begin{array}{c}{\min _{\vec{w}, b} \frac{1}{2}\|\overrightarrow{\mathbf{w}}\|_{2}^{2}} \\ {\text {s.t.} \quad \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}} \cdot \overrightarrow{\mathbf{x}}_{i}+b\right)-1 \geq 0, i=1,2, \cdots, N}\end{array}
$$

# min-max
-	在将问题定义为带有不等式约束的最优化问题之后，就要用到拉格朗日对偶性来将原始问题转为对偶问题
-	统计学习方法的描述如下：
	-	对于最优化问题：
	$$
	\begin{array}{ll}{\min _{x \in \mathbf{R}^{n}} f(x)} & {} \\ {\text { s.t. } \quad c_{i}(x) \leqslant 0, \quad i=1,2, \cdots, k} \\ {\qquad \begin{array}{ll}{h_{j}(x)=0,} & {j=1,2, \cdots, l}\end{array}}\end{array}
	$$
	-	引入广义拉格朗日函数
	$$
	L(x, \alpha, \beta)=f(x)+\sum_{i=1}^{k} \alpha_{i} c_{i}(x)+\sum_{j=1}^{I} \beta_{j} h_{j}(x)
	$$
	-	定义
	$$
	\theta_{P}(x)=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
	$$
	-	可以判断，假如不满足约束条件的话，可以令拉格朗日乘子不为0从而使上式无穷大，满足情况的话，上式要最大只能让$\alpha$为0，$\beta$则不起作用，因此：
	$$
	\theta_{P}(x)=\left\{\begin{array}{l}{f(x)} \ x满足原始问题约束 \\ {+\infty} \ 其他 \end{array}\right.
	$$
-	这样原始的最小化一个f就转换为最小化一个最大化的$\theta$，即
	$$
	\min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
	$$

# 对偶
-	max和min对换位置就得到对偶问题，即先针对$x$优化，再针对$\alpha,\beta$优化
	$$
	\begin{array}{l}{\max _{\alpha, \beta} \theta_{D}(\alpha, \beta)=\max _{\alpha, \beta} \min _{x} L(x, \alpha, \beta)} \\ {\text { s.t. } \quad \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, k}\end{array}
	$$
-	对偶问题和原始问题的关系：
	$$
	d^{*}=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} \min _{x} L(x, \alpha, \beta) \leqslant \min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)=p^{*}
	$$
-	证明：先看两边的里面一部分，左边是$\min \ L$，右边是$\max \ L$，尽管自变量不一样，但是当自变量固定时，必然有$\min _{x} L(x, \alpha, \beta) \leqslant L(x, \alpha, \beta) \leqslant \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)$，现在把里面整体看成一个函数，只看外面，即比较左边的$\max f_1$和$\min f_2$，由上可知$f_1$处处小于等于$f_2$，那即便是$f_1$的最大值，也一定小于等于$f_2$的最小值，也就是所谓的“鸡头小于凤尾”，前提是鸡群里的每一只鸡小于等于凤群里的每一只凤。
-	SVM满足强对偶关系，即上式取到等号，因此优化原问题可以转化为优化对偶问题，方便引入核函数。
-	原始的min-max问题可以直接解，但不是对min-max形式直接求偏导，因为先对$\alpha$求偏导没有意义，我们是希望得到$w,b$使得对任意的$\alpha \neq 0$，$\max L$都可以取到最小值。

# 求解
-	转换为对偶问题之后，先对$w,b$求导令其为0，之后将得到的$w,b$回代入对偶问题得到：
$$
\begin{aligned} \max _{\vec{\alpha}}-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\ s . t . \quad \sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}=0 \\ \alpha_{i} \geq 0, i=1,2, \cdots, N \end{aligned}
$$
-	其中等式约束条件是对$b$求导为0得到的。之前说到SVM的目标函数和约束条件满足强对偶关系，强对偶关系的充要条件是KKT条件，因此上式应该也满足KKT条件，即
	-	拉格朗日函数对$w,b$求偏导为0
	-	拉格朗日函数中原问题的约束部分（求和的每一项）为0，即松弛互补条件
	-	原始问题约束
	-	拉格朗日乘子非负
	$$
	\begin{aligned} \nabla_{\overrightarrow{\mathrm{w}}} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \overrightarrow{\mathrm{w}}^{*}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i} \overrightarrow{\mathrm{x}}_{i}=0 \\ \nabla_{b} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}=0 \\ \alpha_{i}^{*}\left[\tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1\right]=0, i=1,2, \cdots, N \\ \tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1 \geq 0, i=1,2, \cdots, N \\ \alpha_{i}^{*} \geq 0, i=1,2, \cdots, N \end{aligned}
	$$
-	由KKT条件，我们可以用$\alpha$表示$w$（之前求导时也已经得到过），我们知道$w$代表分类超平面的方向，$b$代表偏置，由支持向量决定，因此那些$\alpha _j$不为0对应的
$$
\tilde{y}_{i}\left(\overrightarrow{\mathbf{w}}^{*} \cdot \overrightarrow{\mathbf{x}}_{i}+b^{*}\right)-1
$$
	决定了$b$（因为$\alpha$不为0，由松弛互补条件，则后一项为0，就可以求得$b$）。最后得到：
	$$
	b^{*}=\tilde{y}_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}\left(\overrightarrow{\mathrm{x}}_{i} \cdot \overrightarrow{\mathrm{x}}_{j}\right)
	$$
	可以看到，$w,b$都是求和的形式，但大部分为0，只有不为0的$\alpha$（在$b$的表达式里直接找出不为0的记为$\alpha _j$）项才起作用，即支持向量机只有少数的支持向量决定。
-	因此，给定数据，先求得对偶问题极大值，得到$\alpha$，再由$\alpha$中不为0的部分计算出$w,b$，得到超平面。
-	软间隔和非线性支持向量机的定义和求解过程类似，只不过约束条件和目标函数不同。
-	$\alpha$的求解涉及凸二次规划，有很多解法。支持向量机的一个优点就是学习到的参数只依赖支持向量，推理时避免了维度灾难，但是在学习的过程中，需要对所有样本计算最优化，因此对于大规模数据不友好，这里可以用SMO算法来优化


