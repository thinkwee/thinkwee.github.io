---
title: SVM
date: 2020-02-13 22:04:46
categories: NLP
tags:
  - svm
  - machine learning
  - math
mathjax: true
html: true
---

<img src="https://i.mji.rip/2025/07/16/94237772906ab6b432dad7cb2a98d766.png" width="500"/>


Long time no see, SVM.

<!--more-->

{% language_switch %}

{% lang_content en %}
# Objective Function

- The most primitive idea is naturally to consider the geometric margin
  
  $$
  \begin{array}{cc}{\max _{\vec{w}, b} \gamma} \\ {s . t . \quad \tilde{y}_{i}\left(\frac{\overrightarrow{\mathbf{w}}}{\|\overrightarrow{\mathbf{w}}\|_{2}} \cdot \overrightarrow{\mathbf{x}}_{i}+\frac{b}{\|\overrightarrow{\mathbf{w}}\|_{2}}\right) \geq \gamma, i=1,2, \cdots, N}\end{array}
  $$
- However, the geometric margin can be represented by the functional margin, and the functional margin can be scaled without affecting the choice of the classification hyperplane. Therefore, we set the functional margin to 1, then take the reciprocal to replace max with min, simplifying the objective function
  
  $$
  \begin{array}{c}{\min _{\vec{w}, b} \frac{1}{2}\|\overrightarrow{\mathbf{w}}\|_{2}^{2}} \\ {\text {s.t.} \quad \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}} \cdot \overrightarrow{\mathbf{x}}_{i}+b\right)-1 \geq 0, i=1,2, \cdots, N}\end{array}
  $$

# Min-Max

- After defining the problem as an optimization problem with inequality constraints, Lagrangian duality is used to transform the primal problem into a dual problem
- The description in statistical learning methods is as follows:
  - For the optimization problem:
    
    $$
    \begin{array}{ll}{\min _{x \in \mathbf{R}^{n}} f(x)} & {} \\ {\text { s.t. } \quad c_{i}(x) \leqslant 0, \quad i=1,2, \cdots, k} \\ {\qquad \begin{array}{ll}{h_{j}(x)=0,} & {j=1,2, \cdots, l}\end{array}}\end{array}
    $$
  - Introduce the generalized Lagrangian function
    
    $$
    L(x, \alpha, \beta)=f(x)+\sum_{i=1}^{k} \alpha_{i} c_{i}(x)+\sum_{j=1}^{I} \beta_{j} h_{j}(x)
    $$
  - Define
    
    $$
    \theta_{P}(x)=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
    $$
  - It can be determined that if the constraints are not satisfied, the Lagrangian multipliers can be set to a non-zero value to make the above expression infinite. If satisfied, the maximum value can only be achieved when $\alpha$ is 0, and $\beta$ does not play a role, therefore:
    
    $$
    \theta_{P}(x)=\left\{\begin{array}{l}{f(x)} \ x \text{satisfies original problem constraints} \\ {+\infty} \ \text{otherwise} \end{array}\right.
    $$
- Thus, the original minimization of f is transformed into minimizing a maximized $\theta$, namely
  
  $$
  \min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
  $$

# Duality

- Swapping the positions of max and min yields the dual problem, which is first optimizing for $x$, then optimizing for $\alpha,\beta$
  
  $$
  \begin{array}{l}{\max _{\alpha, \beta} \theta_{D}(\alpha, \beta)=\max _{\alpha, \beta} \min _{x} L(x, \alpha, \beta)} \\ {\text { s.t. } \quad \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, k}\end{array}
  $$
- The relationship between the dual problem and the primal problem:
  
  $$
  d^{*}=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} \min _{x} L(x, \alpha, \beta) \leqslant \min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)=p^{*}
  $$
- Proof: First look at the inner part of both sides. The left side is $\min \ L$, the right side is $\max \ L$. Although the variables are different, when the variables are fixed, there must be $\min _{x} L(x, \alpha, \beta) \leqslant L(x, \alpha, \beta) \leqslant \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)$. Now looking at the whole thing as a function and focusing on the outer part, comparing the max of $f_1$ and the min of $f_2$, from the above we know that $f_1$ is everywhere less than or equal to $f_2$. Therefore, even the maximum of $f_1$ must be less than or equal to the minimum of $f_2$, which is the so-called "chicken head is smaller than phoenix tail", with the premise that each chicken in the chicken flock is smaller than or equal to each phoenix in the phoenix flock.
- SVM satisfies the strong duality relationship, meaning the above equation reaches equality. Therefore, optimizing the original problem can be transformed into optimizing the dual problem, which facilitates the introduction of kernel functions.
- The original min-max problem can be solved directly, but not by directly taking partial derivatives of the min-max form, because taking partial derivatives with respect to $\alpha$ first is meaningless. We hope to obtain $w,b$ such that for any $\alpha \neq 0$, $\max L$ can be minimized.

# Solution

- After converting to the dual problem, first take derivatives of $w,b$ and set them to zero, then substitute the obtained $w,b$ back into the dual problem to get:
  
  $$
  \text{Substituting into the dual problem:} L(\overrightarrow{\mathbf{w}}, b, \vec{\alpha})=\frac{1}{2}\|\overrightarrow{\mathbf{w}}\|_{2}^{2}-\sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}} \cdot \overrightarrow{\mathbf{x}}_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
  $$
  
  $$
  \begin{aligned} L(\overrightarrow{\mathbf{w}}, b, \vec{\alpha})=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} & \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}\left[\left(\sum_{j=1}^{N} \alpha_{j} \tilde{y}_{j} \overrightarrow{\mathbf{x}}_{j}\right) \cdot \overrightarrow{\mathbf{x}}_{i}+b\right]+\sum_{i=1}^{N} \alpha_{i} \\ &=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \end{aligned} \\
  $$
- Here, the first term's coefficient is 1/2, the $b$ part in the second term is 0, and the remaining part is actually the same as the first term, just with a coefficient of 1. Therefore, subtracting gives -1/2. Finally, the outer max optimization objective is obtained
  
  $$
  \begin{aligned} \max _{\vec{\alpha}}-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\ s . t . \quad \sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}=0 \\ \alpha_{i} \geq 0, i=1,2, \cdots, N \end{aligned}
  $$
- The equality constraint condition is obtained by taking the derivative of $b$ to zero. As mentioned earlier, the objective function and constraints of SVM satisfy the strong duality relationship. The necessary and sufficient condition for strong duality is the KKT conditions, so the above equation should also satisfy the KKT conditions, namely:
  - Take partial derivatives of the Lagrangian function with respect to $w,b$ and set to zero
  - The constraint part of the Lagrangian function (each term in the summation) is zero, i.e., the complementary slackness condition
  - Original problem constraints
  - Lagrangian multipliers are non-negative
    
    $$
    \begin{aligned} \nabla_{\overrightarrow{\mathrm{w}}} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \overrightarrow{\mathrm{w}}^{*}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i} \overrightarrow{\mathrm{x}}_{i}=0 \\ \nabla_{b} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}=0 \\ \alpha_{i}^{*}\left[\tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1\right]=0, i=1,2, \cdots, N \\ \tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1 \geq 0, i=1,2, \cdots, N \\ \alpha_{i}^{*} \geq 0, i=1,2, \cdots, N \end{aligned}
    $$
- From the KKT conditions, we can express $w$ using $\alpha$ (which was also obtained when taking derivatives before). We know that $w$ represents the direction of the classification hyperplane, $b$ represents the bias, and is determined by support vectors. Therefore, those corresponding to non-zero $\alpha _j$
  
  $$
  \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}}^{*} \cdot \overrightarrow{\mathbf{x}}_{i}+b^{*}\right)-1
  $$
  
  determines $b$ (because when $\alpha$ is non-zero, by the complementary slackness condition, the latter term is zero, so $b$ can be solved). Finally, we get:
  
  $$
  b^{*}=\tilde{y}_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}\left(\overrightarrow{\mathrm{x}}_{i} \cdot \overrightarrow{\mathrm{x}}_{j}\right)
  $$
  
  It can be seen that $w,b$ are in the form of summation, but most are zero. Only the non-zero $\alpha$ terms (directly finding the non-zero $\alpha _j$ in the $b$ expression) play a role, meaning the support vector machine is determined only by a few support vectors.
- Therefore, given the data, first solve the maximum of the dual problem to obtain $\alpha$, then calculate $w,b$ from the non-zero parts of $\alpha$, to obtain the hyperplane.
- The definition and solving process of soft margin and non-linear support vector machines are similar, just with different constraint conditions and objective functions.
- The solving of $\alpha$ involves convex quadratic programming with many solution methods. One advantage of support vector machines is that the learned parameters only depend on support vectors, avoiding the curse of dimensionality during inference. However, during the learning process, all samples need to be calculated for optimization, so it is not friendly to large-scale data. Here, the SMO algorithm can be used for optimization.
{% endlang_content %}

{% lang_content zh %}
# 目标函数

- 最原始的想法自然是考虑几何间隔
  
  $$
  \begin{array}{cc}{\max _{\vec{w}, b} \gamma} \\ {s . t . \quad \tilde{y}_{i}\left(\frac{\overrightarrow{\mathbf{w}}}{\|\overrightarrow{\mathbf{w}}\|_{2}} \cdot \overrightarrow{\mathbf{x}}_{i}+\frac{b}{\|\overrightarrow{\mathbf{w}}\|_{2}}\right) \geq \gamma, i=1,2, \cdots, N}\end{array}
  $$
- 但是几何间隔可以用函数间隔表示，且函数间隔可以缩放而不影响分类超平面的选择，因此才有令函数间隔等于1，再取倒数把max换成min，化简了目标函数
  
  $$
  \begin{array}{c}{\min _{\vec{w}, b} \frac{1}{2}\|\overrightarrow{\mathbf{w}}\|_{2}^{2}} \\ {\text {s.t.} \quad \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}} \cdot \overrightarrow{\mathbf{x}}_{i}+b\right)-1 \geq 0, i=1,2, \cdots, N}\end{array}
  $$

# min-max

- 在将问题定义为带有不等式约束的最优化问题之后，就要用到拉格朗日对偶性来将原始问题转为对偶问题
- 统计学习方法的描述如下：
  - 对于最优化问题：
    
    $$
    \begin{array}{ll}{\min _{x \in \mathbf{R}^{n}} f(x)} & {} \\ {\text { s.t. } \quad c_{i}(x) \leqslant 0, \quad i=1,2, \cdots, k} \\ {\qquad \begin{array}{ll}{h_{j}(x)=0,} & {j=1,2, \cdots, l}\end{array}}\end{array}
    $$
  - 引入广义拉格朗日函数
    
    $$
    L(x, \alpha, \beta)=f(x)+\sum_{i=1}^{k} \alpha_{i} c_{i}(x)+\sum_{j=1}^{I} \beta_{j} h_{j}(x)
    $$
  - 定义
    
    $$
    \theta_{P}(x)=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
    $$
  - 可以判断，假如不满足约束条件的话，可以令拉格朗日乘子不为0从而使上式无穷大，满足情况的话，上式要最大只能让$\alpha$为0，$\beta$则不起作用，因此：
    
    $$
    \theta_{P}(x)=\left\{\begin{array}{l}{f(x)} \ x满足原始问题约束 \\ {+\infty} \ 其他 \end{array}\right.
    $$
- 这样原始的最小化一个f就转换为最小化一个最大化的$\theta$，即
  
  $$
  \min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)
  $$

# 对偶

- max和min对换位置就得到对偶问题，即先针对$x$优化，再针对$\alpha,\beta$优化
  
  $$
  \begin{array}{l}{\max _{\alpha, \beta} \theta_{D}(\alpha, \beta)=\max _{\alpha, \beta} \min _{x} L(x, \alpha, \beta)} \\ {\text { s.t. } \quad \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, k}\end{array}
  $$
- 对偶问题和原始问题的关系：
  
  $$
  d^{*}=\max _{\alpha, \beta: \alpha_{i} \geqslant 0} \min _{x} L(x, \alpha, \beta) \leqslant \min _{x} \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)=p^{*}
  $$
- 证明：先看两边的里面一部分，左边是$\min \ L$，右边是$\max \ L$，尽管自变量不一样，但是当自变量固定时，必然有$\min _{x} L(x, \alpha, \beta) \leqslant L(x, \alpha, \beta) \leqslant \max _{\alpha, \beta: \alpha_{i} \geqslant 0} L(x, \alpha, \beta)$，现在把里面整体看成一个函数，只看外面，即比较左边的$\max f_1$和$\min f_2$，由上可知$f_1$处处小于等于$f_2$，那即便是$f_1$的最大值，也一定小于等于$f_2$的最小值，也就是所谓的“鸡头小于凤尾”，前提是鸡群里的每一只鸡小于等于凤群里的每一只凤。
- SVM满足强对偶关系，即上式取到等号，因此优化原问题可以转化为优化对偶问题，方便引入核函数。
- 原始的min-max问题可以直接解，但不是对min-max形式直接求偏导，因为先对$\alpha$求偏导没有意义，我们是希望得到$w,b$使得对任意的$\alpha \neq 0$，$\max L$都可以取到最小值。

# 求解

- 转换为对偶问题之后，先对$w,b$求导令其为0，之后将得到的$w,b$回代入对偶问题得到：
  
  $$
  带入对偶问题：L(\overrightarrow{\mathbf{w}}, b, \vec{\alpha})=\frac{1}{2}\|\overrightarrow{\mathbf{w}}\|_{2}^{2}-\sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}} \cdot \overrightarrow{\mathbf{x}}_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
  $$
  
  $$
  \begin{aligned} L(\overrightarrow{\mathbf{w}}, b, \vec{\alpha})=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} & \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}\left[\left(\sum_{j=1}^{N} \alpha_{j} \tilde{y}_{j} \overrightarrow{\mathbf{x}}_{j}\right) \cdot \overrightarrow{\mathbf{x}}_{i}+b\right]+\sum_{i=1}^{N} \alpha_{i} \\ &=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \end{aligned} \\
  $$
- 这里第一项系数为1/2，第二项中b的部分为0，剩余部分其实和第一项相同，只不过系数为1，因此相减得到-1/2。最后得到外层的max优化目标
  
  $$
  \begin{aligned} \max _{\vec{\alpha}}-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} \tilde{y}_{i} \tilde{y}_{j}\left(\overrightarrow{\mathbf{x}}_{i} \cdot \overrightarrow{\mathbf{x}}_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\ s . t . \quad \sum_{i=1}^{N} \alpha_{i} \tilde{y}_{i}=0 \\ \alpha_{i} \geq 0, i=1,2, \cdots, N \end{aligned}
  $$
- 其中等式约束条件是对$b$求导为0得到的。之前说到SVM的目标函数和约束条件满足强对偶关系，强对偶关系的充要条件是KKT条件，因此上式应该也满足KKT条件，即
  - 拉格朗日函数对$w,b$求偏导为0
  - 拉格朗日函数中原问题的约束部分（求和的每一项）为0，即松弛互补条件
  - 原始问题约束
  - 拉格朗日乘子非负
    
    $$
    \begin{aligned} \nabla_{\overrightarrow{\mathrm{w}}} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \overrightarrow{\mathrm{w}}^{*}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i} \overrightarrow{\mathrm{x}}_{i}=0 \\ \nabla_{b} L\left(\overrightarrow{\mathrm{w}}^{*}, b^{*}, \vec{\alpha}^{*}\right)=& \sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}=0 \\ \alpha_{i}^{*}\left[\tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1\right]=0, i=1,2, \cdots, N \\ \tilde{y}_{i}\left(\overrightarrow{\mathrm{w}}^{*} \cdot \overrightarrow{\mathrm{x}}_{i}+b^{*}\right)-1 \geq 0, i=1,2, \cdots, N \\ \alpha_{i}^{*} \geq 0, i=1,2, \cdots, N \end{aligned}
    $$
- 由KKT条件，我们可以用$\alpha$表示$w$（之前求导时也已经得到过），我们知道$w$代表分类超平面的方向，$b$代表偏置，由支持向量决定，因此那些$\alpha _j$不为0对应的
  
  $$
  \tilde{y}_{i}\left(\overrightarrow{\mathbf{w}}^{*} \cdot \overrightarrow{\mathbf{x}}_{i}+b^{*}\right)-1
  $$
  
  决定了$b$（因为$\alpha$不为0，由松弛互补条件，则后一项为0，就可以求得$b$）。最后得到：
  
  $$
  b^{*}=\tilde{y}_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} \tilde{y}_{i}\left(\overrightarrow{\mathrm{x}}_{i} \cdot \overrightarrow{\mathrm{x}}_{j}\right)
  $$
  
  可以看到，$w,b$都是求和的形式，但大部分为0，只有不为0的$\alpha$（在$b$的表达式里直接找出不为0的记为$\alpha _j$）项才起作用，即支持向量机只有少数的支持向量决定。
- 因此，给定数据，先求得对偶问题极大值，得到$\alpha$，再由$\alpha$中不为0的部分计算出$w,b$，得到超平面。
- 软间隔和非线性支持向量机的定义和求解过程类似，只不过约束条件和目标函数不同。
- $\alpha$的求解涉及凸二次规划，有很多解法。支持向量机的一个优点就是学习到的参数只依赖支持向量，推理时避免了维度灾难，但是在学习的过程中，需要对所有样本计算最优化，因此对于大规模数据不友好，这里可以用SMO算法来优化


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
