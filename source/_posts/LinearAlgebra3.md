---
title: MIT线性代数笔记3
date: 2017-01-22 19:21:02
tags: [linearalgebra,math]
categories: 数学
mathjax: true
html: true
---

***
# <font size=5 >第十七讲：行列式及其性质</font>

## <font size=4 >行列式</font>
-	矩阵A的行列式是与矩阵相关的一个数，记作$detA或者|A|$
-	行列式的性质
 -	$detI=1$
 -	交换行，行列式的值的符号会相反
 -	一个置换矩阵的行列式是1或-1，取决于交换行次数的奇偶
 -	两行相等使得行列式为0(由性质二可以直接推出)
 -	矩阵消元不改变其行列式(证明见下)
 -	某一行为0，行列式为0(与0相乘等价于某一行为0，结果为0)
 -	$detA=0$当且仅当A是奇异矩阵
 -	$det(A+B) \neq detA+detB \\ detAB=(detA)(detB)$
 -	$detA^{-1}detA=1$
 -	$detA^2=(detA)^2$
 -	$det2A=2^n detA$
 -	$detA^T=detA$(证明见下)
<!--more-->
-	行列式按行是线性的，但行列式本身不是线性的
	$$
	\begin{vmatrix}
	1 & 0 \\
	0 & 1 \\
	\end{vmatrix}=1 \\
	\begin{vmatrix}
	0 & 1 \\
	1 & 0 \\
	\end{vmatrix}=-1 \\
	\begin{vmatrix}
	ta & tb \\
	c & d \\
	\end{vmatrix}=
	t\begin{vmatrix}
	a & b \\
	c & d \\
	\end{vmatrix} \\
	\begin{vmatrix}
	t+a & t+b \\
	c & d \\
	\end{vmatrix}=
	\begin{vmatrix}
	a & b \\
	c & d \\
	\end{vmatrix}+
	\begin{vmatrix}
	t & t \\
	c & d \\
	\end{vmatrix}
	$$
-	证明消元不改变行列式
	$$
	\begin{vmatrix}
	a & b \\
	c-la & d-lb \\
	\end{vmatrix}=
	\begin{vmatrix}
	a & b \\
	c & d \\
	\end{vmatrix}-l
	\begin{vmatrix}
	a & b \\
	a & b \\
	\end{vmatrix}=
	\begin{vmatrix}
	a & b \\
	c & d \\
	\end{vmatrix}
	$$
-	证明转置不改变行列式
	$$
	A=LU \\
	即证 |U^TL^T|=|LU| \\
	|U^T||L^T|=|L||U| \\
	以上四个矩阵都是三角矩阵，行列式等于对角线乘积，转置没有影响，所以相等 \\
	$$
	
	
## <font size=4 >三角阵行列式</font>
-	对三角阵U的行列式,值为对角线上元素乘积(主元乘积)
-	为什么三角阵其他元素不起作用？因为通过消元我们可以得到只有对角元素的矩阵，而消元不改变行列式
-	为什么是对角线元素的乘积？因为可以消元后可以依次把对角元素提出来，即得到$d_1d_2d_3...d_nI$，其中单位矩阵的行列式为1
-	奇异矩阵行列式为0，存在全0行；可逆矩阵行列式不为0，能化成三角阵，行列式是三角矩阵对角元素乘积

## <font size=4 >A little more</font>
-	进行奇数次置换和偶数次置换得到的行列式肯定不一样(符号不同)，这意味着进行奇数次置换和偶数次置换后的矩阵不会一样，即置换是严格区分奇偶的
	
# <font size=5 >第十八讲：行列式公式和代数余子式</font>

## <font size=4 >行列式公式</font>
-	推导2*2行列式
	$$
	\begin{vmatrix}
	a & b \\
	c & d \\
	\end{vmatrix}=
	\begin{vmatrix}
	a & 0 \\
	c & d \\
	\end{vmatrix}+
	\begin{vmatrix}
	0 & b \\
	c & d \\
	\end{vmatrix}=
	\begin{vmatrix}
	a & 0 \\
	c & 0 \\
	\end{vmatrix}+
	\begin{vmatrix}
	a & 0 \\
	0 & d \\
	\end{vmatrix}+
	\begin{vmatrix}
	0 & b \\
	c & 0 \\
	\end{vmatrix}+
	\begin{vmatrix}
	0 & b \\
	0 & d \\
	\end{vmatrix} \\
	=0+ad-bc+0
	$$
	我们可以发现这种方法是一次取一行，将这一行拆解(行列式按行是线性的)，再提取出因子，通过行交换得到单位矩阵，通过性质一和性质二得到答案
-	如果扩展到3*3矩阵，则第一行分解成三部分，每部分针对第二行又分解成三部分，所以最后得到27部分，其中不为0的部分是那些各行各列均有元素的矩阵。
-	例如
	$$
	\begin{vmatrix}
	a & 0 & 0\\
	0 & 0 & b\\
	0 & c & 0\\
	\end{vmatrix}
	$$
	先提取出因子，得到$abc$，交换第二行第三行得到单位矩阵，于是答案就是$abc*detI=abc$，又因为进行了一次行交换，所以答案是负的，$-abc$
-	n*n的矩阵可以分成$n!$个部分，因为第一行分成n个部分，第二行不能重复，选择n-1行，一次重复，所以得到$n!$部分
-	行列式公式就是这$n!$个部分加起来


## <font size=4 >代数余子式</font>
-	$det=a_{11}(a_{22}a_{33}-a_{23}{32})+a_{12}(....)+a_{13}(....)$
-	提取出一个因子，由剩余的因子即括号内的内容组成的就是余子式
-	从矩阵上看，选择一个元素，它的代数余子式就是排除这个元素所在行和列剩下的矩阵的行列式
-	$a_{ij}$的代数余子式记作$c_{ij}$
-	注意代数余子式的正负，与$i+j$的奇偶性有关，偶数取正，奇数取负，这里的符号是指代数余子式对应的子矩阵正常计算出行列式后前面的符号
-	$detA=a_{11}C_{11}+a_{12}C_{12}+....+a_{1n}C_{1n}$	

# <font size=5 >第十九讲：克拉默法则，逆矩阵，体积</font>

## <font size=4 >逆矩阵</font>
-	只有行列式不为0时，矩阵才是可逆的
-	逆矩阵公式
	$$
	A^{-1}=\frac{1}{detA}C^T
	$$
	其中$C_{ij}$是$A_{ij}$的代数余子式
-	证明：即证$AC^T=(detA)I$
	$$
	\begin{bmatrix}
	a_{11} & ... & a_{1n} \\
	a_{n1} & ... & a_{nn} \\
	\end{bmatrix}
	\begin{bmatrix}
	c_{11} & ... & c_{n1} \\
	c_{1n} & ... & c_{nn} \\
	\end{bmatrix}=
	\begin{bmatrix}
	detA & 0 & 0 \\
	0 & detA & 0 \\
	0 & 0 & detA \\
	\end{bmatrix}
	$$
	对角线上都是行列式，因为$det=a_{11}(a_{22}a_{33}-a_{23}{32})+a_{12}(....)+a_{13}(....)$
	其他位置都是0，因为行a乘以行b的代数余子式相当于求一个矩阵的行列式，这个矩阵行a与行b相等，行列式为0
	
## <font size=4 >克拉默法则</font>
-	解Ax=b
	$$
	Ax=b \\
	x=A^{-1}b \\
	x=\frac{1}{detA}C^Tb \\
	 \\
	x_1=\frac{detB_1}{detA} \\
	x_3=\frac{detB_2}{detA} \\
	... \\
	$$
-	克拉默法则即发现矩阵$B_i$就是矩阵$A$的第i列换成b，其余不变

## <font size=4 >体积</font>
-	A的行列式可以代表一个体积，例如3*3矩阵的行列式代表一个三维空间内的体积
-	矩阵的每一行代表一个盒子的一条边(从同一顶点连出的)，行列式就是这个盒子的体积，行列式的正负代表左手或者右手系。
-	(1)单位矩阵对应单位立方体，体积为1
-	对正交矩阵Q,
	$$
	QQ^T=I \\
	|QQ^T|=|I| \\
	|Q||Q^T|=1 \\
	{|Q|}^2=1 \\
	|Q|=1 \\
	$$
	Q对应的盒子是单位矩阵对应的单位立方体在空间中旋转过一个角度
-	(3a)如果矩阵的某一行翻倍，即盒子一组边翻倍，体积也翻倍，从行列式角度可以把倍数提出来，因此行列式也是翻倍
-	(2)交换矩阵两行，盒子的体积不变
-	(3b)矩阵某一行拆分，盒子也相应切分为两部分
-	以上，行列式的三条性质(1,2,3a,3b)均可以在体积上验证
	
# <font size=5 >第二十讲：特征值和特征向量</font>

## <font size=4 >特征向量</font>
-	给定矩阵A，矩阵A可以看成一个函数，作用在一个向量x上，得到向量Ax
-	当Ax平行于x时，即$Ax=\lambda x$，我们称$x$为特征向量，$\lambda$为特征值
-	如果A是奇异矩阵，$\lambda = 0$是一个特征值

## <font size=4 >几个例子</font>
-	如果A是投影矩阵，可以发现它的特征向量就是投影平面上的任意向量，因为$Ax$即投影到平面上，平面上的所有向量投影后不变，自然平行，同时特征值就是1。如果向量垂直于平面，$Ax=0$，特征值为0.因此投影矩阵A的特征向量就分以上两种情况，特征值为1或0.
-	再举一例
	$$
	A=
	\begin{bmatrix}
	0 & 1 \\
	1 & 0 \\
	\end{bmatrix} \\
	对\lambda =1, x=
	\begin{bmatrix}
	1 \\
	1 \\
	\end{bmatrix}
	Ax=
	\begin{bmatrix}
	1 \\
	1 \\
	\end{bmatrix} \\
	对\lambda =-1, x=
	\begin{bmatrix}
	-1 \\
	1 \\
	\end{bmatrix}
	Ax=
	\begin{bmatrix}
	1 \\
	-1 \\
	\end{bmatrix} \\	
	$$
-	n*n矩阵有n个特征值
-	特征值的和等于对角线元素和，这个和称为迹(trace)，
-	如何求解$Ax=\lambda x$
	$$
	(A-\lambda I)x=0 \\
	可见方程有非零解，(A-\lambda I)必须是奇异的 \\
	即: det(A-\lambda I)=0 \\
	$$
-	$$
	If \qquad Ax=\lambda x \\
	Then \qquad (A+3I)x=(\lambda +3)x \\
	因为加上单位矩阵，特征向量不变依然为x，特征值加上单位矩阵的系数即(\lambda +3) \\
	$$
-	A+B的特征值不一定是A的特征值加上B的特征值，因为他们的特征向量不一定相同。同理AB的特征值也不一定是他们的特征值的乘积
-	再举一例，对旋转矩阵Q
	$$
	Q=
	\begin{bmatrix}
	0 & -1 \\
	1 & 0 \\
	\end{bmatrix} \\
	trace=0=\lambda _1 +\lambda _2 \\
	det=1=\lambda _1 \lambda _2 \\
	但是可以看出 \lambda _1，\lambda _2 无实数解 \\
	$$
-	再看看更加糟糕的情况(矩阵更加不对称，更难得到实数解的特征值)
	$$
	A=
	\begin{bmatrix}
	3 & 1 \\
	0 & 3 \\
	\end{bmatrix} \\
	det(A-\lambda I)=
	\begin{vmatrix}
	3-\lambda & 1 \\
	0 & 3-\lambda \\
	\end{vmatrix}
	==(3-\lambda )^2=0 \\
	\lambda _1=\lambda _2=3 \\
	x_1=
	\begin{bmatrix}
	1 \\
	0 \\
	\end{bmatrix}
	$$
	
	# <font size=5 >第二十一讲：对角化和A的幂</font>

## <font size=4 >对角化</font>
-	假设A有n个线性无关特征向量，按列组成矩阵S，即特征向量矩阵
-	以下所有关于矩阵对角化的讨论都在S可逆，即n个特征向量线性无关的前提下
-	$$
	AS=A[x_1,x_2...x_n]=[\lambda _1 x_1,....\lambda _n x_n] \\
	=[x_1,x_2,...x_n]
	\begin{bmatrix}
	\lambda _1 & 0 & ... & 0 \\
	0 & \lambda _2 & ... & 0 \\
	... & ... & ... & ... \\
	0 & 0  & 0 & \lambda _n \\
	\end{bmatrix} \\
	=S \Lambda \\
	$$


-	假设S可逆，即n个特征向量无关，此时可以得到
	$$
	S^{-1}AS=\Lambda \\
	A=S\Lambda S^{-1} \\
	$$
-	$\Lambda$是对角矩阵，这里我们得到了除了$A=LU$和$A=QR$之外的一种矩阵分解
-	$$
	if \qquad Ax=\lambda x \\
	A^2 x=\lambda AX=\lambda ^2 x \\
	A^2=S\Lambda S^{-1} S \Lambda S^{-1}=S \Lambda ^2 S^{-1} \\
	$$
-	上面关于$A^2$的两式说明平方后特征向量不变，特征值平方，K次方同理
-	特征值和特征向量帮助我们理解矩阵幂，当计算矩阵幂时，我们可以把矩阵分解成特征向量矩阵和对角阵相乘的形式，K个相乘两两可以抵消，如上式
-	什么样的矩阵的幂趋向于0(稳定)
	$$
	A^K \rightarrow 0 \quad as \quad K \rightarrow \infty \\
	if \quad all |\lambda _i|<1 \\ 
	$$
-	哪些矩阵可以对角化？
	如果所有特征值不同，则A可以对角化
-	如果矩阵A已经是对角阵，则$\Lambda$与A相同
-	特征值重复的次数称为代数重度，对三角阵，如
	$$
	A=
	\begin{bmatrix}
	2 & 1 \\
	0 & 2 \\
	\end{bmatrix} \\
	det(A-\lambda I)=
	\begin{vmatrix}
	2-\lambda & 1 \\
	0 & 2-\lambda \\
	\end{vmatrix}=0 \\
	\lambda =2 \\
	A-\lambda I=
	\begin{bmatrix}
	0 & 1 \\
	0 & 0 \\
	\end{bmatrix} \\
	$$
	对$A-\lambda I$，几何重数是1，而特征值的代数重度是2
	特征向量只有(1,0)，因此对于三角阵，它不可以对角化，不存在两个线性无关的特征向量。
## <font size=4 >A的幂</font>
-	多数矩阵拥有互相线性无关的一组特征值，可以对角化。假如可以对角化，我们需要关注如何求解A的幂
-	$$
	give \quad u_0 \\
	u_{k+1}=Au_k \\
	u_k=A^ku_0 \\
	how \quad to \quad solve \quad u_k \\
	u_0=c_1x_1+c_2x_2+...+c_nx_n=SC \\
	Au_0=c_1 \lambda _1 x_1 + c_2 \lambda _2 x_2 +...+c_n \lambda _n x_n \\
	A^{100}u_0=c_1 \lambda _1^{100} x_1 + c_2 \lambda _2^{100} x_2 +...+c_n \lambda _n^{100} x_n \\
	=S\Lambda ^{100} C \\
	=u_{100} \\
	$$
	因为n个特征向量互相不线性相关，因此它们可以作为一组基覆盖整个n维空间，自然$u_0$可以用特征向量的线性组合表示，C是线性系数向量。上式得出了矩阵幂的解法，接下来以斐波那契数列为例
	$$
	F_0=0 \\
	F_1=1 \\
	F_2=1 \\
	F_3=2 \\
	F_4=3 \\
	F_5=5 \\
	..... \\
	F_{100}=? \\
	$$
	斐波那契数列的增长速度有多快?由特征值决定，我们尝试构造向量，来找到斐波那契数列迭代的矩阵关系
	$$
	F_{k+2}=F_{k+1}+F_k \\
	F_{k+1}=F_{k+1} \\
	定义向量u_k=
	\begin{bmatrix}
	F_{k+1} \\
	F_k \\
	\end{bmatrix} \\
	利用这个向量可以将前两个等式写成矩阵形式 \\
	u_{k+1}=
	\begin{bmatrix}
	1 & 1 \\
	1 & 0 \\
	\end{bmatrix}
	u_k \\
	A=
	\begin{bmatrix}
	1 & 1 \\
	1 & 0 \\
	\end{bmatrix} \\
	\lambda =\frac {1 \pm \sqrt 5}2 \\
	$$
	得到两个特征值，我们很容易得到特征向量
	回到斐波那契数列，斐波那契数列的增长速率由我们构造的"数列更新矩阵"的特征值决定，而且由$A^{100}u_0=c_1 \lambda _1^100 x_1 + c_2 \lambda _2^100 x_2 +...+c_n \lambda _n^100 x_n$可以看出增长率主要由由较大的特征值决定，因此$F_{100}$可以写成如下形式
	$$
	F_{100} \approx c_1 {\frac {1 + \sqrt 5}2}^{100} \\
	$$
	再有初始值有
	$$
	u_0=
	\begin{bmatrix}
	F_1 \\
	F_0 \\
	\end{bmatrix}=
	\begin{bmatrix}
	1 \\
	0 \\
	\end{bmatrix}
	=c_1x_1+c_2x_2
	$$
	其中$x_1,x_2$是两个特征向量，线性系数可求，代入公式可求$F_{100}$的近似值

## <font size=4 >总结</font>
-	我们发现在A可逆的情况下，A可以分解成$S\Lambda S^{-1}$的形式
-	这种形式有一个特点，方便求A的幂，即分解后可以看出A的幂的特征值单位矩阵是A的特征值单位矩阵的幂
-	我们在求解斐波那契数列中尝试运用此特点，首先将数列的更新转换为矩阵形式
-	求出矩阵的特征值，特征向量
-	由A的幂的展开式可以看出A的幂主要由较大的特征值决定，因此$F_{100}$可以写成$F_{100} \approx c_1 {(\frac {1 + \sqrt 5}2)}^{100}$的形式
-	由初始值$F_0$求出线性系数，代入上式，得到$F_{100}$的近似值
-	以上是差分方程的一个例子，下一节将讨论微分方程

# <font size=5 >第二十二讲：微分方程和exp(At)</font>

## <font size=4 >微分方程</font>
-	常系数线性方程的解是指数形式的，如果微分方程的解是指数形式，只需利用线代求出指数，系数，就可以求出解
-	举个例子
	$$
	\frac{du_1}{dt}=-u_1+2u_2 \\
	\frac{du_2}{dt}=u_1-2u_2 \\
	u(0)=
	\begin{bmatrix}
	1 \\
	0 \\
	\end{bmatrix} \\
	$$
	首先我们列出系数矩阵，并找出矩阵的特征值和特征向量
	$$
	A=
	\begin{bmatrix}
	-1 & 2 \\
	1 & -2 \\
	\end{bmatrix}
	$$
	易得$\lambda=0$是这个奇异矩阵的一个解，由迹可以看出第二个特征值是$\lambda=-3$，并得到两个特征向量
	$$
	x_1=
	\begin{bmatrix}
	2 \\
	1 \\
	\end{bmatrix} \\
	x_2=
	\begin{bmatrix}
	1 \\
	-1 \\
	\end{bmatrix}
	$$
	微分方程解的通解形式将是
	$$
	u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2
	$$
	为什么？
	$$
	\frac{du}{dt} \\
	=c_1 \lambda _1 e^{\lambda _1 t}x_1 \\
	=A c_1 e^{\lambda _1 t}x_1 \\
	because \quad A x_1=\lambda _1 x_1 \\
	$$
-	在差分方程$u_{k+1}=Au_k$当中，解的形式是$c_1\lambda _1 ^k x_1+c_2 \lambda _2 ^k x_2$
	在微分方程$\frac {du}{dt}=Au$当中，解的形式是$u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2$
	$c_1,c_2$由初始值解出，即系数矩阵C乘特征向量矩阵S得到初始值
	可以看出t趋于无穷时，例子方程的解只剩下稳态部分，即$(\frac 23,\frac 13)
-	什么时候解趋向于0？存在负数特征值，因为$e^{\lambda t}$需要趋向于0
	如果特征值是复数呢？虚数部分的模值是1，所以如果复数的实数部分是负数，解依然趋向于0
-	什么时候存在稳态？特征值中只存在0和负数，就如上面的例子
-	什么时候解无法收敛？任何特征值的实数部分大于0
-	改变系数矩阵的符号，特征值也改变符号，稳态的解依然稳态，收敛的解就会变成发散
-	如何从矩阵直接判断解是否收敛？即特征值的实数部分都小于0？
	矩阵的迹应该小于0，但对角线之和为0依然不一定收敛，如
	$$
	\begin{bmatrix}
	-2 & 0 \\
	0 & 1 \\
	\end{bmatrix}
	$$
	因此还需要另一个条件：行列式的值是特征值乘积，因此行列式的值应该大于0
	
## <font size=4 >exp(At)</font>	
-	是否可以把解表示成$S,\Lambda$的形式
-	矩阵A表示$u_1,u_2$耦合，首先我们需要将u对角化，解耦
-	$$
	\frac{du}{dt} = Au \\
	set \quad u=Sv \\
	S \frac{dv}{dt} = ASv \\
	\frac{dv}{dt}=S^{-1}ASv=\Lambda v \\
	v(t)=e^{\Lambda t}v(0) \\
	u(t)=Se^{\Lambda t}S^{-1}u(0) \\
	$$
	
	# <font size=5 >第二十一讲：马尔科夫矩阵;傅立叶级数</font>

## <font size=4 >马尔科夫矩阵</font>
-	一个典型的马尔科夫矩阵
	$$
	\begin{bmatrix}
	0.1 & 0.01 & 0.3 \\
	0.2 & 0.99 & 0.3 \\
	0.7 & 0 & 0.4 \\
	\end{bmatrix}
	$$
-	每一个元素大于等于0，每一列之和为1，马尔科夫矩阵的幂都是马尔科夫矩阵
-	$\lambda=1$是一个特征值，其余的特征值的绝对值都小于1


-	在上一讲中我们谈到矩阵的幂可以分解为
	$$
	u_k=A^ku_0=c_1\lambda _1 ^kx_1+c_2\lambda _2 ^kx_2+.....
	$$
	当A是马尔科夫矩阵时，只有一个特征值为1，其余特征值小于1，随着k的变大，小于1的特征值所在项趋向于0，只保留特征值为1的那一项，同时对应的特征向量的元素都大于0
-	当每一列和为1时，必然存在一个特征值$\lambda =1$
	证明：
	$$
	A-I=
	\begin{bmatrix}
	-0.9 & 0.01 & 0.3 \\
	0.2 & -0.01 & 0.3 \\
	0.7 & 0 & -0.6 \\
	\end{bmatrix}
	$$
	若1是一个特征值，则$A-I$应该是奇异的，可以看到$A-I$每一列和为0，即说明行向量线性相关，即矩阵奇异,同时全1向量在左零空间。
-	对于马尔科夫矩阵A，我们研究$u_{k+1}=Au_k$
	一个例子，u是麻省和加州的人数，A是人口流动矩阵
	$$
	\begin{bmatrix}
	u_{cal} \\
	u_{mass} \\
	\end{bmatrix}_{t=k+1}
	=
	\begin{bmatrix}
	0.9 & 0.2 \\
	0.1 & 0.8 \\
	\end{bmatrix}
	\begin{bmatrix}
	u_{cal} \\
	u_{mass} \\
	\end{bmatrix}_{t=k}
	$$
	可以看到每一年(k)80%的人留在麻省，20%的人前往加州，加州那边也有10%移居麻省
	对马尔科夫矩阵A
	$$
	\begin{bmatrix}
	0.9 & 0.2 \\
	0.1 & 0.8 \\
	\end{bmatrix} \\
	\lambda _1 =1 \\
	\lambda _2 =0.7 \\
	$$
	对特征值为1的项，容易求出特征向量为$(2,1)$，对特征值为0.7的项，特征向量为(-1,1)
	得到我们要研究的公式
	$$
	u_k=c_1\*1^k\*
	\begin{bmatrix}
	2 \\
	1 \\
	\end{bmatrix}
	+c_2\*(0.7)^k\*
	\begin{bmatrix}
	-1 \\
	1 \\
	\end{bmatrix}
	$$
	假设一开始加州有0人，麻省有1000人，即$u_0$，代入公式可以得到$c_1,c_2$，可以看到很多年之后，加州和麻省的人数将稳定，各占1000人中的三分之一和三分之二。
-	行向量为和为1是另外一种定义马尔科夫矩阵的方式

## <font size=4 >傅里叶级数</font>
-	先讨论带有标准正交基的投影问题
-	假设$q_1....q_n$是一组标准正交基，任何向量$v$都是这组基的线性组合
-	现在我们要求出线性组合系数$x_1....x_n$
	$v=x_1q_1+x_2q_2+...x_nq_n$
	一种方法是将$v$与$q_i$做内积，逐一求出系数
	$$
	q_1^Tv=x_1q_1^Tq_1+0+0+0....+0=x_1 \\
	$$
	写成矩阵形式
	$$
	\begin{bmatrix}
	q_1 & q_2 & ... & q_n \\
	\end{bmatrix}
	\begin{bmatrix}
	x_1 \\
	x_2 \\
	... \\
	x_n \\
	\end{bmatrix}=
	v \\
	Qx=v \\
	x=Q^{-1}v=Q^Tv \\
	$$
-	现在讨论傅里叶级数
	我们希望将函数分解
	$$
	f(x)=a_0+a_1cosx+b_1sinx+a_2cos2x+b_2cos2x+.......
	$$
	关键是，在这种分解中，$coskx,sinkx$构成一组函数空间的无穷正交基，即这些函数内积为0(向量的内积是离散的值累加，函数的内积是连续的值积分)。
-	如何求出傅里叶系数？
	利用之前的向量例子来求
	将$f(x)$逐一与正交基元素内积，得到这个正交基元素对应的系数乘$\pi$，例如
	$$
	\int _0 ^{2\pi} f(x)cosx dx=0+ a_1 \int _0^{2\pi}(cosx)^2dx+0+0...+0=\pi a_1 \\
	$$

# <font size=5 >第二十二讲：对称矩阵及其正定性</font>

## <font size=4 >对称矩阵</font>
-	对称矩阵的特征值是实数，不重复的特征值对应的特征向量互相正交
-	对一般矩阵$A=S\Lambda S^{-1}$，S为特征向量矩阵
	对对称矩阵$A=Q\Lambda Q^{-1}=Q\Lambda Q^T$，Q为标准正交的特征向量矩阵
-	为什么特征值都是实数？
	$Ax=\lambda x$
	对左右同时取共轭，因为我们现在只考虑实数矩阵A
	$Ax^{\*}=\lambda ^{\*} x^{\*}$
	即$\lambda$和它的共轭都是特征值，现在再对等式两边取转置
	$x^{\* T}A^T=x^{\* T} \lambda ^{\* T} $
	上式中$A=A^T$，且两边同乘以$x$，与$x^{\* T}A\lambda x^{\* T}x$对比可得
	$\lambda ^{\*}=\lambda$，即特征值是实数
-	可见，对于复数矩阵，需要$A=A^{\* T}$才满足对称
-	对于对称矩阵
	$$
	A=Q\Lambda Q^{-1}=Q\Lambda Q^T \\
	=\lambda _1 q_1 q_1^T+\lambda _2 q_2 q_2^T+.... \\
	$$
	所以每一个对称矩阵都是一些互相垂直的投影矩阵的组合
-	对于对称矩阵，正主元的个数等于正特征值的个数，且主元的乘积等于特征值的乘积等于矩阵的行列式

## <font size=4 >正定性</font>
-	正定矩阵都是对称矩阵，是对称矩阵的一个子类，其所有特征值为正数，所有主元为正数，所有的子行列式都是正数
-	特征值的符号与稳定性有关
-	主元、行列式、特征值三位一体，线性代数将其统一





