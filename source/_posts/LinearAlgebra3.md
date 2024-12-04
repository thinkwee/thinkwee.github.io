---
title: Note for Linear Algebra 3
date: 2017-01-22 19:21:02
tags: [linearalgebra,math]
categories: Math
mathjax: true
html: true
---

***


Lecture 17: Determinants and Their Properties
=============================================

Determinant
-----------

*   The determinant of matrix A is a number associated with the matrix, denoted as $detA或者|A|$
    
*   Properties of determinants
    
    *   $detI=1$
        
    *   The sign of the determinant value will be reversed when rows are exchanged
        
    *   The determinant of a permutation matrix is 1 or -1, depending on the parity of the number of rows exchanged
        
    *   Two rows being equal makes the determinant equal to 0 (which can be directly deduced from property two)
        
    *   Matrix elimination does not change its determinant (proof is below)
        
    *   A certain row is 0, the determinant is 0 (multiplying by 0 is equivalent to a certain row being 0, resulting in 0)
        
    *   When and only when A is a singular matrix
        
    *   $det(A+B) \neq detA+detB \\ detAB=(detA)(detB)$
        
    *   $detA^{-1}detA=1$
        
    *   $detA^2=(detA)^2$
        
    *   $det2A=2^n detA$
        
    *   $detA^T=detA$ (Proof see below)
    
<!--more-->

{% language_switch %}

{% lang_content en %}

*   The determinant is linear by row, but the determinant itself is not linear
    
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
    
*   Proof that elimination does not change the determinant
    
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
    
*   Proof that the transpose does not change the determinant
    
    $$
    A=LU \\
    $$
    
*   Translation: $|U^TL^T|=|LU|$ 
    $$
    |U^T||L^T|=|L||U|
    $$
    
*   The above four matrices are all triangular matrices, the determinant equals the product of the diagonal elements, the transpose has no effect, so they are equal
    

Triangular matrix determinant
-----------------------------

*   The determinant of the triangular matrix U is the product of the elements on the diagonal (the pivot product)
*   Why do the other elements of the triangular matrix not work? Because by elimination we can obtain a matrix with only diagonal elements, and elimination does not change the determinant
*   Why is it the product of the diagonal elements? Because after elimination, the diagonal elements can be successively extracted, yielding $d_1d_2d_3...d_nI$ , where the determinant of the unit matrix is 1
*   The determinant of a singular matrix is 0, and it has rows of all zeros; the determinant of an invertible matrix is not 0, and it can be reduced to a triangular matrix, with the determinant being the product of the diagonal elements of the triangular matrix

A little more
-------------

*   The determinant obtained from odd-numbered permutations and even-numbered permutations is definitely different (signs differ), which means the matrices after odd-numbered and even-numbered permutations will not be the same, i.e., permutations strictly distinguish between odd and even

Eighteenth Lecture: Determinant Formulas and Algebraic Cofactors
================================================================

Determinant formula
-------------------

*   Derive the 2x2 determinant
    
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
    
    We can find that this method involves taking one row at a time, decomposing this row (determinants are linear by rows), extracting factors, obtaining the unit matrix through row exchanges, and then obtaining the answer through properties one and two
    
*   If expanded to a 3x3 matrix, the first row is decomposed into three parts, each of which is further decomposed into three parts for the second row, resulting in a total of 27 parts. The parts that are not zero are those matrices where there are elements in each row and column.
    
*   For example
    
    $$
    \begin{vmatrix}
    a & 0 & 0\\
    0 & 0 & b\\
    0 & c & 0\\
    \end{vmatrix}
    $$
    
    Extract the factors to obtain $abc$ , swap the second and third rows to get the identity matrix, so the answer is $abc*detI=abc$ , and since a row swap was performed, the answer is negative, $-abc$
    
*   A matrix of size n\*n can be divided into $n!$ parts, because the first row is divided into n parts, the second row cannot be repeated, and n-1 rows are chosen, each with one repetition, thus obtaining $n!$ parts
    
*   The determinant formula is the sum of these $n!$ parts
    

Algebraic cofactor
------------------

*   $det=a_{11}(a_{22}a_{33}-a_{23}a_{32})+a_{12}(....)+a_{13}(....)$
*   Extract a factor, the remainder formed by the remaining factors, i.e., the content within the parentheses, is the minor determinant
*   From the matrix perspective, selecting an element, its algebraic cofactor is the determinant of the matrix obtained by excluding the row and column of this element
*   The algebraic cofactor of $a_{ij}$ is denoted as $c_{ij}$
*   Pay attention to the sign of the algebraic cofactor, which is related to the parity of $i+j$ . Even numbers take the positive sign, and odd numbers take the negative sign. Here, the symbol refers to the sign in front of the determinant after the normal calculation of the submatrix corresponding to the algebraic cofactor
*   $detA=a_{11}C_{11}+a_{12}C_{12}+....+a_{1n}C_{1n}$

19th Lecture: Cramer's Rule, Inverse Matrix, Volume
===================================================

Invertible matrix
-----------------

*   Only when the determinant is not zero is the matrix invertible
    
*   Invertible matrix formula
    
    $$
    A^{-1}=\frac{1}{detA}C^T
    $$
    
    The algebraic cofactor of $C_{ij}$ is $A_{ij}$
    
*   Proof: i.e., prove $AC^T=(detA)I$
    
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
    
    On the diagonal are determinants, because $det=a_{11}(a_{22}a_{33}-a_{23}a_{32})+a_{12}(....)+a_{13}(....)$ other positions are all 0, because the algebraic cofactor of row a multiplied by row b is equivalent to calculating the determinant of a matrix where row a and row b are equal, and the determinant is 0
    

Kramer's Rule
-------------

*   Ax=b
    
    $$
    Ax=b \\
    x=A^{-1}b \\
    x=\frac{1}{detA}C^Tb \\
    \\
    x_1=\frac{detB_1}{detA} \\
    x_3=\frac{detB_2}{detA} \\
    ... \\
    $$
    
*   Kramer's rule states that the determinant of matrix $B_i$ is obtained by replacing the ith column of matrix $A$ with b, while keeping the rest unchanged
    

Volume
------

*   The determinant of A can represent a volume, for example, the determinant of a 3x3 matrix represents a volume within a three-dimensional space
    
*   Each row of the matrix represents one edge of a box (originating from the same vertex), and the determinant is the volume of the box; the sign of the determinant represents the left-hand or right-hand system.
    
*   (1) The unit matrix corresponds to the unit cube, with a volume of 1
    
*   For the orthogonal matrix Q,
    
    $$
    QQ^T=I \\
    |QQ^T|=|I| \\
    |Q||Q^T|=1 \\
    {|Q|}^2=1 \\
    |Q|=1 \\
    $$
    
    The box corresponding to Q is the unit cube corresponding to the unit matrix rotated by an angle in space
    
*   (3a) If a row of a matrix is doubled, i.e., one set of edges of the box is doubled, the volume is also doubled. From the perspective of determinants, the factor can be factored out, so the determinant is also doubled
    
*   (2) Swapping two rows of a permutation matrix does not change the volume of the box
    
*   (3b) A row of the matrix is split, and the box is also divided into two parts accordingly
    
*   The above, the three properties of determinants (1, 2, 3a, 3b) can all be verified in terms of volume
    

Lecture 20: Eigenvalues and Eigenvectors
========================================

Feature vector
--------------

*   Given matrix A, matrix A can be regarded as a function acting on a vector x, resulting in the vector Ax
*   When \\( \\mathbf{A} \\) is parallel to \\( \\mathbf{x} \\), i.e., \\( \\frac{\\partial}{\\partial x} \\), we call \\( \\mathbf{v} \\) the eigenvector and \\( \\lambda \\) the eigenvalue
*   If A is a singular matrix, $\lambda = 0$ is an eigenvalue

Several examples
----------------

*   If A is a projection matrix, it can be observed that its eigenvectors are any vectors on the projection plane, because $Ax$ represents the projection onto the plane, and all vectors on the plane remain unchanged after projection, thus being parallel. At the same time, the eigenvalues are 1. If a vector is perpendicular to the plane, $Ax=0$ , the eigenvalue is 0. Therefore, the eigenvectors of the projection matrix A fall into two cases, with eigenvalues of 1 or 0.
    
*   Another example
    
    $$
    A=
    \begin{bmatrix}
    0 & 1 \\
    1 & 0 \\
    \end{bmatrix} \\
    \lambda =1, x=
    \begin{bmatrix}
    1 \\
    1 \\
    \end{bmatrix}
    Ax=
    \begin{bmatrix}
    1 \\
    1 \\
    \end{bmatrix} \\
    \lambda =-1, x=
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
    
*   An n\*n matrix has n eigenvalues
    
*   The sum of the eigenvalues equals the sum of the diagonal elements, this sum being called the trace
    
*   How to solve $Ax=\lambda x$
    
    $$
    (A-\lambda I)x=0 \\
    $$
    
*   The visible equation has non-zero solutions, $(A-\lambda I)$ must be singular, i.e.:
    
    $$
    det(A-\lambda I)=0 \\
    $$
    
*   $$
    If \qquad Ax=\lambda x \\
    Then \qquad (A+3I)x=(\lambda +3)x \\
    $$
    
*   Because the unit matrix is added, the eigenvector remains unchanged as x, and the eigenvalue is increased by the coefficient of the unit matrix, i.e., $(\lambda +3)$
    
*   The eigenvalues of A+B are not necessarily the sum of the eigenvalues of A and B, because their eigenvectors may not be the same. Similarly, the eigenvalues of AB are not necessarily the product of their eigenvalues.
    
*   For another example, consider the rotation matrix Q
    
    $$
    Q=
    \begin{bmatrix}
    0 & -1 \\
    1 & 0 \\
    \end{bmatrix} \\
    trace=0=\lambda _1 +\lambda _2 \\
    det=1=\lambda _1 \lambda _2 \\
    $$
    
*   However, it can be seen that $\lambda _1，\lambda _2$ has no real solutions
    
*   Consider an even worse case (the matrix is more asymmetric, and it is even harder to obtain real eigenvalues)
    
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
    

21st Lecture: Diagonalization and Powers of A
=============================================

Diagonalization
---------------

*   Assuming A has n linearly independent eigenvectors, arranged as columns to form the matrix S, i.e., the eigenvector matrix
    
*   All discussions about matrix diagonalization presented here are under the premise that S is invertible, i.e., the n eigenvectors are linearly independent
    
*   $$
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
    
*   Assuming S is invertible, i.e., the n eigenvectors are linearly independent, we can obtain
    
    $$
    S^{-1}AS=\Lambda \\
    A=S\Lambda S^{-1} \\
    $$
    
*   $\Lambda$ is a diagonal matrix, here we obtain a matrix decomposition other than $A=LU$ and $A=QR$
    
*   $$
    if \qquad Ax=\lambda x \\
    A^2 x=\lambda AX=\lambda ^2 x \\
    A^2=S\Lambda S^{-1} S \Lambda S^{-1}=S \Lambda ^2 S^{-1} \\
    $$
    
*   The two equations above regarding $A^2$ indicate that the squared eigen vectors remain unchanged, the eigenvalues are squared, and similarly for the K-th power
    
*   Eigenvalues and eigenvectors help us understand matrix powers. When calculating matrix powers, we can decompose the matrix into the form of a matrix of eigenvectors multiplied by a diagonal matrix, where K multiplications can cancel each other out, as shown in the above formula
    
*   What kind of matrix's power tends to 0 (stable)
    
    $$
    A^K \rightarrow 0 \quad as \quad K \rightarrow \infty \\
    if \quad all |\lambda _i|<1 \\
    $$
    
*   Which matrices can be diagonalized? If all eigenvalues are different, then A can be diagonalized
    
*   If matrix A is already diagonal, then $\Lambda$ is the same as A
    
*   The number of times an eigenvalue repeats is called the algebraic multiplicity, for triangular matrices, such as
    
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
    
*   For $A-\lambda I$ , the geometric multiplicity is 1, while the algebraic multiplicity of the eigenvalue is 2
    
*   The eigenvector is only (1,0), therefore, for a triangular matrix, it cannot be diagonalized, and there do not exist two linearly independent eigenvectors.
    

A's power
---------

*   Most matrices have a set of linearly independent eigenvalues that can be diagonalized. If diagonalization is possible, we need to focus on how to solve for the powers of A.
    
*   $$
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
    
*   Because the n feature vectors are mutually linearly independent, they can serve as a set of bases to cover the entire n-dimensional space, and naturally, $u_0$ can be represented as a linear combination of the feature vectors, with C being the linear coefficient vector. The above formula has derived the method for solving matrix powers, and the next example will be given using the Fibonacci sequence.
    
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
    
*   The growth rate of the Fibonacci sequence is how fast? Determined by the eigenvalues, we attempt to construct vectors to find the matrix relationship of the iterative Fibonacci sequence
    
    $$
    F_{k+2}=F_{k+1}+F_k \\
    F_{k+1}=F_{k+1} \\
    $$
    
*   Define vector
    
    $$
    u_k=
    \begin{bmatrix}
    F_{k+1} \\
    F_k \\
    \end{bmatrix} \\
    $$
    
*   Using this vector, the first two equations can be written in matrix form
    
    $$
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
    
*   Obtaining two eigenvalues, it is easy to obtain the corresponding eigenvectors
    
*   Returning to the Fibonacci sequence, the growth rate of the Fibonacci sequence is determined by the eigenvalues of the "sequence update matrix" we construct, and as can be seen from $A^{100}u_0=c_1 \lambda _1^100 x_1 + c_2 \lambda _2^100 x_2 +...+c_n \lambda _n^100 x_n$ , the growth rate is mainly determined by the larger eigenvalues, therefore $F_{100}$ can be written in the following form
    
    $$
    F_{100} \approx c_1 {\frac {1 + \sqrt 5}2}^{100} \\
    $$
    
*   There are initial values
    
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
    
*   Among them, $x_1,x_2$ are two feature vectors, whose linear coefficients can be calculated, and by substituting them into the formula, an approximate value of $F_{100}$ can be obtained
    

Summary
-------

*   We find that under the condition of A being invertible, A can be decomposed into the form $S\Lambda S^{-1}$
*   This form has a characteristic that facilitates the calculation of the power of A, as it can be observed that the unit matrix of the eigenvalues of A's power is the power of the unit matrix of A's eigenvalues
*   We attempt to apply this feature in solving the Fibonacci sequence, first converting the sequence update into a matrix form
*   Determine the eigenvalues and eigenvectors of the matrix
*   From the expansion of the power series of A, it can be seen that the power of A is mainly determined by the larger eigenvalues, therefore $F_{100}$ can be written in the form of $F_{100} \approx c_1 {(\frac {1 + \sqrt 5}2)}^{100}$
*   By the initial value $F_0$ , calculate the linear coefficients, substitute them into the above formula, and obtain the approximate value of $F_{100}$
*   This is an example of a difference equation; the next section will discuss differential equations

Lecture 22: Differential Equations and exp(At)
==============================================

Differential Equation
---------------------

*   The solutions to linear equations with constant coefficients are in exponential form; if the solution to the differential equation is in exponential form, one can find the solution by using linear algebra to determine the exponents and coefficients
    
*   For example
    
    $$
    \frac{du_1}{dt}=-u_1+2u_2 \\
    \frac{du_2}{dt}=u_1-2u_2 \\
    u(0)=
    \begin{bmatrix}
    1 \\
    0 \\
    \end{bmatrix} \\
    $$
    
*   First, we list the coefficient matrix and find the eigenvalues and eigenvectors of the matrix
    
    $$
    A=
    \begin{bmatrix}
    -1 & 2 \\
    1 & -2 \\
    \end{bmatrix}
    $$
    
*   $\lambda=0$ is a solution of this singular matrix, and from the trace it can be seen that the second eigenvalue is $\lambda=-3$ , and two eigenvectors are obtained
    
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
    
*   The general solution form of the differential equation will be
    
    $$
    u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2
    $$
    
*   Why?
    
    $$
    \frac{du}{dt} \\
    =c_1 \lambda _1 e^{\lambda _1 t}x_1 \\
    =A c_1 e^{\lambda _1 t}x_1 \\
    because \quad A x_1=\lambda _1 x_1 \\
    $$
    
*   In the differential equation $u_{k+1}=Au_k$ , the form of the solution is $c_1\lambda _1 ^k x_1+c_2 \lambda _2 ^k x_2$
    
*   In the differential equation $\frac {du}{dt}=Au$ , the form of the solution is $u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2$
    
*   Solved from the initial values, i.e., the coefficient matrix C multiplied by the eigenvector matrix S yields the initial values
    
*   It can be seen that as t approaches infinity, the solution of the example equation is reduced to only the steady-state part, i.e., $(\frac 23,\frac 13)$
    
*   When does the solution tend towards 0? There exist negative eigenvalues because $e^{\lambda t}$ needs to tend towards 0
    
*   If the eigenvalues are complex? The magnitude of the imaginary part is 1, so if the real part of the complex number is negative, the solution still tends towards 0
    
*   When does a steady state exist? Only 0 and negative eigenvalues exist, as in the example above
    
*   When does the solution fail to converge? Any eigenvalue has a real part greater than 0
    
*   The sign of the coefficient matrix changes, the eigenvalues also change sign, the steady-state solution remains steady-state, and the convergent solution will become divergent
    
*   How to directly determine if the solution converges from a matrix? That is, do all the real parts of the eigenvalues have a value less than 0?
    
*   The trace of the matrix should be less than 0, but the sum of the diagonal elements being 0 does not necessarily converge, as
    
    $$
    \begin{bmatrix}
    -2 & 0 \\
    0 & 1 \\
    \end{bmatrix}
    $$
    
*   Therefore, another condition is required: the value of the determinant is the product of the eigenvalues, so the value of the determinant should be greater than 0
    

exp(At)
-------

*   Can the solution be expressed in the form of $S,\Lambda$
*   Matrix A represents $u_1,u_2$ coupling, first we need to diagonalize u to decouple
*   $$
    \frac{du}{dt} = Au \\
    set \quad u=Sv \\
    S \frac{dv}{dt} = ASv \\
    \frac{dv}{dt}=S^{-1}ASv=\Lambda v \\
    v(t)=e^{\Lambda t}v(0) \\
    u(t)=Se^{\Lambda t}S^{-1}u(0) \\
    $$
    

21st Lecture: Markov Matrix; Fourier Series
===========================================

Markov matrix
-------------

*   A typical Markov matrix
    
    $$
    \begin{bmatrix}
    0.1 & 0.01 & 0.3 \\
    0.2 & 0.99 & 0.3 \\
    0.7 & 0 & 0.4 \\
    \end{bmatrix}
    $$
    
*   Each element is greater than or equal to 0, the sum of each column is 1, and the powers of the Markov matrix are all Markov matrices
    
*   $\lambda=1$ is an eigenvalue, and the absolute values of the other eigenvalues are all less than 1
    
*   In the previous lecture, we discussed that the power of a matrix can be decomposed into
    
    $$
    u_k=A^ku_0=c_1\lambda _1 ^kx_1+c_2\lambda _2 ^kx_2+.....
    $$
    
*   When A is a Markov matrix, there is only one eigenvalue of 1, and the other eigenvalues are less than 1. As k increases, the terms with eigenvalues less than 1 tend to approach 0, retaining only the term with the eigenvalue of 1, and the elements of the corresponding eigenvector are all greater than 0
    
*   When the sum of each column is 1, there necessarily exists an eigenvalue $\lambda =1$
    
*   Proof:
    
    $$
    A-I=
    \begin{bmatrix}
    -0.9 & 0.01 & 0.3 \\
    0.2 & -0.01 & 0.3 \\
    0.7 & 0 & -0.6 \\
    \end{bmatrix}
    $$
    
*   If 1 is an eigenvalue, then $A-I$ should be singular. It can be seen that the sum of each column of $A-I$ is 0, indicating that the row vectors are linearly dependent, i.e., the matrix is singular, and the all-ones vector lies in the left null space.
    
*   For the Markov matrix A, we study $u_{k+1}=Au_k$
    
*   An example, u is the population in Massachusetts and California, A is the population mobility matrix
    
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
    
*   It can be seen that each year (k), 80% of people stay in Massachusetts, 20% move to California, and 10% from California also relocate to Massachusetts
    
*   On the Markov matrix A
    
    $$
    \begin{bmatrix}
    0.9 & 0.2 \\
    0.1 & 0.8 \\
    \end{bmatrix} \\
    \lambda _1 =1 \\
    \lambda _2 =0.7 \\
    $$
    
*   For the eigenvalue of 1, the eigenvector is easily found as $(2,1)$ , and for the eigenvalue of 0.7, the eigenvector is (-1,1).
    
*   Obtain the formula we are to study
    
    $$
    u_k=c_1*1^k*
    \begin{bmatrix}
    2 \\
    1 \\
    \end{bmatrix}
    +c_2*(0.7)^k*
    \begin{bmatrix}
    -1 \\
    1 \\
    \end{bmatrix}
    $$
    
*   Assuming there were initially 0 people in California and 1000 people in Massachusetts, i.e., $u_0$ , substituting this into the formula yields $c_1,c_2$ . It can be seen that after many years, the populations of California and Massachusetts will stabilize, each accounting for one-third and two-thirds of the total 1000 people, respectively.
    
*   The vector with a sum of 1 is another way to define a Markov matrix
    

Fourier Series
--------------

*   Discussion of projection problems with standard orthogonal bases
    
*   If $q_1....q_n$ is a set of standard orthogonal bases, any vector $v$ is a linear combination of this set of bases
    
*   We now need to determine the linear combination coefficients $x_1....x_n$ , $v=x_1q_1+x_2q_2+...x_nq_n$ . One method is to take the inner product of $v$ and $q_i$ , and calculate the coefficients one by one
    
    $$
    q_1^Tv=x_1q_1^Tq_1+0+0+0....+0=x_1 \\
    $$
    
*   Written in matrix form
    
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
    
*   Now discussing Fourier series
    
*   We hope to decompose the function
    
    $$
    f(x)=a_0+a_1cosx+b_1sinx+a_2cos2x+b_2cos2x+.......
    $$
    
*   The key is that in this decomposition, $coskx,sinkx$ constitutes an infinite orthogonal basis for a set of function spaces, i.e., the inner products of these functions are 0 (the inner product of vectors is a discrete value summation, while the inner product of functions is a continuous value integration).
    
*   How to calculate the Fourier coefficients?
    
*   Using the previous vector example to calculate
    
*   Sequentially compute the inner product of $f(x)$ with each element of the orthogonal basis, obtaining the corresponding coefficient multiplied by $\pi$ , for example
    
    $$
    \int _0 ^{2\pi} f(x)cosx dx=0+ a_1 \int _0^{2\pi}(cosx)^2dx+0+0...+0=\pi a_1 \\
    $$
    

Lecture 22: Symmetric Matrices and Positive Definiteness
========================================================

Symmetric matrix
----------------

*   The eigenvalues of a symmetric matrix are real numbers, and the eigenvectors corresponding to distinct eigenvalues are mutually orthogonal
    
*   For a general matrix $A=S\Lambda S^{-1}$ , S is the matrix of eigenvectors
    
*   For the symmetric matrix $A=Q\Lambda Q^{-1}=Q\Lambda Q^T$ , Q is the matrix of standard orthogonal eigenvectors
    
*   Why are all eigenvalues real numbers?
    
*   Conjugate both left and right, as we are now only considering the real matrix A, $Ax^{*}=\lambda ^{*} x^{*}$
    
*   $\lambda$ and its conjugate are eigenvalues; now take the transpose of both sides of the equation, $x^{* T}A^T=x^{* T} \lambda ^{* T}$
    
*   In the above formula, $A=A^T$ , and both sides are multiplied by $x$ , comparing with $x^{* T}A\lambda x^{* T}x$ yields $\lambda ^{*}=\lambda$ , i.e., the eigenvalues are real numbers
    
*   It is evident that for multiple matrices, the condition $A=A^{* T}$ is required to satisfy symmetry
    
*   For symmetric matrices
    
    $$
    A=Q\Lambda Q^{-1}=Q\Lambda Q^T \\
    =\lambda _1 q_1 q_1^T+\lambda _2 q_2 q_2^T+.... \\
    $$
    
*   So every symmetric matrix is a combination of some mutually orthogonal projection matrices
    
*   For symmetric matrices, the number of positive principal minors is equal to the number of positive eigenvalues, and the product of the principal minors equals the product of the eigenvalues, which equals the determinant of the matrix
    

Positivity
----------

*   Positive definite matrices are symmetric matrices, a subclass of symmetric matrices, whose all eigenvalues are positive, all leading principal minors are positive, and all subdeterminants are positive
*   The sign of eigenvalues is related to stability
*   The eigenvalue, determinant, and main element are unified as one in linear algebra



{% endlang_content %}

{% lang_content zh %}

- 行列式按行是线性的，但行列式本身不是线性的
  
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
- 证明消元不改变行列式
  
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
- 证明转置不改变行列式
  
  $$
  A=LU \\
  $$
- 即证 $|U^TL^T|=|LU|$
  $$
  |U^T||L^T|=|L||U|
  $$ 
- 以上四个矩阵都是三角矩阵，行列式等于对角线乘积，转置没有影响，所以相等 

## 三角阵行列式

- 对三角阵U的行列式,值为对角线上元素乘积(主元乘积)
- 为什么三角阵其他元素不起作用？因为通过消元我们可以得到只有对角元素的矩阵，而消元不改变行列式
- 为什么是对角线元素的乘积？因为可以消元后可以依次把对角元素提出来，即得到$d_1d_2d_3...d_nI$，其中单位矩阵的行列式为1
- 奇异矩阵行列式为0，存在全0行；可逆矩阵行列式不为0，能化成三角阵，行列式是三角矩阵对角元素乘积

## A little more

- 进行奇数次置换和偶数次置换得到的行列式肯定不一样(符号不同)，这意味着进行奇数次置换和偶数次置换后的矩阵不会一样，即置换是严格区分奇偶的

# 第十八讲：行列式公式和代数余子式

## 行列式公式

- 推导2*2行列式
  
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
- 如果扩展到3*3矩阵，则第一行分解成三部分，每部分针对第二行又分解成三部分，所以最后得到27部分，其中不为0的部分是那些各行各列均有元素的矩阵。
- 例如
  
  $$
  \begin{vmatrix}
a & 0 & 0\\
0 & 0 & b\\
0 & c & 0\\
\end{vmatrix}
  $$
  
  先提取出因子，得到$abc$，交换第二行第三行得到单位矩阵，于是答案就是$abc*detI=abc$，又因为进行了一次行交换，所以答案是负的，$-abc$
- n*n的矩阵可以分成$n!$个部分，因为第一行分成n个部分，第二行不能重复，选择n-1行，一次重复，所以得到$n!$部分
- 行列式公式就是这$n!$个部分加起来

## 代数余子式

- $det=a_{11}(a_{22}a_{33}-a_{23}a_{32})+a_{12}(....)+a_{13}(....)$
- 提取出一个因子，由剩余的因子即括号内的内容组成的就是余子式
- 从矩阵上看，选择一个元素，它的代数余子式就是排除这个元素所在行和列剩下的矩阵的行列式
- $a_{ij}$的代数余子式记作$c_{ij}$
- 注意代数余子式的正负，与$i+j$的奇偶性有关，偶数取正，奇数取负，这里的符号是指代数余子式对应的子矩阵正常计算出行列式后前面的符号
- $detA=a_{11}C_{11}+a_{12}C_{12}+....+a_{1n}C_{1n}$    

# 第十九讲：克拉默法则，逆矩阵，体积

## 逆矩阵

- 只有行列式不为0时，矩阵才是可逆的
- 逆矩阵公式
  
  $$
  A^{-1}=\frac{1}{detA}C^T
  $$
  
  其中$C_{ij}$是$A_{ij}$的代数余子式
- 证明：即证$AC^T=(detA)I$
  
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
  
  对角线上都是行列式，因为$det=a_{11}(a_{22}a_{33}-a_{23}a_{32})+a_{12}(....)+a_{13}(....)$
  其他位置都是0，因为行a乘以行b的代数余子式相当于求一个矩阵的行列式，这个矩阵行a与行b相等，行列式为0

## 克拉默法则

- 解Ax=b
  
  $$
  Ax=b \\
x=A^{-1}b \\
x=\frac{1}{detA}C^Tb \\
\\
x_1=\frac{detB_1}{detA} \\
x_3=\frac{detB_2}{detA} \\
... \\
  $$
- 克拉默法则即发现矩阵$B_i$就是矩阵$A$的第i列换成b，其余不变

## 体积

- A的行列式可以代表一个体积，例如3*3矩阵的行列式代表一个三维空间内的体积
- 矩阵的每一行代表一个盒子的一条边(从同一顶点连出的)，行列式就是这个盒子的体积，行列式的正负代表左手或者右手系。
- (1)单位矩阵对应单位立方体，体积为1
- 对正交矩阵Q,
  
  $$
  QQ^T=I \\
|QQ^T|=|I| \\
|Q||Q^T|=1 \\
{|Q|}^2=1 \\
|Q|=1 \\
  $$
  
  Q对应的盒子是单位矩阵对应的单位立方体在空间中旋转过一个角度
- (3a)如果矩阵的某一行翻倍，即盒子一组边翻倍，体积也翻倍，从行列式角度可以把倍数提出来，因此行列式也是翻倍
- (2)交换矩阵两行，盒子的体积不变
- (3b)矩阵某一行拆分，盒子也相应切分为两部分
- 以上，行列式的三条性质(1,2,3a,3b)均可以在体积上验证

# 第二十讲：特征值和特征向量

## 特征向量

- 给定矩阵A，矩阵A可以看成一个函数，作用在一个向量x上，得到向量Ax
- 当Ax平行于x时，即$Ax=\lambda x$，我们称$x$为特征向量，$\lambda$为特征值
- 如果A是奇异矩阵，$\lambda = 0$是一个特征值

## 几个例子

- 如果A是投影矩阵，可以发现它的特征向量就是投影平面上的任意向量，因为$Ax$即投影到平面上，平面上的所有向量投影后不变，自然平行，同时特征值就是1。如果向量垂直于平面，$Ax=0$，特征值为0.因此投影矩阵A的特征向量就分以上两种情况，特征值为1或0.
- 再举一例
  
  $$
  A=
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
\end{bmatrix} \\
\lambda =1, x=
\begin{bmatrix}
1 \\
1 \\
\end{bmatrix}
Ax=
\begin{bmatrix}
1 \\
1 \\
\end{bmatrix} \\
\lambda =-1, x=
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
- n*n矩阵有n个特征值
- 特征值的和等于对角线元素和，这个和称为迹(trace)，
- 如何求解$Ax=\lambda x$
  
  $$
  (A-\lambda I)x=0 \\
  $$
- 可见方程有非零解，$(A-\lambda I)$必须是奇异的 
  即: 
  
  $$
  det(A-\lambda I)=0 \\
  $$
- $$
  If \qquad Ax=\lambda x \\
Then \qquad (A+3I)x=(\lambda +3)x \\
  $$
- 因为加上单位矩阵，特征向量不变依然为x，特征值加上单位矩阵的系数即$(\lambda +3)$
- A+B的特征值不一定是A的特征值加上B的特征值，因为他们的特征向量不一定相同。同理AB的特征值也不一定是他们的特征值的乘积
- 再举一例，对旋转矩阵Q
  
  $$
  Q=
\begin{bmatrix}
0 & -1 \\
1 & 0 \\
\end{bmatrix} \\
trace=0=\lambda _1 +\lambda _2 \\
det=1=\lambda _1 \lambda _2 \\
  $$
- 但是可以看出 $\lambda _1，\lambda _2$无实数解 
- 再看看更加糟糕的情况(矩阵更加不对称，更难得到实数解的特征值)
  
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

# 第二十一讲：对角化和A的幂

## 对角化

- 假设A有n个线性无关特征向量，按列组成矩阵S，即特征向量矩阵
- 以下所有关于矩阵对角化的讨论都在S可逆，即n个特征向量线性无关的前提下
- $$
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

- 假设S可逆，即n个特征向量无关，此时可以得到
  
  $$
  S^{-1}AS=\Lambda \\
A=S\Lambda S^{-1} \\
  $$
- $\Lambda$是对角矩阵，这里我们得到了除了$A=LU$和$A=QR$之外的一种矩阵分解
- $$
  if \qquad Ax=\lambda x \\
A^2 x=\lambda AX=\lambda ^2 x \\
A^2=S\Lambda S^{-1} S \Lambda S^{-1}=S \Lambda ^2 S^{-1} \\
  $$
- 上面关于$A^2$的两式说明平方后特征向量不变，特征值平方，K次方同理
- 特征值和特征向量帮助我们理解矩阵幂，当计算矩阵幂时，我们可以把矩阵分解成特征向量矩阵和对角阵相乘的形式，K个相乘两两可以抵消，如上式
- 什么样的矩阵的幂趋向于0(稳定)
  
  $$
  A^K \rightarrow 0 \quad as \quad K \rightarrow \infty \\
if \quad all |\lambda _i|<1 \\ 
  $$
- 哪些矩阵可以对角化？
  如果所有特征值不同，则A可以对角化
- 如果矩阵A已经是对角阵，则$\Lambda$与A相同
- 特征值重复的次数称为代数重度，对三角阵，如
  
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
- 对$A-\lambda I$，几何重数是1，而特征值的代数重度是2
- 特征向量只有(1,0)，因此对于三角阵，它不可以对角化，不存在两个线性无关的特征向量。

## A的幂

- 多数矩阵拥有互相线性无关的一组特征值，可以对角化。假如可以对角化，我们需要关注如何求解A的幂
- $$
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
- 因为n个特征向量互相不线性相关，因此它们可以作为一组基覆盖整个n维空间，自然$u_0$可以用特征向量的线性组合表示，C是线性系数向量。上式得出了矩阵幂的解法，接下来以斐波那契数列为例
  
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
- 斐波那契数列的增长速度有多快?由特征值决定，我们尝试构造向量，来找到斐波那契数列迭代的矩阵关系
  
  $$
  F_{k+2}=F_{k+1}+F_k \\
F_{k+1}=F_{k+1} \\
  $$
- 定义向量
  
  $$
  u_k=
\begin{bmatrix}
F_{k+1} \\
F_k \\
\end{bmatrix} \\
  $$
- 利用这个向量可以将前两个等式写成矩阵形式 
  
  $$
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
- 得到两个特征值，我们很容易得到特征向量
- 回到斐波那契数列，斐波那契数列的增长速率由我们构造的"数列更新矩阵"的特征值决定，而且由$A^{100}u_0=c_1 \lambda _1^100 x_1 + c_2 \lambda _2^100 x_2 +...+c_n \lambda _n^100 x_n$可以看出增长率主要由由较大的特征值决定，因此$F_{100}$可以写成如下形式
  
  $$
  F_{100} \approx c_1 {\frac {1 + \sqrt 5}2}^{100} \\
  $$
- 再有初始值有
  
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
- 其中$x_1,x_2$是两个特征向量，线性系数可求，代入公式可求$F_{100}$的近似值

## 总结

- 我们发现在A可逆的情况下，A可以分解成$S\Lambda S^{-1}$的形式
- 这种形式有一个特点，方便求A的幂，即分解后可以看出A的幂的特征值单位矩阵是A的特征值单位矩阵的幂
- 我们在求解斐波那契数列中尝试运用此特点，首先将数列的更新转换为矩阵形式
- 求出矩阵的特征值，特征向量
- 由A的幂的展开式可以看出A的幂主要由较大的特征值决定，因此$F_{100}$可以写成$F_{100} \approx c_1 {(\frac {1 + \sqrt 5}2)}^{100}$的形式
- 由初始值$F_0$求出线性系数，代入上式，得到$F_{100}$的近似值
- 以上是差分方程的一个例子，下一节将讨论微分方程

# 第二十二讲：微分方程和exp(At)

## 微分方程

- 常系数线性方程的解是指数形式的，如果微分方程的解是指数形式，只需利用线代求出指数，系数，就可以求出解
- 举个例子
  
  $$
  \frac{du_1}{dt}=-u_1+2u_2 \\
\frac{du_2}{dt}=u_1-2u_2 \\
u(0)=
\begin{bmatrix}
1 \\
0 \\
\end{bmatrix} \\
  $$
- 首先我们列出系数矩阵，并找出矩阵的特征值和特征向量
  
  $$
  A=
\begin{bmatrix}
-1 & 2 \\
1 & -2 \\
\end{bmatrix}
  $$
- 易得$\lambda=0$是这个奇异矩阵的一个解，由迹可以看出第二个特征值是$\lambda=-3$，并得到两个特征向量
  
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
- 微分方程解的通解形式将是
  
  $$
  u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2
  $$
- 为什么？
  
  $$
  \frac{du}{dt} \\
=c_1 \lambda _1 e^{\lambda _1 t}x_1 \\
=A c_1 e^{\lambda _1 t}x_1 \\
because \quad A x_1=\lambda _1 x_1 \\
  $$
- 在差分方程$u_{k+1}=Au_k$当中，解的形式是$c_1\lambda _1 ^k x_1+c_2 \lambda _2 ^k x_2$
- 在微分方程$\frac {du}{dt}=Au$当中，解的形式是$u(t)=c_1e^{\lambda _1 t}x_1+c_1e^{\lambda _2 t}x_2$
- $c_1,c_2$由初始值解出，即系数矩阵C乘特征向量矩阵S得到初始值
- 可以看出t趋于无穷时，例子方程的解只剩下稳态部分，即$(\frac 23,\frac 13)$
- 什么时候解趋向于0？存在负数特征值，因为$e^{\lambda t}$需要趋向于0
- 如果特征值是复数呢？虚数部分的模值是1，所以如果复数的实数部分是负数，解依然趋向于0
- 什么时候存在稳态？特征值中只存在0和负数，就如上面的例子
- 什么时候解无法收敛？任何特征值的实数部分大于0
- 改变系数矩阵的符号，特征值也改变符号，稳态的解依然稳态，收敛的解就会变成发散
- 如何从矩阵直接判断解是否收敛？即特征值的实数部分都小于0？
- 矩阵的迹应该小于0，但对角线之和为0依然不一定收敛，如
  
  $$
  \begin{bmatrix}
-2 & 0 \\
0 & 1 \\
\end{bmatrix}
  $$
- 因此还需要另一个条件：行列式的值是特征值乘积，因此行列式的值应该大于0

## exp(At)

- 是否可以把解表示成$S,\Lambda$的形式
- 矩阵A表示$u_1,u_2$耦合，首先我们需要将u对角化，解耦
- $$
  \frac{du}{dt} = Au \\
set \quad u=Sv \\
S \frac{dv}{dt} = ASv \\
\frac{dv}{dt}=S^{-1}ASv=\Lambda v \\
v(t)=e^{\Lambda t}v(0) \\
u(t)=Se^{\Lambda t}S^{-1}u(0) \\
  $$

# 第二十一讲：马尔科夫矩阵;傅立叶级数

## 马尔科夫矩阵

- 一个典型的马尔科夫矩阵
  
  $$
  \begin{bmatrix}
0.1 & 0.01 & 0.3 \\
0.2 & 0.99 & 0.3 \\
0.7 & 0 & 0.4 \\
\end{bmatrix}
  $$
- 每一个元素大于等于0，每一列之和为1，马尔科夫矩阵的幂都是马尔科夫矩阵
- $\lambda=1$是一个特征值，其余的特征值的绝对值都小于1

- 在上一讲中我们谈到矩阵的幂可以分解为
  
  $$
  u_k=A^ku_0=c_1\lambda _1 ^kx_1+c_2\lambda _2 ^kx_2+.....
  $$
- 当A是马尔科夫矩阵时，只有一个特征值为1，其余特征值小于1，随着k的变大，小于1的特征值所在项趋向于0，只保留特征值为1的那一项，同时对应的特征向量的元素都大于0
- 当每一列和为1时，必然存在一个特征值$\lambda =1$
- 证明：
  
  $$
  A-I=
\begin{bmatrix}
-0.9 & 0.01 & 0.3 \\
0.2 & -0.01 & 0.3 \\
0.7 & 0 & -0.6 \\
\end{bmatrix}
  $$
- 若1是一个特征值，则$A-I$应该是奇异的，可以看到$A-I$每一列和为0，即说明行向量线性相关，即矩阵奇异,同时全1向量在左零空间。
- 对于马尔科夫矩阵A，我们研究$u_{k+1}=Au_k$
- 一个例子，u是麻省和加州的人数，A是人口流动矩阵
  
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
- 可以看到每一年(k)80%的人留在麻省，20%的人前往加州，加州那边也有10%移居麻省
- 对马尔科夫矩阵A
  
  $$
  \begin{bmatrix}
0.9 & 0.2 \\
0.1 & 0.8 \\
\end{bmatrix} \\
\lambda _1 =1 \\
\lambda _2 =0.7 \\
  $$
- 对特征值为1的项，容易求出特征向量为$(2,1)$，对特征值为0.7的项，特征向量为(-1,1)
- 得到我们要研究的公式
  
  $$
  u_k=c_1*1^k*
\begin{bmatrix}
2 \\
1 \\
\end{bmatrix}
+c_2*(0.7)^k*
\begin{bmatrix}
-1 \\
1 \\
\end{bmatrix}
  $$
- 假设一开始加州有0人，麻省有1000人，即$u_0$，代入公式可以得到$c_1,c_2$，可以看到很多年之后，加州和麻省的人数将稳定，各占1000人中的三分之一和三分之二。
- 行向量为和为1是另外一种定义马尔科夫矩阵的方式

## 傅里叶级数

- 先讨论带有标准正交基的投影问题
- 假设$q_1....q_n$是一组标准正交基，任何向量$v$都是这组基的线性组合
- 现在我们要求出线性组合系数$x_1....x_n$
  $v=x_1q_1+x_2q_2+...x_nq_n$
  一种方法是将$v$与$q_i$做内积，逐一求出系数
  
  $$
  q_1^Tv=x_1q_1^Tq_1+0+0+0....+0=x_1 \\
  $$
- 写成矩阵形式
  
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
- 现在讨论傅里叶级数
- 我们希望将函数分解
  
  $$
  f(x)=a_0+a_1cosx+b_1sinx+a_2cos2x+b_2cos2x+.......
  $$
- 关键是，在这种分解中，$coskx,sinkx$构成一组函数空间的无穷正交基，即这些函数内积为0(向量的内积是离散的值累加，函数的内积是连续的值积分)。
- 如何求出傅里叶系数？
- 利用之前的向量例子来求
- 将$f(x)$逐一与正交基元素内积，得到这个正交基元素对应的系数乘$\pi$，例如
  
  $$
  \int _0 ^{2\pi} f(x)cosx dx=0+ a_1 \int _0^{2\pi}(cosx)^2dx+0+0...+0=\pi a_1 \\
  $$

# 第二十二讲：对称矩阵及其正定性

## 对称矩阵

- 对称矩阵的特征值是实数，不重复的特征值对应的特征向量互相正交
- 对一般矩阵$A=S\Lambda S^{-1}$，S为特征向量矩阵
- 对对称矩阵$A=Q\Lambda Q^{-1}=Q\Lambda Q^T$，Q为标准正交的特征向量矩阵
- 为什么特征值都是实数？
- $Ax=\lambda x$对左右同时取共轭，因为我们现在只考虑实数矩阵A，$Ax^{*}=\lambda ^{*} x^{*}$
- 即$\lambda$和它的共轭都是特征值，现在再对等式两边取转置，$x^{* T}A^T=x^{* T} \lambda ^{* T} $
- 上式中$A=A^T$，且两边同乘以$x$，与$x^{* T}A\lambda x^{* T}x$对比可得$\lambda ^{*}=\lambda$，即特征值是实数
- 可见，对于复数矩阵，需要$A=A^{* T}$才满足对称
- 对于对称矩阵
  
  $$
  A=Q\Lambda Q^{-1}=Q\Lambda Q^T \\
=\lambda _1 q_1 q_1^T+\lambda _2 q_2 q_2^T+.... \\
  $$
- 所以每一个对称矩阵都是一些互相垂直的投影矩阵的组合
- 对于对称矩阵，正主元的个数等于正特征值的个数，且主元的乘积等于特征值的乘积等于矩阵的行列式

## 正定性

- 正定矩阵都是对称矩阵，是对称矩阵的一个子类，其所有特征值为正数，所有主元为正数，所有的子行列式都是正数
- 特征值的符号与稳定性有关
- 主元、行列式、特征值三位一体，线性代数将其统一

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