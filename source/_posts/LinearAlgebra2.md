---
title: Note for Linear Algebra 2
date: 2017-01-21 19:28:03
tags: [linearalgebra,math]
categories: Math
mathjax: true
html: true
---

<img src="https://i.mji.rip/2025/07/16/fcab3101f40bfbdcf2ec3e30b6171a26.png" width="500"/>


Lecture 9: Linear Correlation, Basis, Dimension
===============================================

Linear Correlation
------------------

*   Background knowledge: Assume a matrix A, where m < n, i.e., the number of unknowns is greater than the number of equations. Therefore, in the null space, there are vectors other than the zero vector, up to m leading principal elements, and there exist n-m free vectors, and the entire equation system has non-zero solutions.
*   Under what conditions is the vector $x_1,x_2,x_3...x_n$ linearly independent? If there exists a combination of coefficients not all equal to zero such that the linear sum results in 0, then it is linearly dependent; otherwise, it is linearly independent.
*   If there exists a zero vector in the set of vectors, then the set of vectors cannot be linearly independent.
*   If three vectors are randomly drawn in two-dimensional space, they must be linearly dependent. Why? This can be deduced from background knowledge.
*   For a matrix A, we are concerned with whether the columns are linearly dependent; if there exists a non-zero vector in the null space, then the columns are dependent.
*   When $v_1,v_2...v_n$ is the columns of A, if they are unrelated, then what is the null space of A? Only the zero vector. If they are related, then in addition to the zero vector, there exists a non-zero vector in the null space.
*   When the column vectors are linearly independent, all column vectors are leading vectors, and the rank is n. When the column vectors are linearly dependent, the rank is less than n.

<!--more-->

{% language_switch %}

{% lang_content en %}

Generated space, base
---------------------

*   Generated a space, referring to the space containing all linear combinations of these vectors.
    
*   A set of basis in a vector space refers to a group of vectors that have two characteristics: they are linearly independent, and they generate the entire space.
    
*   For example: The most easily thought of basis for $R^3$ is
    
    $$
    \begin{bmatrix}
    1   \\
    0   \\
    0   \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
    0   \\
    1   \\
    0   \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
    0   \\
    0   \\
    1   \\
    \end{bmatrix}
    $$
    
*   This is a set of standard bases, another example:
    
    $$
    \begin{bmatrix}
    1   \\
    1   \\
    2   \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
    2   \\
    2   \\
    5   \\
    \end{bmatrix}
    $$
    
*   It is evident that a space cannot be formed, as taking any vector that does not lie in the plane spanned by these two vectors will suffice.
    
*   How to test if they form a basis? Treat them as columns to form a matrix, which must be invertible (since it is a square matrix in this example).
    
*   If the two vectors in Example 2 cannot form a basis for three-dimensional space, then what space can they form a basis for? The plane formed by these two vectors.
    
*   The basis is not uniquely determined, but all bases share a common feature: the number of vectors in the basis is the same.
    

Dimension
---------

*   The number of all the basis vectors mentioned above is the same, and this number is the dimension of the space. It is not the dimension of the basis vectors, but the number of basis vectors.

Give an example
---------------

On matrix A 
$$
\begin{bmatrix}
1 & 2 & 3 &1  \\
1 & 1 & 2 & 1   \\
1 & 2 & 3 & 1 \\
\end{bmatrix}
$$

*   Four columns are not linearly independent; the first and second columns can be taken as the main columns
    
*   2 = rank of A = number of leading columns = dimension of column space
    
*   The first and second columns form a set of basis for the column space.
    
*   If you know the dimension of the column space, you have determined the number of vectors, and as long as they are linearly independent, these vectors can form a basis.
    
*   What is the dimension of the null space? In this example, the two vectors in the null space (special solutions) are:
    
    $$
    \begin{bmatrix}
    -1   \\
    -1  \\
    1   \\
    0   \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
    -1   \\
    0  \\
    0   \\
    1    \\
    \end{bmatrix}
    $$
    
*   Yes, these two special solutions form a basis for the null space. The dimension of the null space is the number of free variables, which is n-r, i.e., 4-2=2 in this case.
    

Tenth Lecture: Four Basic Subspaces
===================================

*   C(A), N(A), C( $A^T$ ), N( $A^T$ ).
    
*   Respectively located in the $R^m、R^n、R^n、R^m$ space
    
*   The dimension of the column space and the row space are both rank r, the dimension of the null space is n-r, and the dimension of the left null space is m-r
    
*   Basis of the column space: the leading columns, a total of r columns. Basis of the null space: special solutions (free columns), a total of n-r columns. Basis of the row space: non-zero rows in the reduced form of R, a total of r rows.
    
*   Row transformations are linear combinations of row vectors, thus the row spaces of A and R are the same, but the column spaces have changed
    
*   Why is it called the left null space?
    
    $$
    rref\begin{bmatrix}
    A_{m*n} & I_{m*n}
    \end{bmatrix}\rightarrow
    \begin{bmatrix}
    R_{m*n} & E_{m*n}
    \end{bmatrix} \\
    $$
    
*   rref=E, i.e., EA=R
    
*   Through E, the left zero 空洞 can be calculated
    
*   Find the row combination that generates a zero row vector
    
*   The basis of the left null space is the rows corresponding to the non-zero rows of R, totaling m-r rows
    

Eleventh Lecture: Matrix Spaces, Rank-1 Matrices, and Small-World Graphs
========================================================================

Matrix Space
------------

*   Can be regarded as a vector space, can be multiplied by a scalar, and can be added together
    
*   For the matrix space M with $3*3$ as an example, a basis for the space consists of 9 matrices, each containing only one element, 1. This is a set of standard basis, and thus the dimension of this matrix space is 9
    
    $$
    \begin{bmatrix}
    1 & 0 & 0  \\
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 1 & 0  \\
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 0 & 1  \\
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}.....
    \begin{bmatrix}
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    0 & 0 & 1  \\
    \end{bmatrix}
    $$
    
*   The dimension of the subspace S of symmetric matrices in the matrix space $3*3$ is studied again, and it can be seen that among the 9 matrices in the original space basis, 3 belong to the subspace of symmetric matrices, and there are also 3 matrices that are symmetric both above and below the diagonal, so the dimension of the subspace of symmetric matrices is 6
    
    $$
    \begin{bmatrix}
    1 & 0 & 0  \\
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 0 & 0  \\
    0 & 1 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 0 & 0  \\
    0 & 0 & 0  \\
    0 & 0 & 1  \\
    \end{bmatrix}
    $$
    
    $$
    \begin{bmatrix}
    0 & 1 & 0  \\
    1 & 0 & 0  \\
    0 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 0 & 1  \\
    0 & 0 & 0  \\
    1 & 0 & 0  \\
    \end{bmatrix}，
    \begin{bmatrix}
    0 & 0 & 0  \\
    0 & 0 & 1  \\
    0 & 1 & 0  \\
    \end{bmatrix}
    $$
    
*   For the subspace U of the upper triangular matrix, it is easy to obtain that its dimension is 6, and the basis of the element space includes the basis of the subspace
    
*   Next, let's study $S \bigcap U$ , and it can be easily obtained that this subspace is the diagonal matrix D, with a dimension of 3
    
*   If it is $S \bigcup U$ , their union basis can obtain all bases of M, so its dimension is 9
    
*   Organize accordingly
    
    $$
    dim(S)=6,dim(U)=6,dim(S \bigcap U)=3,dim(S \bigcup U)=3 \\
    dim(S)+dim(U)=dim(S \bigcap U)+dim(S \bigcup U) \\
    $$
    
*   Another example can be given to illustrate that a vector space does not necessarily contain vectors, such as the following vector space based on differential equations
    
    $$
    \frac{d^2y}{dx^2}+y=0
    $$
    
*   His several solutions are
    
    $$
    y=cos(x),y=sin(x)
    $$
    
*   Complete solution is
    
    $$
    y=c_1cos(x)+c_2sin(x)
    $$
    
*   A vector space is obtained, with a basis of 2
    

Rank 1 matrix
-------------

*   Write a simple rank-1 matrix
    
    $$
    \begin{bmatrix}
    1 & 4 & 5 \\
    2 & 8 & 10 \\
    \end{bmatrix}=
    \begin{bmatrix}
    1  \\
    2  \\
    \end{bmatrix}*
    \begin{bmatrix}
    1 & 4 & 5 \\
    \end{bmatrix}
    $$
    
*   All rank 1 matrices can be represented as a column multiplied by a row
    
*   Rank 1 matrices are like building blocks, for example, a rank 4 matrix can be constructed from 4 rank 1 matrices
    
*   Consider an example of a rank-1 matrix, in four-dimensional space, let vector $v=(v_1,v_2,v_3,v_4)$ , set $S=\{v|v_1+v_2+v_3+v_4=0\}$ , if S is regarded as the zero space, then the matrix A in the corresponding equation $Av=0$ is
    
    $$
    A=\begin{bmatrix}
    1 & 1 & 1 & 1 \\
    \end{bmatrix}
    $$
    
*   Easily obtainable $dimN(A)=n-r$ , thus the dimension of S is $4-1=3$ , and a set of basis for S is
    
    $$
    \begin{bmatrix}
    -1  \\
    1  \\
    0  \\
    0  \\
    \end{bmatrix},
    \begin{bmatrix}
    -1  \\
    0  \\
    1  \\
    0  \\
    \end{bmatrix},
    \begin{bmatrix}
    -1  \\
    0  \\
    0  \\
    1  \\
    \end{bmatrix}
    $$
    
*   Four subspaces of matrix A: the rank (dimension) of the null space and the column space are both 1, the row space $C(A^T)=\{a,a,a,a\}​$ , the column space $C(A)=R^1​$ , the null space $N(A)​$ , which is the linear combination of the basis of S, $N(A^T)={0}​$
    
*   Organize
    
    $$
    dim(N(A))+dim(C(A^T))=3+1=4=n \\
    dim(C(A))+dim(N(A^T))=1+0=1=m \\
    $$
    

Small World Graph
-----------------

*   Just introduced the concept of graphs, preparing for the next lecture

Twelfth Lecture: Graphs and Networks
====================================

图
-

*   Some basic concepts of graphs, omitted

Internet
--------

*   The adjacency matrix A of the graph, with columns as the nodes of the graph, rows as the edges of the matrix, the starting point as -1, the endpoint as 1, and the rest as 0
    
*   Several rows of linear correlation constitute the circuit, where the circuit implies correlation
    
*   The adjacency matrix A describes the topological structure of the graph
    
*   $dimN(A^T)=m-r​$
    
*   If the nodes of the graph are voltages, $Ax$ where x represents the voltage, $Ax=0$ yields a set of voltage difference equations, the null space is one-dimensional, $A^Ty$ where y represents the current on the edges, the relationship between current and voltage difference is Ohm's law, $A^Ty=0$ obtains Kirchhoff's laws, the null space includes two solutions of Kirchhoff's current equations, which, from the circuit diagram, correspond to two small loops
    
*   Tree is a graph without cycles
    
*   Take another look at $dimN(A^T)=m-r$
    
*   $dimN(A^T)$ = Number of irrelevant circuits
    
*   $m$ = Number of edges
    
*   $r=n-1$ = number of nodes - 1 (since the null space is one-dimensional)
    
*   The: number of nodes - number of edges + number of circuits = 1 (Euler's formula)
    

Summary
-------

*   Potential is denoted as e, $e=Ax$
    
*   Potential difference causes the generation of current, $y=Ce$
    
*   Current satisfies Kirchhoff's Current Law, $A^Ty=0$
    
*   Combine the three equations:
    
    $$
    A^TCAx=f
    $$
    
    This is the most basic balance equation in applied mathematics
    

Lecture Thirteen: Orthogonal Vectors and Subspaces
==================================================

Orthogonal vectors
------------------

*   Orthogonal means perpendicular, indicating that in n-dimensional space, the angles between these vectors are 90 degrees
    
*   When $x^Ty=0$ , x and y are orthogonal, prove:
    
*   If x is orthogonal to y, it follows that:
    
    $$
    {||x||}^2+{||y||}^2={||x+y||}^2 \\
    $$
    
*   That is to say:
    
    $$
    x^Tx+y^Ty={(x+y)}^T(x+y)=x^Tx+y^Ty+x^Ty+xy^T=2x^Ty \\
    $$
    
*   That is to say:
    
    $$
    x^Ty=0 \\
    $$
    
*   Subspaces are orthogonal if all vectors within one subspace are orthogonal to every vector in another subspace. It is obvious that if two two-dimensional subspaces intersect at some vector, then these two spaces are not orthogonal
    
*   If two subspaces are orthogonal, they must not intersect at any non-zero vector, because such a non-zero vector exists in both subspaces simultaneously, and it cannot be perpendicular to itself
    
*   The row space is orthogonal to the null space because $Ax=0$ , i.e., the dot product of each row of the matrix and these linear combinations of the rows (row space) with the solution vector (null space) is 0. This proves the left half of the figure.
    
*   In the right half of the figure, the column space and left null space are the row space and null space of the transpose of matrix A, respectively. The proof given earlier is still valid, thus the column space and left null space are orthogonal, and the right half of the figure holds
    
*   The figure presents the orthogonal subspaces of n-dimensional and m-dimensional spaces, the orthogonal subspace of the n-dimensional space: r-dimensional row space and (n-r)-dimensional null space. The orthogonal subspace of the m-dimensional space: r-dimensional column space and (m-r)-dimensional left null space.
    

Orthogonal subspace
-------------------

*   For example, in three-dimensional space, if the row space is one-dimensional, the null space is two-dimensional. The row space is a straight line, and the null space is a plane perpendicular to this line. This orthogonality can be intuitively seen from a geometric perspective
*   Because the null space is the orthogonal complement of the row space, the null space contains all vectors orthogonal to the row space
*   This is all the knowledge about solving $Ax=0$ . What should we do if we need to solve an unsolvable equation, or to find the optimal solution? We introduce an important matrix $A^TA$
*   $A^TA$ is a $n*n$ square matrix, and it is also symmetric
*   Transforming bad equation into good equation, multiply both sides by $A^T$
*   Not always reversible; if reversible, then $N(A^TA)=N(A)$ , and their ranks are the same
*   Reversible if and only if the null space contains only the zero vector, i.e., the columns are linearly independent; these properties will be proven in the next lecture

14th Lecture: Subspace Projection
=================================

Projection
----------

*   Discussing projection in two-dimensional cases
    
*   A projection of a point b onto another line a, which is the perpendicular line segment drawn from this point to intersect line a at point p; p is the projection point of b onto a, and the vector from the origin to p is the projection vector p. The perpendicular line segment is the error e, where e = b - p
    
*   p in the one-dimensional subspace of a is x times a, i.e., p = xa
    
*   a perpendicular to e, i.e
    
    $$
    a^T(b-xa)=0 \\
    xa^Ta=a^Tb \\
    x= \frac {a^Tb}{a^Ta} \\
    p=a\frac {a^Tb}{a^Ta} \\
    $$
    
*   From the equation, it can be seen that if b is doubled, the projection is also doubled; if a changes, the projection remains unchanged, because the numerator and denominator cancel each other out
    

Projection matrix
-----------------

*   Projection matrix P one-dimensional pattern
*   Multiplying the projection matrix by any vector b will always lie on a line through vector a (i.e., the projection of b onto a, denoted as p), thus the column space of the projection matrix is this line, and the rank of the matrix is 1
*   Other two properties of the projection matrix:
    *   Symmetry, i.e., $P^T=P$
    *   Two projections at the same location, i.e., $P^2=P$

The Significance of Projection
------------------------------

*   Below is discussed in the high-dimensional case
    
*   When the number of equations exceeds the number of unknowns, there is often no solution, and in this case, we can only find the closest solution
    
*   How to find? Refine b such that b is in the column space
    
*   How to fine-tune? Change b to p, which is the one closest to b in the column space, i.e., the projection of b onto the column space when solving $Ax^{'}=p$ , p
    
*   Now we require $x^{'}$ , $p=Ax^{'}$ , the error vector $e=b-Ax^{'}$ , and according to the definition of projection, e needs to be orthogonal to the column space of A
    
*   In summary
    
    $$
    A^T(b-Ax^{'})=0 \\
    $$
    
*   From the above equation, it can be seen that e is in the left null space of A, orthogonal to the column space. Solving the equation yields
    
    $$
    x^{'}=(A^TA)^{-1}A^Tb \\
    p=Ax^{'}=A(A^TA)^{-1}A^Tb \\
    $$
    
*   The n-dimensional mode of the projection matrix P:
    
    $$
    A(A^TA)^{-1}A^T \\
    $$
    
*   The n-dimensional mode of projection matrix P still retains the two properties of the 1-dimensional mode
    
*   Returning to the pursuit of the optimal solution, a common example is to fit a straight line using the least squares method
    
*   Known three points $a_1,a_2,a_3$ , find a straight line to fit close to the three points, $b=C+Da$
    
*   If $a_1=(1,1),a_2=(2,2),a_3=(3,2)$ , then
    
    $$
    C+D=1 \\
    C+2D=2 \\
    C+3D=2 \\
    $$
    
    Written in linear form as:
    
    $$
    \begin{bmatrix}
    1 & 1  \\
    1 & 2  \\
    1 & 3  \\
    \end{bmatrix}
    \begin{bmatrix}
    C  \\
    D  \\
    \end{bmatrix}
    \begin{bmatrix}
    1  \\
    2  \\
    2  \\
    \end{bmatrix}
    $$
    
*   Ax=b, the number of equations is greater than the number of unknowns. If both sides are multiplied by the transpose of A, that is, to find $x^{'}$ , then the fitting line can be obtained. The next lecture will continue with this example.
    

Lecture 15: Projection Matrices and Least Squares Method
========================================================

Projection matrix
-----------------

*   Reviewing, $P=A(A^TA)^{-1}A^T$ , $Pb$ is the projection of b onto the column space of A. Now consider two extreme cases: b being in the column space and b being orthogonal to the column space: b in the column space: $Pb=b$ ; Proof: If b is in the column space, it can be expressed as $b=Ax$ , under the condition that the columns of A are linearly independent, $(A^TA)$ is invertible, substituting $P=A(A^TA)^{-1}A^T$ yields $Pb=b$ ; b is orthogonal to the column space, $Pb=0$ ; Proof: If b is orthogonal to the column space, then b is in the left null space, i.e., $A^Tb=0$ , it is obvious that substituting $P=A(A^TA)^{-1}A^T$ gives $Pb=0$
    
*   p is the projection of b onto the column space, since the column space is orthogonal to the left null space, and thus e is the projection of b onto the left null space, as shown in the figure:
    
    $$
    b=p+e \\
    p=Pb \\
    $$
    
*   Therefore
    
    $$
    e=(I-P)b \\
    $$
    
*   Therefore, the projection matrix of the left null space is $(I-P)$
    

Least Squares Method
--------------------

*   Returning to the example from the previous lecture, find the optimal straight line that approximates three points, minimizing the error, as shown in the figure
    
*   Establish the line as $y=C+Dt$ , substitute the coordinates of three points to obtain a system of equations
    
    $$
    C+D=1 \\
    C+2D=2 \\
    C+3D=2 \\
    $$
    
*   This equation set has no solution but has an optimal price, from an algebraic perspective:
    
    $$
    ||e||^2=(C+D-1)^2+(C+2D-2)^2+(C+3D-2)^2 \\
    $$
    
*   分别对 C 和 D 求偏导为 0，得到方程组:
    
    $$
    \begin{cases}
    3C+6D=5\\
    6C+14D=11\\
    \end{cases}
    $$
    
*   Written in matrix form, here, $C,D$ exists in only one form, and they are unsolvable. To solve $C,D$ , it is treated as a fitting line, i.e., b is replaced by $C,D$ when p is the projection.
    
    $$
    Ax=b \\
    \begin{bmatrix}
    1 & 1 \\
    1 & 2 \\
    1 & 3 \\
    \end{bmatrix}
    \begin{bmatrix}
    C \\
    D \\
    \end{bmatrix}=
    \begin{bmatrix}
    1 \\
    2 \\
    2 \\
    \end{bmatrix}
    $$
    
*   A satisfies the linear independence of each column, b is not in the column space of A, now we want to minimize the error $e=Ax-b$ , how to quantify the error? By squaring its length $||e||^2$ , which in the graph is the sum of the squares of the distances of points to the fitted line along the y-axis. The error line segments $b_1,b_2,b_3$ of these points $e_1,e_2,e_3$ intersect with the fitted line at $p_1,p_2,p_3$ , and when the three b points are replaced by three p points, the system of equations has a solution.
    
*   To solve $x^{'},p$ , given $p=Ax^{'}=A(A^TA)^{-1}A^Tb$ , $Ax=b$ , multiplying both sides by $A^T$ and combining them yields
    
    $$
    A^TAx^{'}=A^Tb
    $$
    
*   Substituting the values yields
    
    $$
    \begin{cases}
    3C+6D=5\\
    6C+14D=11\\
    \end{cases}
    $$
    
*   As with the result of taking partial derivatives algebraically, it is then possible to solve out $C,D$ , thus obtaining the fitting line
    
*   Review the two preceding figures, one explaining the relationship $b,p,e$ , and the other using $C,D$ to determine the fitting line, with the column combination determined by $C,D$ being vector p
    
*   If the columns of matrix A are linearly independent, then $A^TA$ is invertible, and this is a prerequisite for the use of the least squares method. Proof: If a matrix is invertible, then its null space consists only of the zero vector, i.e., x in $A^TAx=0$ must be the zero vector
    
    $$
    A^TAx=0 \\
    x^TA^TAx=0 \\
    (Ax)^T(Ax)=0 \\
    Ax=0 \\
    $$
    
*   Since the columns of A are linearly independent, therefore
    
    $$
    x=0
    $$
    
*   Proof by construction
    
*   For handling mutually perpendicular unit vectors, we introduce the standard orthogonal vector group, where the columns of this matrix are both standard orthogonal and unit vectors. The next lecture will introduce more about the standard orthogonal vector group
    

Lecture 16: Orthogonal Matrices and Gram-Schmidt Orthogonalization
==================================================================

Orthogonal matrix
-----------------

*   A set of orthogonal vectors is known
    
    $$
    q_i^Tq_j=
    \begin{cases}
    0 \quad if \quad i \neq j \\
    1 \quad if \quad i=j \\
    \end{cases}
    $$
    
    $$
    Q=
    \begin{bmatrix}
    q_1 & q_2 & ... & q_n \\
    \end{bmatrix} \\
    Q^TQ=I \\
    $$
    
*   Therefore, for a square matrix with standard orthogonal columns, $Q^TQ=I$ , $Q^T=Q^{-1}$ , i.e., orthogonal matrices, for example
    
    $$
    Q=\begin{bmatrix}
    cos \theta & -sin \theta \\
    sin \theta & cos \theta \\
    \end{bmatrix}or
    \frac {1}{\sqrt 2} 
    \begin{bmatrix}
    1 & 1 \\
    1 & -1 \\
    \end{bmatrix}
    $$
    
*   Q is not necessarily a square matrix. The columns of Q will be the standard orthogonal basis of the column space.
    
*   What is the projection matrix P onto the column space of Q for Q?
    

Gram-Schmidt orthogonalization
------------------------------

*   Given two non-orthogonal vectors a and b, we wish to obtain two orthogonal vectors A, B from a, b, where A can be set as a, and B is the error vector e, which is the projection of b onto A:
    
    $$
    B=e=b-\frac{A^Tb}{A^TA}A
    $$
    
*   Orthogonal basis is A, B divided by their lengths $q_1=\frac{A}{||A||}$
    
*   Extended to the case of three vectors, A, B, and C, from the above formula we know that A, B, and similarly, C needs to have the projection components onto A and B subtracted
    
    $$
    C=c- \frac {A^Tc}{A^TA}A- \frac {B^Tc}{B^TB} B
    $$
    
*   The matrix A composed of column vectors a, b is orthogonalized into an orthogonal matrix Q through Schmidt orthogonalization, and it can be seen from the formula derivation that the columns $q_1,q_2,...$ and $a,b,....$ of Q are in the same column space; orthogonalization can be written as
    
    $$
    A=QR \\
    $$
    
*   即
    
    $$
    \begin{bmatrix}
    a & b \\
    \end{bmatrix}=
    \begin{bmatrix}
    q_1 & q_2 \\
    \end{bmatrix}
    \begin{bmatrix}
    q_1^Ta & q_1^Tb \\
    q_2^Ta & q_2^Tb \\
    \end{bmatrix} \\
    $$
    
*   Among them, because of $QQ^T=I$
    
*   Therefore $R=Q^TA$
    
*   $q_2$ is orthogonal to $q_1$ , while $q_1$ is just the unitization of $a$ , therefore $q_2^Ta=0$ , i.e., $R$ , is an upper triangular matrix
    



{% endlang_content %}

{% lang_content zh %}

## 生成空间、基

- $v_1...,v_l$生成了一个空间，是指这个空间包含这些向量的所有线性组合。

- 向量空间的一组基是指一个向量组，这些向量有两个特性：他们线性无关、他们生成整个空间。

- 举个栗子：求$R^3$的一组基，最容易想到的是
  
  $$
  \begin{bmatrix}
1   \\
0   \\
0   \\
\end{bmatrix}
,
\begin{bmatrix}
0   \\
1   \\
0   \\
\end{bmatrix}
,
\begin{bmatrix}
0   \\
0   \\
1   \\
\end{bmatrix}
  $$

- 这是一组标准基，另一个栗子:
  
  $$
  \begin{bmatrix}
1   \\
1   \\
2   \\
\end{bmatrix}
,
\begin{bmatrix}
2   \\
2   \\
5   \\
\end{bmatrix}
  $$

- 显然无法构成一个空间，只要再取一个不在这两个向量构成的平面上的任意一个向量即可。

- 如何检验他们是一组基？将他们作为列构成一个矩阵，矩阵必须可逆(因为此例中为方阵)。

- 若只有例2中2个向量，他们无法构成三维空间的基，那么他们能构成什么空间的基呢？这两个向量所构成的平面。

- 基不是唯一确定的，但所有的基都有共同点：基中向量的个数是相同的。

## 维数

- 上面提到的所有基向量的个数相同，这个个数就是空间的维数。**不是基向量的维数，而是基向量的个数**

## 最后举个栗子

对矩阵A
$$
\begin{bmatrix}
1 & 2 & 3 &1  \\
1 & 1 & 2 & 1   \\
1 & 2 & 3 & 1 \\
\end{bmatrix}
$$

- 四列并不线性无关，可取第一列第二列为主列
- 2=A的秩=主列数=列空间维数
- 第一列和第二列构成列空间的一组基。
- 如果你知道列空间的维数，则确定了向量的个数，再满足线性无关，这些向量就可以构成一组基。
- 零空间的维数是多少？在本例中零空间中的两个向量(特殊解)为：
  
  $$
  \begin{bmatrix}
-1   \\
-1  \\
1   \\
0   \\
\end{bmatrix}
,
\begin{bmatrix}
-1   \\
0  \\
0   \\
1    \\
\end{bmatrix}
  $$
- 这两个特殊解是否构成了零空间的一组基？是的，零空间的维数就是自由变量的个数,即n-r,在本例中是4-2=2。

# 第十讲：四个基本子空间

- 列空间C(A)，零空间N(A)，行空间C($A^T$)，左零空间N($A^T$)。
- 分别处于$R^m、R^n、R^n、R^m$空间中
- 列空间与行空间的维数都是秩r，零空间维数是n-r，左零空间维数是m-r
- 列空间的基：主列，共r列。零空间的基：特殊解(自由列)，共n-r个。行空间的基：最简形式R的非0行,共r行
- 行变换是行向量的线性组合，因此A和R的行空间相同，列空间发生了变化
- 为什么叫做左零空间？
  
  $$
  rref\begin{bmatrix}
A_{m*n} & I_{m*n}
\end{bmatrix}\rightarrow
\begin{bmatrix}
R_{m*n} & E_{m*n}
\end{bmatrix} \\
  $$
- 易得rref=E，即EA=R 
- 通过E可以计算左零空
- 求左零空间即找一个产生零行向量的行组合 
- 左零空间的基就是R非0行对应的E行,共m-r行 

# 第十一讲：矩阵空间、秩1矩阵和小世界图

## 矩阵空间

- 可以看成向量空间，可以数乘，可以相加
- 以$3*3$矩阵空间M为例，空间的一组基即9个矩阵，每个矩阵中只包含一个元素1,这是一组标准基，可得这个矩阵空间维数是9
  
  $$
  \begin{bmatrix}
1 & 0 & 0  \\
0 & 0 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 1 & 0  \\
0 & 0 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 0 & 1  \\
0 & 0 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}.....
\begin{bmatrix}
0 & 0 & 0  \\
0 & 0 & 0  \\
0 & 0 & 1  \\
\end{bmatrix}
  $$
- 再来研究$3*3$矩阵空间中对称矩阵子空间S的维数，可以看到原空间基中9个矩阵，有3个矩阵属于对称矩阵子空间，另外还有上三角与下三角对称的三个矩阵，所以对称矩阵子空间的维数是6
  
  $$
  \begin{bmatrix}
1 & 0 & 0  \\
0 & 0 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 0 & 0  \\
0 & 1 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 0 & 0  \\
0 & 0 & 0  \\
0 & 0 & 1  \\
\end{bmatrix}
  $$
  
  $$
  \begin{bmatrix}
0 & 1 & 0  \\
1 & 0 & 0  \\
0 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 0 & 1  \\
0 & 0 & 0  \\
1 & 0 & 0  \\
\end{bmatrix}，
\begin{bmatrix}
0 & 0 & 0  \\
0 & 0 & 1  \\
0 & 1 & 0  \\
\end{bmatrix}
  $$
- 对于上三角矩阵子空间U，易得维数为6，且元空间的基包含了子空间的基
- 接着再来研究$S \bigcap U$ ，易得这个子空间即对角矩阵D，维度为3
- 如果是$S \bigcup U $呢？他们的并的基可以得到所有M的基，所以其维数是9
- 整理一下可得
  
  $$
  dim(S)=6,dim(U)=6,dim(S \bigcap U)=3,dim(S \bigcup U)=3 \\
dim(S)+dim(U)=dim(S \bigcap U)+dim(S \bigcup U) \\
  $$
- 再来举一个栗子，说明向量空间不一定有向量，比如下面这个基于微分方程的向量空间
  
  $$
  \frac{d^2y}{dx^2}+y=0 
  $$
- 他的几个解为 
  
  $$
  y=cos(x),y=sin(x) 
  $$
- 完整解为  
  
  $$
  y=c_1cos(x)+c_2sin(x) 
  $$
- 即得到一个向量空间，基为2

## 秩1矩阵

- 先写一个简单的秩1矩阵
  
  $$
  \begin{bmatrix}
1 & 4 & 5 \\
2 & 8 & 10 \\
\end{bmatrix}=
\begin{bmatrix}
1  \\
2  \\
\end{bmatrix}*
\begin{bmatrix}
1 & 4 & 5 \\
\end{bmatrix}
  $$
- 所有的秩1矩阵都可以表示为一列乘一行
- 秩1矩阵就像积木，比如一个秩为4的矩阵可以由4个秩1矩阵构建而成
- 再来看一个秩1矩阵的栗子，在四维空间中，设向量$v=(v_1,v_2,v_3,v_4)$,集合$S=\{v|v_1+v_2+v_3+v_4=0\}$,假如把S看成零空间，则相应的方程$Av=0$中的矩阵A为
  
  $$
  A=\begin{bmatrix}
1 & 1 & 1 & 1 \\
\end{bmatrix}
  $$
- 易得$dimN(A)=n-r$，所以S的维数是$4-1=3$，S的一组基为
  
  $$
  \begin{bmatrix}
-1  \\
1  \\
0  \\
0  \\
\end{bmatrix},
\begin{bmatrix}
-1  \\
0  \\
1  \\
0  \\
\end{bmatrix},
\begin{bmatrix}
-1  \\
0  \\
0  \\
1  \\
\end{bmatrix}
  $$
- 矩阵A的四个子空间:易得行空间和列空间的秩(维数)均为1，行空间$C(A^T)=\{a,a,a,a\}​$，列空间$C(A)=R^1​$，零空间$N(A)​$即S的基线性组合，$N(A^T)={0}​$
- 整理一下
  
  $$
  dim(N(A))+dim(C(A^T))=3+1=4=n \\
dim(C(A))+dim(N(A^T))=1+0=1=m \\
  $$

## 小世界图

- 仅仅引入了图的概念，为下一讲准备

# 第十二讲：图和网络

## 图

- 图的一些基础概念，略过

## 网络

- 图的关联矩阵A，将列作为图的节点，行作为矩阵的边，起点为-1，终点为1，其余为0

- 构成回路的几行线性相关，回路意味着相关

- 关联矩阵A描述了图的拓扑结构

- $dimN(A^T)=m-r​$

- 假如图的节点是电势，$Ax$中x即电势，$Ax=0$得到一组电势差方程，零空间是一维的，$A^Ty$中y即边上的电流，电流与电势差的关系即欧姆定律，$A^Ty=0$得到基尔霍夫定律，零空间包含了基尔霍夫电流方程的两个解，从电路图上看即两个小回路

- 树就是没有回路的图

- 再来看看$dimN(A^T)=m-r$

- $dimN(A^T)$=无关回路数  

- $m$=边数 

- $r=n-1$=节点数-1 (因为零空间是一维的) 

- 即:节点数-边数+回路数=1(欧拉公式) 

## <font size=4>总结

- 将电势记为e,$e=Ax$

- 电势差导致电流产生，$y=Ce$

- 电流满足基尔霍夫电流方程,$A^Ty=0$

- 将三个方程联立：
  
  $$
  A^TCAx=f
  $$
  
  这就是应用数学中最基本的平衡方程

# 第十三讲：正交向量与子空间

## 正交向量

- 正交即垂直，意味着在n维空间内，这些向量的夹角是90度
- 当$x^Ty=0$,x与y正交，证明：
- 若x与y正交，易得:
  
  $$
  {||x||}^2+{||y||}^2={||x+y||}^2 \\
  $$
- 即：
  
  $$
  x^Tx+y^Ty={(x+y)}^T(x+y)=x^Tx+y^Ty+x^Ty+xy^T=2x^Ty \\
  $$
- 即：
  
  $$
  x^Ty=0 \\
  $$

- 子空间正交意味着一个子空间内的所有向量与另一个子空间内的每一个向量正交，显然，如果两个二维子空间在某一向量处相交，则这两个空间一定不正交
- 若两个子空间正交，则他们一定不会相交于某一个非零向量，因为这个非零向量同时存在于两个子空间内，它不可能自己垂直于自己
- 行空间正交于零空间，因为$Ax=0$，即矩阵的每一行以及这些行的线性组合(行空间)与解向量(零空间)点乘都为0。这样就证明了图中左半部分
- 图中右半部份，列空间和左零空间分别是矩阵A的转置矩阵的行空间和零空间，刚才的证明同样有效，因此列空间和左零空间正交，图中右半部份成立
- 图中给出了n维空间和m维空间的正交子空间，n维空间的正交子空间:r维行空间和n-r维零空间。m维空间的正交子空间:r维列空间和m-r维左零空间。

## 正交子空间

- 例如三维空间，假如行空间是一维的，则零空间是二维的，行空间是一条直线，零空间是垂直于这个直线的平面，从几何上可以直观看出他们正交
- 因为零空间是行空间的正交补集，所以零空间包含了所有正交于行空间的向量
- 以上是所有关于解$Ax=0$的知识，如果要解不可解的方程，或者说求最优解，该怎么办呢？我们引入一个重要的矩阵$A^TA$
- $A^TA$是一个$n*n$的方阵，而且对称
- 坏方程转换为好方程，两边同乘$A^T$
- $A^TA$不总是可逆，若可逆，则$N(A^TA)=N(A)$，且他们的秩相同
- $A^TA$可逆当且仅当零空间内只有零向量，即各列线性无关，下一讲将证明这些性质

# 第十四讲：子空间投影

## 投影

- 在二维情况下讨论投影
- 一个点b到另一条直线a的投影，即从这个点做垂直于a的垂线段交a于p点，p即b在a上的投影点，以p为终点的向量即投影p，垂线段即误差e，e=b-p
- p在a的一维子空间里，是a的x倍，即p=xa
- a垂直于e，即
  
  $$
  a^T(b-xa)=0 \\
xa^Ta=a^Tb \\
x= \frac {a^Tb}{a^Ta} \\
p=a\frac {a^Tb}{a^Ta} \\
  $$
- 从式子中可以看到，若b翻倍，则投影翻倍，若a变化，则投影不变，因为分子分母抵消了

## 投影矩阵

- 现在可以引入投影矩阵P的一维模式(projection matrix)，$p=Pb$,$P= \frac {aa^T}{a^Ta}$
- 用任意b乘投影矩阵，总会落在通过a的一条线上(即b在a上的投影p),所以投影矩阵的列空间是这条线，矩阵的秩为1
- 投影矩阵的另外两条性质：
  - 对称,即$P^T=P$
  - 两次投影在相同的位置，即$P^2=P$

## 投影的意义

- 下面在高维情况下讨论
- 当方程数大于未知数个数时，经常无解，这时我们只能找出最接近的解
- 如何找？将b微调，使得b在列空间中
- 怎么微调？将b变成p，即列空间中最接近b的那一个，即转换求解$Ax^{'}=p$,p时b在列空间上的投影
- 现在我们要求$x^{'}$,$p=Ax^{'}$，误差向量$e=b-Ax^{'}$，由投影定义可知e需要垂直于A的列空间
- 综上可得
  
  $$
  A^T(b-Ax^{'})=0 \\
  $$
- 由上式可以看出e在A的左零空间，与列空间正交。解上式可得
  
  $$
  x^{'}=(A^TA)^{-1}A^Tb \\
p=Ax^{'}=A(A^TA)^{-1}A^Tb \\
  $$
- 即投影矩阵P的n维模式:
  
  $$
  A(A^TA)^{-1}A^T \\
  $$
- 投影矩阵P的n维模式依然保留了1维模式的两个性质
- 现在回到求最优解，最常见的一个例子是通过最小二乘法拟合一条直线
- 已知三个点$a_1,a_2,a_3$，找出一条直线拟合接近三个点,$b=C+Da$
- 假如$a_1=(1,1),a_2=(2,2),a_3=(3,2)$,则
  
  $$
  C+D=1 \\
C+2D=2 \\
C+3D=2 \\
  $$
  
  写成线代形式为:
  
  $$
  \begin{bmatrix}
1 & 1  \\
1 & 2  \\
1 & 3  \\
\end{bmatrix}
\begin{bmatrix}
C  \\
D  \\
\end{bmatrix}
\begin{bmatrix}
1  \\
2  \\
2  \\
\end{bmatrix}
  $$
- 即Ax=b,方程数大于未知数个数，若两边乘以A转置，即求$x^{'}$，这样就可以求出拟合直线。下一讲继续此例

# 第十五讲：投影矩阵和最小二乘法

## 投影矩阵

- 回顾，$P=A(A^TA)^{-1}A^T$,$Pb$即b在A的列空间上的投影，现在考虑两种极端情况，b在列空间上和b正交于列空间：
  b在列空间上：$Pb=b$；证明：若b在列空间上，则可以表示为$b=Ax$，在A各列线性无关的条件下，$(A^TA)$可逆，代入$P=A(A^TA)^{-1}A^T$有$Pb=b$
  b正交于列空间，$Pb=0$；证明：若b正交于列空间则b在左零空间内，即$A^Tb=0$，显然代入$P=A(A^TA)^{-1}A^T$有$Pb=0$

- p是b在列空间上的投影，因为列空间正交于左零空间，自然e就是b在左零空间上的投影，如图：
  
  $$
  b=p+e \\
p=Pb \\
  $$
- 所以
  
  $$
  e=(I-P)b \\
  $$
- 所以左零空间的投影矩阵为$(I-P)$ 

## 最小二乘法

- 回到上一讲的例子，找到一条最优直线接近三个点，最小化误差，如图

- 设直线为$y=C+Dt$，代入三个点坐标得到一个方程组
  
  $$
  C+D=1 \\
C+2D=2 \\
C+3D=2 \\
  $$

- 此方程组无解但是存在最优价，从代数角度看：
  
  $$
  ||e||^2=(C+D-1)^2+(C+2D-2)^2+(C+3D-2)^2 \\
  $$

- 分别对C和D求偏导为0，得到方程组: 
  
  $$
  \begin{cases}
3C+6D=5\\
6C+14D=11\\
\end{cases}
  $$

- 写成矩阵形式，这里的$C,D$仅仅存在一个形式，他们无解，要解出$C,D$是将其作为拟合直线，即b被替换为投影p时的$C,D$。
  
  $$
  Ax=b \\
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3 \\
\end{bmatrix}
\begin{bmatrix}
C \\
D \\
\end{bmatrix}=
\begin{bmatrix}
1 \\
2 \\
2 \\
\end{bmatrix}
  $$

- A满足各列线性无关，b不在A的列空间里，现在我们想最小化误差$e=Ax-b$，怎么量化误差？求其长度的平方$||e||^2$，在图中即y轴方向上点到拟合直线的距离的平方和。这些点$b_1,b_2,b_3$的误差线段$e_1,e_2,e_3$与拟合直线交于$p_1,p_2,p_3$，当将三个b点用三个p点取代时，方程组有解。

- 现在要解出$x^{'},p$，已知$p=Ax^{'}=A(A^TA)^{-1}A^Tb$，$Ax=b$，两边同乘$A^T$联立有
  
  $$
  A^TAx^{'}=A^Tb
  $$

- 代入数值可得
  
  $$
  \begin{cases}
3C+6D=5\\
6C+14D=11\\
\end{cases}
  $$

- 与代数求偏导数结果一样,之后可以解出$C,D$，也就得到了拟合直线

- 回顾一下上面两幅图，一张解释了$b,p,e$的关系，另一张用$C,D$确定了拟合直线，由$C,D$确定的列组合就是向量p

- 如果A的各列线性无关，则$A^TA$是可逆的，这时最小二乘法使用的前提，证明：
  如果矩阵可逆，则它的零空间仅为零向量，即$A^TAx=0$中x必须是零向量
  
  $$
  A^TAx=0 \\
x^TA^TAx=0 \\
(Ax)^T(Ax)=0 \\
Ax=0 \\
  $$

- 又因为A各列线性无关，所以
  
  $$
  x=0
  $$

- 即证

- 对于处理相互垂直的单位向量，我们引入标准正交向量组，这个矩阵的各列是标准正交而且是单位向量组，下一讲将介绍更多关于标准正交向量组的内容

# 第十六讲：正交矩阵和Gram-Schmidt正交化

## 正交矩阵

- 已知一组正交向量集
  
  $$
  q_i^Tq_j=
\begin{cases}
0 \quad if \quad i \neq j \\
1 \quad if \quad i=j \\
\end{cases}
  $$
  
  $$
  Q=
\begin{bmatrix}
q_1 & q_2 & ... & q_n \\
\end{bmatrix} \\
Q^TQ=I \\
  $$
- 所以，对有标准正交列的方阵，$Q^TQ=I$,$Q^T=Q^{-1}$,即正交矩阵，例如
  
  $$
  Q=\begin{bmatrix}
cos \theta & -sin \theta \\
sin \theta & cos \theta \\
\end{bmatrix}or
\frac {1}{\sqrt 2} 
\begin{bmatrix}
1 & 1 \\
1 & -1 \\
\end{bmatrix}
  $$
- Q不一定是方阵。Q的各列将是列空间的标准正交基
- 对Q，投影到Q的列空间的投影矩阵P是什么？$P=Q(Q^TQ)^{-1}Q^T=QQ^T$

## Gram-Schmidt正交化

- 给定两个不正交的向量a和b，我们希望从a,b中得到两个正交向量A,B，可设A=a，则B就是b投影到A上的误差向量e：
  
  $$
  B=e=b-\frac{A^Tb}{A^TA}A
  $$
- 正交基就是A,B除以他们的长度$q_1=\frac{A}{||A||}$
- 扩展到求三个向量，即A,B,C的情况，从上式我们已知A，B，同理，C需要c剪去在A和B上的投影分量
  
  $$
  C=c- \frac {A^Tc}{A^TA}A- \frac {B^Tc}{B^TB} B
  $$
- 由a,b组成列向量的矩阵A就通过施密特正交化变成了正交矩阵Q,用公式推导可以看出，Q的各列$q_1,q_2,...$与$a,b,....$在同一列空间内，正交化可以写成
  
  $$
  A=QR \\
  $$
- 即
  
  $$
  \begin{bmatrix}
a & b \\
\end{bmatrix}=
\begin{bmatrix}
q_1 & q_2 \\
\end{bmatrix}
\begin{bmatrix}
q_1^Ta & q_1^Tb \\
q_2^Ta & q_2^Tb \\
\end{bmatrix} \\
  $$
- 其中，因为$QQ^T=I$
- 所以$R=Q^TA$
- $q_2$与$q_1$正交，而$q_1$只是$a$的单位化，所以$q_2^Ta=0$，即$R$是上三角矩阵

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