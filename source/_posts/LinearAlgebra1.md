---
title: Note for Linear Algebra 1
date: 2017-01-21 11:45:28
tags: [linearalgebra,math]
categories: Math
mathjax: true
html: true
---

<img src="https://i.mji.rip/2025/07/16/fcab3101f40bfbdcf2ec3e30b6171a26.png" width="500"/>


First Lecture: Geometric Interpretation of Systems of Equations
===============================================================

*   From three perspectives to view the system of equations: row graph, column graph, matrix

<!--more-->

{% language_switch %}

{% lang_content en %}

*   For example, for the system of equations:

$$
\begin{cases}
2x-y=0\\
-x+2y=3\\
\end{cases}
$$

Image processing
----------------

*   Image of the line:

$$
\begin{bmatrix}
2 & -1 \\
-1 & 2 \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
=
\begin{bmatrix}
0 \\
3 \\
\end{bmatrix}
$$

*   Can also be written as

$$
Ax=b
$$

*   The intersection point of two lines in a two-dimensional plane is the solution to the equation, and this is generalized to n dimensions as the intersection point of n lines in an n-dimensional plane

List images
-----------

*   List of images:
    
    $$
    x
    \begin{bmatrix}
    2  \\
    -1 \\
    \end{bmatrix}
    +y
    \begin{bmatrix}
    -1 \\
    2 \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    0 \\
    3 \\
    \end{bmatrix}
    $$
    
*   The solution to the equation is the linear combination coefficients of the vector set, under which the vectors combine to form the target vector
    

Matrix
------

*   Consider the list of images, different x, y values can lead to different linear combinations. Does there exist a solution for any b, x? Or can the linear combination of these two vectors cover the entire space? Or are these two (or n) vectors linearly independent?
*   If so, then the matrix composed of these two (or n) vectors is called a nonsingular matrix, which is invertible; otherwise, it is called a singular matrix, which is not invertible

Second Lecture: Elimination, Back Substitution, and Replacement
===============================================================

Elimination
-----------

*   Consider the system of equations 
    $$
    \begin{cases}
      x+2y+z=2\\
      3x+8y+z=12\\
      4y+z=2\\
      \end{cases}
    $$
    
*   His A matrix is
    
    $$
    \begin{bmatrix}
    1 & 2 & 1  \\
    3 & 8 & 1  \\
    0 & 4 & 1  \\
    \end{bmatrix}
    $$
    
*   After row transformation becomes
    
    $$
    \begin{bmatrix}
    1 & 2 & 1  \\
    0 & 2 & -2  \\
    0 & 4 & 1  \\
    \end{bmatrix}
    $$
    
*   Re-transformed into
    
    $$
    \begin{bmatrix}
    1 & 2 & 1  \\
    0 & 2 & -2  \\
    0 & 0 & 5  \\
    \end{bmatrix}
    $$
    
*   This series of transformations is the elimination method
    
*   The rule of transformation is that the ith element of the ith row is set as the pivot (p), and through row transformations, the elements before the pivot in each row are successively eliminated, so that matrix A becomes matrix U (upper triangular matrix)
    
*   Matrix
    
    $$
    \left[\begin{array}{c|c}
    A & X \\
    \end{array}\right]
    $$
    
*   Augmented matrix. Applying the same transformation to b yields c.
    

Back-substitution
-----------------

*   Solving the equation Ax=b is equivalent to solving the equation Ux=c, and solving Ux=c is very easy to obtain the solution, taking the three-term equation as an example
*   Because U is an upper triangular matrix, z can be easily obtained
*   Substitute z into the second row to find y
*   Substitute z, y into the first row to find x
*   This process is called back substitution

Replacement
-----------

$$
\begin{bmatrix}
a & b & c  \\
\end{bmatrix}*A
$$

*   The meaning of this expression is to obtain a row matrix whose values are a times the first row of A, b times the second row of A, and c times the third row of A

Similarly 
$$
A*\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix}
$$

*   The meaning of this expression is to obtain a column matrix whose values are a times column 1 of A plus b times column 2 of A plus c times column 3 of A
*   It can be deduced that the matrix obtained by swapping two rows of matrix A is

$$
\begin{bmatrix}
0 & 1  \\
1 & 0  \\
\end{bmatrix}*A
$$

*   The matrix with columns A exchanged is

$$
A*\begin{bmatrix}
0 & 1  \\
1 & 0  \\
\end{bmatrix}
$$

*   The multiplication with a matrix completes row and column transformations; such a matrix is called a permutation matrix
    
*   In elimination, the row and column transformations required to eliminate the element at the i-th row and j-th column are represented as a permutation matrix, denoted as $E_{ij}$
    
*   Elimination can be written as
    
    $$
    E_{32}E_{31}E_{21}A=U
    $$
    

Third Lecture: Multiplication and Inverse Matrices
==================================================

Matrix multiplication
---------------------

*   Consider matrix multiplication
    
    $$
    A*B=C
    $$
    
*   First Algorithm: Dot Product $C_{ij}=\sum_iA_{ik}B_{kj}$
    
*   Second Algorithm: Viewed as matrix multiplication by a vector, the C column is a linear combination of the A columns, with the combination coefficients in the B matrix. For example, the elements in each row of the first column of B are the linear combination coefficients of the individual columns in A. After linear combination, the first column of C is obtained
    
*   Third algorithm: Viewed as a vector multiplying a matrix, the C row is a linear combination of the B row, with the combination coefficients in the A matrix; for example, each element in the first row of the A matrix is the linear combination coefficient for each row in B, and after the linear combination, the first row of C is obtained
    
*   Fourth algorithm: Multiply a column of A with a row of B to obtain a submatrix, and the sum of all submatrices is C
    
*   Fifth Algorithm: Matrix Blocking Algorithm
    

Invertible matrix
-----------------

*   For the inverse matrix $A^{-1}$ , there exists $AA^{-1}=I$ , I is the identity matrix
*   The inverse matrix on the left and the inverse matrix on the right are the same
*   If there exists a non-zero matrix X such that $AX=0$ , then A is not invertible
*   Gaussian Jordan idea for finding the inverse matrix: Treat A|I as an augmented matrix, and when transforming A to I, I is correspondingly transformed to the inverse matrix of A
    *   Proof:
        
        $$
        EA=I \\
        E=A^{-1} \\
        EI=A^{-1} \\
        $$
        

Fourth Lecture: LU Decomposition of A
=====================================

LU decomposition
----------------

*   $(AB)^{-1}=B^{-1}A^{-1}$
    
*   The transpose matrix of A, denoted as $A^T$ , can be easily obtained
    
    $$
    AA^{-1}=I \\
    (A^{-1})^TA^T=I \\
    所以(A^T)^{-1}=(A^{-1})^T \\
    $$
    
*   For a single matrix, transpose and inverse can be interchanged
    
*   Matrix Decomposition: A = LU, where U is transformed back into A through a series of permutation matrices, and L is the cumulative permutation matrix. Taking a 3x3 matrix as an example
    
    $$
    E_{32}E_{31}E_{21}A=U \\
    所以可得L: \\
    L=E_{21}^{-1}E_{31}^{-1}E_{32}^{-1} \\
    $$
    
*   Why study A=LU rather than EA=U: Because if there are no row transformations, the elimination coefficients can be directly written into L; conversely, if E is studied, the operation of the nth row is related to the operation of the already eliminated (n-1)th row, and the elimination coefficients cannot be written down intuitively
    

Elimination Consumption
-----------------------

*   A single multiplication and a single subtraction in the elimination process consume one element at a time (the unit of consumption is the product and subtraction of numbers rather than the product and subtraction of rows), with the total consumption being
    
    $$
    \sum_{i=1}^{n}i*(i-1) \approx \sum_{i=1}^{n}i^2 \approx \frac 13 n^3
    $$
    

群
-

*   For example, using a 3\*3 unitary matrix, there are a total of 6 (i.e., permutation matrices)
*   For these matrices, $P^{-1}=P^T$
*   The permutations and inverses of these 6 matrices still lie within these 6 matrices, and are called a group
*   There are n! row permutation matrices for an n\*n matrix

Lecture 5: Transposition, Permutation, Vector Space R
=====================================================

换
-

*   Replacement matrix is used to perform row exchanges
*   A = LU, L has 1s on the diagonal, below which are the elimination multipliers, and U has zeros below the diagonal
*   PA=LU is used to describe LU decomposition with row permutations
*   P(Permutation permutation matrix) is a unit matrix with rows rearranged, and there are n! permutations of n\*n permutation matrices, which is the number of rearrangements of the rows. They are all invertible, and finding the inverse is equivalent to finding the transpose

Transpose
---------

*   Row exchange is equivalent to transpose, denoted as $A^T$ , $A_{ij}=A_{ji}^T$
*   $(AB)^T=B^TA^T$
*   Symmetric matrix (symmetric), $A^T=A$
*   For any matrix A, $AA^T$ is always symmetric, because $(A^TA)^T=(A^TA^{TT})=(A^TA)$

Vector Space
------------

*   Vectors can be added and subtracted, and can be dotted
*   **A space represents a set of vectors, not all vectors, and a vector space is constrained, requiring the condition of closure under linear combinations**
*   For example, $R^2$ represents the two-dimensional vector space of all real numbers
*   Any vector in a vector space remains within the vector space after linear combination, thus the vector space must contain (0,0)
*   Not an example of a vector space: Only the first quadrant of $R^2$ is taken, vector addition within any space remains within the space, but scalar multiplication is not necessarily so (it can be multiplied by a negative number), and vector spaces are closed
*   A straight line passing through the origin within $R^2$ can be called the vector subspace $R^2$ , which still satisfies the property of self-closedness (addition, subtraction, and scalar multiplication)
*   What are the subspaces of $R^2$ ?
    *   Itself
    *   A straight line extending infinitely on both sides beyond the origin (note that this is different from $R^1$ )
    *   (0,0), abbreviated as Z
*   What are the subspaces of $R^3$ ?
    *   Itself
    *   A straight line extending infinitely on both sides beyond the origin (note that this is different from $R^1$ )
    *   Infinite Plane Beyond Zero Point
    *   (0,0,0)

Through matrix construction into quantum space
----------------------------------------------

$$
A=\begin{bmatrix}
1 & 3  \\
2 & 3  \\
4 & 1  \\
\end{bmatrix}
$$

*   Each column belongs to $R^3$ , and any linear combination (scalar multiplication and addition) of these two columns should be in the subspace, which is called the column space, denoted as C(A). In three-dimensional space, this column space is a plane, passing through these two column vectors and (0,0,0) ![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170122/203313042.png) 

Lecture 6: Column Space and Null Space
======================================

List space
----------

*   The previous lecture mentioned two subspaces, the plane P and the line L. $P \bigcup L$ is not a subspace, $P \bigcap L$ is a subspace.
    
*   For any subspaces S, T, $S \bigcap T$ is a subspace
    
*   Give an example
    
    $$
    A=\begin{bmatrix}
    1 & 1 & 2  \\
    2 & 1 & 3  \\
    3 & 1 & 5  \\
    4 & 1 & 5  \\
    \end{bmatrix}
    $$
    
*   C(A) is the $R^4$ subspace, and the linear combination of these three column vectors yields the subspace
    
*   The subspace is related to the linear system below
    
*   Are there solutions for Ax=b for any b? How can b make x have a solution?
    
    *   The answer to the former is no, because with four equations and three unknowns, the linear combination of three column vectors cannot fill the entire $R^4$ space, i.e., the column space cannot fill the entire four-dimensional space
    *   The latter responds that obviously b=(0,0,0,0) is an answer, and b=(1,2,3,4) is also an answer, i.e., first write any solution (x1,x2,x3), and the calculated b is the b that makes x have a solution, which is equivalent to only when b is in the column space of A, x has a solution
*   If we remove the third column, we can still obtain the same column space, because these three columns are not linearly independent; the third column is the sum of the first two columns. At this point, we call the first two columns the main columns, so the column space in this case is a two-dimensional subspace
    

Zero Space
----------

*   The null space (null space) is completely different from the column space; the null space of A contains all solutions x to Ax = 0
    
*   Column space concerns A, the null space concerns x (in the case where b=0), in the example above, the column space is a subspace of four-dimensional space, and the null space is a subspace of three-dimensional space
    
    $$
    \begin{bmatrix}
    1 & 1 & 2  \\
    2 & 1 & 3  \\
    3 & 1 & 5  \\
    4 & 1 & 5  \\
    \end{bmatrix}
    \begin{bmatrix}
    X_1 \\
    X_2 \\
    X_3 \\
    \end{bmatrix}
    \begin{bmatrix}
    0 \\
    0 \\
    0 \\
    0 \\
    \end{bmatrix}
    $$
    
*   Clearly, the zero space contains (0,0,0) and (1,1,-1), and these two vectors determine a line (c,c,-c), so this line is the zero space
    
*   Why can the zero space be called a space (satisfying the closure property of vector spaces?): namely, to prove that for any two solutions of Ax=0, their linear combination is still a solution. Because: ...matrix multiplication can be expanded... the distributive law...
    

$$
\begin{bmatrix}
1 & 1 & 2  \\
2 & 1 & 3  \\
3 & 1 & 5  \\
4 & 1 & 5  \\
\end{bmatrix}
\begin{bmatrix}
X_1 \\
X_2 \\
X_3 \\
\end{bmatrix}
\begin{bmatrix}
1 \\
2 \\
3 \\
4 \\
\end{bmatrix}
$$

*   We replaced b, and the solution is (1,0,0). Are there other solutions? If there are, can these solutions form a subspace?
*   Clearly not a subspace, as the solution does not contain (0,0,0), which does not satisfy the basic conditions of a vector space, as in this case, the two solutions (1,0,0), (0,-1,1), but the linear combination of these vectors does not pass through the origin, and they cannot form a vector space. Therefore, the discussion of the solution space or null space is based on the condition that b=0.
*   Two methods of constructing subspaces are the column space and the null space
    *   From several vectors, a subspace is obtained through linear combination
    *   From a system of equations, obtain a subspace by making x satisfy specific conditions

Lecture 7: Main Variables, Special Solutions
============================================

Main Variable
-------------

*   How to Solve Ax=0 with Algorithms
    
*   Give an example:
    
    $$
    A=\begin{bmatrix}
    1 & 2 & 2 & 2  \\
    2 & 4 & 6 & 8  \\
    3 & 6 & 8 & 10  \\
    \end{bmatrix}
    $$
    
*   The third line is the sum of the first and second lines; they are linearly related, which will be reflected in the elimination process later
    
*   Elimination does not change the set of equations, because elimination modifies the column space, but not the solution space
    
*   After the first elimination, only the leading element in the first row of the first column is non-zero
    

$$
A=\begin{bmatrix}
1 & 2 & 2 & 2  \\
0 & 0 & 2 & 4  \\
0 & 0 & 2 & 4  \\
\end{bmatrix}
$$

*   At this point, due to linear correlation between the second and third columns, the pivot of the second row has shifted to the third column, and the elimination process continues
    
    $$
    A=\begin{bmatrix}
    1 & 2 & 2 & 2  \\
    0 & 0 & 2 & 4  \\
    0 & 0 & 0 & 0  \\
    \end{bmatrix}=U
    $$
    
*   If we separate the non-zero elements from the zeros, we obtain a step line, with the number of steps being the number of leading elements (non-zero), which is 2 in this case, and we call this the rank of the matrix (the number of equations remaining after elimination), the columns containing the leading elements are called leading columns (1,3), and the remaining columns are free columns (2,4)
    
*   Now we can solve Ux=0 and perform back substitution
    
*   The solutions corresponding to the free columns are the free variables x2, x4, which can be arbitrarily chosen. After selection, the main variables x1, x3 corresponding to the main columns can be solved out by back substitution
    

Special Solution
----------------

*   In this case, if we take x2=1, x4=0, we can obtain x=(-2,1,0,0), and (-2,1,0,0) multiplied by any real number is still a solution, thus determining a line. However, is this line the solution space? No. Because we have two free variables, we can determine more than one line, for example, taking x2=0, x4=1, we can obtain x=(2,0,-2,1).
*   So the algorithm first eliminates variables, obtaining the leading column and the free column, then assigns values (1,0) to the free variables to complete the solution (-2,1,0,0), and then assigns another set of values (0,1) to the free variables to obtain another complete solution (2,0,-2,1).
*   Two special values for the free variables (one being 1, the rest being 0, with none being 0, as that would result in a complete solution that is all zeros) yield two sets of solutions, which are called particular solutions. Based on the particular solutions, we can obtain the solution space: the linear combination of the two particular solutions, a\*(-2,1,0,0) + b\*(2,0,-2,1)
*   r represents the number of principal variables, i.e., the number of principal elements, and only r equations are active. The m\*n matrix A has n-r free variables

Simplified row ladder form
--------------------------

*   U can be further simplified
    
    $$
    U=\begin{bmatrix}
    1 & 2 & 2 & 2  \\
    0 & 0 & 2 & 4  \\
    0 & 0 & 0 & 0  \\
    \end{bmatrix}
    $$
    
*   In the reduced row echelon form (RREF), all entries above the leading pivot are also 0
    
    $$
    U=\begin{bmatrix}
    1 & 2 & 0 & -2  \\
    0 & 0 & 2 & 4  \\
    0 & 0 & 0 & 0  \\
    \end{bmatrix}
    $$
    
*   And the leading element must be made into 1, because b=0, so the second row can be directly divided by 2
    
    $$
    U=\begin{bmatrix}
    1 & 2 & 0 & -2  \\
    0 & 0 & 1 & 2  \\
    0 & 0 & 0 & 0  \\
    \end{bmatrix}=R
    $$
    
*   Simplified row 阶梯 form contains all the information of the matrix in its simplest form
    
*   The unit matrix is located at the intersection of the main row and the main column
    
*   An extremely simplified system of equations is obtained: Rx = 0 (columns can be arbitrarily interchanged), F represents free columns
    
    $$
    R=\begin{bmatrix}
    I & F \\
    0 & 0 \\
    \end{bmatrix}
    $$
    
    Among them, I is the unit matrix (principal column), F is the matrix corresponding to the free columns, R has r rows, I has r columns, and F has n-r columns
    

Zero-space matrix
-----------------

*   Zero space matrix, whose columns are composed of particular solutions, denoted as N, it can be seen that if there are a number of free variables, then N has a columns, and if there are no free variables, N does not exist, x has only a unique solution or no solution
    
    $$
    R*N=0
    $$
    
    $$
    N=\begin{bmatrix}
    -F \\
    I  \\
    \end{bmatrix}
    $$
    
*   The entire equation can be written as
    
    $$
    \begin{bmatrix}
    I & F \\
    \end{bmatrix}
    \begin{bmatrix}
    x_{pivot} \\
    x_{free}  \\
    \end{bmatrix}=0
    $$
    
    $$
    x_{pivot}=-F
    $$
    

Take an example to go through the algorithm again
-------------------------------------------------

*   Original Matrix
    
    $$
    A=\begin{bmatrix}
    1 & 2 & 3 \\
    2 & 4 & 6 \\
    2 & 6 & 8 \\
    2 & 8 & 10 \\
    \end{bmatrix}
    $$
    
*   First elimination
    
    $$
    A=\begin{bmatrix}
    1 & 2 & 3 \\
    0 & 0 & 0 \\
    0 & 2 & 2 \\
    0 & 4 & 4 \\
    \end{bmatrix}
    $$
    
*   Second elimination (perform a row swap to make the second pivot element in the second row)
    
    $$
    A=\begin{bmatrix}
    1 & 2 & 3 \\
    0 & 2 & 2 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}=U
    $$
    
*   Clearly, r=2, 1 degree of freedom, let the degree of freedom be 1, obtain the particular solution x
    
    $$
    x=\begin{bmatrix}
    -1 \\
    -1 \\
    1 \\
    \end{bmatrix}
    $$
    
*   Zero space is denoted as cx, a straight line, with x being the basis of the zero space
    
*   Next, continue to simplify U
    
    $$
    U=\begin{bmatrix}
    1 & 0 & 1 \\
    0 & 1 & 1 \\
    0 & 0 & 0 \\
    0 & 0 & 0 \\
    \end{bmatrix}=R=
    \begin{bmatrix}
    I & F  \\
    0 & 0  \\
    0 & 0  \\
    \end{bmatrix}
    $$
    

$$
F=\begin{bmatrix}
1 \\
1 \\
\end{bmatrix}=U
$$

$$
x=\begin{bmatrix}
-F \\
I  \\
\end{bmatrix}=N
$$

Lecture 8: Solvability and the Structure of Solutions
=====================================================

Solvability
-----------

$$
\begin{cases}
x_1+2x_2+2x_3+2x_4=b_1\\
2x_1+4x_2+6x_3+8x_4=b_2\\
3x_1+6x_2+8x_3+10x_4=b_3\\
\end{cases}
$$

*   Written in the expanded matrix form:
    
    $$
    \left[\begin{array}{c c c c|c}
    1 & 2 & 2 & 2 & b_1 \\
    2 & 4 & 6 & 8 & b_2 \\
    3 & 6 & 8 & 10 & b_3 \\
    \end{array}\right]
    $$
    
*   Elimination yields:
    
    $$
    \left[\begin{array}{c c c c|c}
    1 & 2 & 2 & 2 & b_1 \\
    0 & 0 & 2 & 4 & b_2-2b_1 \\
    0 & 0 & 0 & 0 & b_3-b_2-b_1 \\
    \end{array}\right]
    $$
    
*   The first and third columns are the main columns, while the second and fourth columns are free columns
    
*   Solvability: What conditions must b satisfy when there is a solution? The easily obtainable condition is that b must be in the column space of A
    
*   **If the linear combination of each row of A results in 0, what conditions must b satisfy? Then, the same combination of elements in b must also be zero**
    
*   How to find all solutions of Ax=b?
    
    *   First step: Find a particular solution, set all free variables to 0, and solve for all principal variables. In the example, $x_2和x_4$ is set to 0, which yields $x_1和x_3$ to be -2 and 1.5, respectively.
        
    *   The second step: The complete solution is a particular solution plus any vector in the null space
        
    *   $Ax_{particular}=b   \\   Ax_{nullspace}=0   \\   A(x_{particular}+x_{nullspace})=b$
        
    *   In this case, the particular solution is (-2,0,1.5,0), and the solutions in the null space are (-2,1,0,0) and (2,0,-2,1).
        
    *   Complete solution is:
        
        $$
        x_{complete}=
        \begin{bmatrix}
        -2 \\
        0 \\
        1.5 \\
        0 \\
        \end{bmatrix}+
        c_1\begin{bmatrix}
        -2 \\
        1\\
        0 \\
        0 \\
        \end{bmatrix}+
        c_2\begin{bmatrix}
        2 \\
        0 \\
        -2 \\
        1 \\
        \end{bmatrix}
        $$
        
    *   If its image is plotted in a four-dimensional space with four solutions as axes, it forms a plane, similar to a subspace translating from the zero point to a particular solution point
        

Structure of the Explanation
----------------------------

*   Consider an m\*n matrix of rank r, where r <= m and r <= n, with the case when r is full rank, i.e., r = min(m, n)
    
*   Rank full: r=n
    
    $$
    R=\begin{bmatrix}
    I \\
    0 \\
    \end{bmatrix}
    $$
    
*   Rank full: r = m < n, at this time, no zero row will appear during elimination, and for any b, Ax = b has a solution. There are n - r, i.e., n - m, free variables. At this time, the form of r is
    
    $$
    R=\begin{bmatrix}
    I & F \\
    \end{bmatrix}
    $$
    
*   When r=m=n, A is a reversible matrix, R=I, N(A)={0}, and the system Ax=b has a solution for any b, and the solution is unique
    

An explanation from a netizen's perspective of the vector space
---------------------------------------------------------------

> When the dimension r occupied by the vectors equals the number of vectors n and also equals the dimension m of the ambient space, these vectors can be combined to form any vector within the ambient space, that is, for any value of b, there is always a solution. However, since all vectors must be combined together to reach any coordinate point in the entire ambient space, the stretching of each vector must be a specific amount, meaning x has only one solution. When the dimension r occupied by the vectors equals the dimension m of the ambient space but is less than the number of vectors n, that is, the stretching and combination of some vectors in A can reach any coordinate point in the ambient space, there exist free vectors here. Regardless of the position b takes in the space, you can first arbitrarily stretch your free vector to obtain a new vector, and then combine this new vector with the part of vectors that can completely reach the ambient space through a specific contraction to obtain vector b. As long as the stretching amount of the free vector changes, the contraction amount of the other vectors must also change, so X has infinitely many solutions. (Expressed in terms of the x formula, this means you can use the stretching and combination of some vectors in A (m primitive vectors) to obtain b (this is the particular solution) and then use m primitive vectors and the other n-m free vectors to arbitrarily form a zero vector, thus obtaining infinitely many sets of x.) When the dimension occupied by the vectors equals the number of vectors but is less than the dimension of the ambient space, that is, the vectors in A can only cover a subspace of the ambient space but there are free vectors in this subspace, then if b is within this subspace, the situation is the same as the second point, X has infinitely many solutions; if b is outside the subspace, X cannot be contracted to reach it, so there is no solution.



{% endlang_content %}

{% lang_content zh %}

- 例如对方程组：

$$\begin{cases}
2x-y=0\\
-x+2y=3\\
\end{cases}
$$

## 行图像

- 行图像为：

$$
\begin{bmatrix}
2 & -1 \\
-1 & 2 \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
=
\begin{bmatrix}
0 \\
3 \\
\end{bmatrix}
$$

- 也可以写成

$$
Ax=b
$$

- 即二维平面上两条直线的交点为方程的解，推广到n维就是n维平面上n条直线的交点

## 列图像

- 列图像为：
  
  $$
  x
\begin{bmatrix}
2  \\
-1 \\
\end{bmatrix}
+y
\begin{bmatrix}
-1 \\
2 \\
\end{bmatrix}
=
\begin{bmatrix}
0 \\
3 \\
\end{bmatrix}
  $$

- 方程的解即向量组的线性组合系数，在这个组合系数下向量组合成目标向量

## 矩阵

- 现在考虑列图像，不同的x,y可以导致不同的线性组合，是否对任意b，x都有解？或者说这两个向量的线性组合能否覆盖整个空间？或者说这两(或n)个向量是否线性独立？
- 如果是，那么这两个(或n个)向量组成的矩阵我们称之为非奇异矩阵(nonsingular matrix)，且可逆(invertible)；反之称之为奇异矩阵，不可逆

# 第二讲：消元、回代和置换

## 消元

- 考虑方程组
  $$\begin{cases}
  x+2y+z=2\\
  3x+8y+z=12\\
  4y+z=2\\
  \end{cases}
  $$
- 他的A矩阵为
  
  $$
  \begin{bmatrix}
1 & 2 & 1  \\
3 & 8 & 1  \\
0 & 4 & 1  \\
\end{bmatrix}
  $$
- 经过行变换后为
  
  $$
  \begin{bmatrix}
1 & 2 & 1  \\
0 & 2 & -2  \\
0 & 4 & 1  \\
\end{bmatrix}
  $$
- 再变换为
  
  $$
  \begin{bmatrix}
1 & 2 & 1  \\
0 & 2 & -2  \\
0 & 0 & 5  \\
\end{bmatrix}
  $$
- 这样一系列变换即消元
- 变换的规律是，第i行的第i个元素设为主元(pivot)，通过行变换依次消除每一行主元前面的元素，这样矩阵A就变成了矩阵U(Upper Triangle上三角）
- 矩阵
  
  $$
  \left[\begin{array}{c|c}
A & X \\
\end{array}\right]
  $$
- 称为增广矩阵(Augmented matrix)。b做同样变换可以得到c

## 回代

- 解方程Ax=b等同于解方程Ux=c，Ux=c非常容易求得解，以三元方程为例
- 因为U为上三角矩阵，z很容易求得
- 将z代入第二行求得y
- 将z,y代入第一行求得x
- 这个过程即回代

## 置换

$$
\begin{bmatrix}
a & b & c  \\
\end{bmatrix}*A
$$

- 这个式子的含义是求得一个行矩阵，其值为a倍A行1+b倍A行2+c倍A行3

同理
$$
A*\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix}
$$

- 这个式子的含义是求得一个列矩阵，其值为a倍A列1+b倍A列2+c倍A列3
- 可以推出，交换A两行的矩阵为

$$
\begin{bmatrix}
0 & 1  \\
1 & 0  \\
\end{bmatrix}*A
$$

- 交换A两列的矩阵为

$$
A*\begin{bmatrix}
0 & 1  \\
1 & 0  \\
\end{bmatrix}
$$

- 与矩阵相乘完成了行列变换，这样的矩阵就是置换矩阵
- 在消元中，把第i行第j列处元素消掉所需要的行列变换表示为置换矩阵，记作$E_{ij}$
- 消元可写成
  
  $$
  E_{32}E_{31}E_{21}A=U
  $$

# 第三讲：乘法和逆矩阵

## 矩阵乘法

- 考虑矩阵乘法
  
  $$
  A*B=C
  $$
- 第一种算法：点乘 $C_{ij}=\sum_iA_{ik}B_{kj}$
- 第二种算法：看成矩阵乘向量，C列为A列的线性组合,组合系数在B矩阵中，例如：B的第一列中每行元素就是A中各个列的线性组合系数，线性组合之后得到C的第一列
-  第三种算法：看成向量乘矩阵，C行为B行的线性组合,组合系数在A矩阵中，例如：A的第一行中每列元素就是B中各个行的线性组合系数，线性组合之后得到C的第一行
- 第四种算法：A的某一列乘B的某一行得到一个子矩阵，所有子矩阵相加即为C
- 第五种算法：矩阵分块算

## 逆矩阵

- 对逆矩阵$A^{-1}$,有$AA^{-1}=I$,I为单位矩阵
- 对方阵，左逆矩阵与右逆矩阵相同
- 若存在非零矩阵X,使得$AX=0$,则A不可逆
- 求逆矩阵的高斯若尔当思想:将A|I作为增广矩阵，将A变换到I时，I相应变换到A的逆矩阵
  - 证明：
    
    $$
    EA=I \\
E=A^{-1} \\
EI=A^{-1} \\
    $$

# 第四讲：A的LU分解

## LU分解

- $(AB)^{-1}=B^{-1}A^{-1}$
- 对A的转置矩阵$A^T$,易得
  
  $$
  AA^{-1}=I \\
(A^{-1})^TA^T=I \\
所以(A^T)^{-1}=(A^{-1})^T \\
  $$
- 对单个矩阵而言，转置和求逆可以互换
- 矩阵分解：A=LU,即U通过一系列置换矩阵变回为A，L就是置换矩阵的累积.以3*3矩阵为例
  
  $$
  E_{32}E_{31}E_{21}A=U \\
所以可得L: \\
L=E_{21}^{-1}E_{31}^{-1}E_{32}^{-1} \\
  $$
- 为什么研究A=LU而不是EA=U：因为如果不存在行变换，消元系数可以直接写进L中，反之，如果研究E,第n行的运算与前面已经消元过的第n-1行运算相关，不能直观的写出消元系数

## 消元消耗

- 记消元中一次乘法加一次减法即消掉某一元素为一次消耗(是数的乘和减为单位而不是行的乘和减)，总消耗为
  
  $$
  \sum_{i=1}^{n}i*(i-1) \approx \sum_{i=1}^{n}i^2 \approx \frac 13 n^3
  $$

## 群

- 以3*3单位置换矩阵为例，总共有6个(即行互换矩阵)
- 对这些矩阵，$P^{-1}=P^T$
- 这6个矩阵的置换和逆依然在这6个矩阵之中，称之为群
- n*n矩阵共有n!个行置换矩阵

# 第五讲：转置、置换、向量空间R

## 置换

- 置换矩阵是用来完成行交换的矩阵
- A=LU,L对角线上都是1，下方为消元乘数，U下三角为0
- PA=LU用于描述包含行交换的lu分解
- P(Permutation置换矩阵)是行重新排列了的单位矩阵，n*n置换矩阵共n!种，即各行重新排列后的数目，他们均可逆，且求逆与求转置等价

## 转置

- 行列交换即转置，记作$A^T$，$A_{ij}=A_{ji}^T$
- $(AB)^T=B^TA^T$
- 对称矩阵(symmetric),$A^T=A$
- 对任意矩阵A，$AA^T$总是对称的,因为$(A^TA)^T=(A^TA^{TT})=(A^TA)$

## 向量空间

- 向量可以相加减，点乘
- **空间代表一些向量的集合，不代表所有向量，向量空间是有约束条件的，需要满足对线性组合自封闭的条件**
- 例如$R^2$，代表所有实数的二维向量空间
- 向量空间内的任何向量进行线性组合后依然在向量空间内，所以$R^2$向量空间内必须存在(0,0)
- 不是向量空间的一个例子：只取$R^2$的第一象限，任意空间内的向量相加依然在空间内，但数乘就不一定(可以乘以一个负数),向量空间是封闭的
- 在$R^2$内取一条过零点直线可以称为$R^2$的向量子空间，这个子空间依然满足自封闭性(加减和数乘)
- $R^2$的子空间都有哪些？
  - $R^2$本身
  - 过零点两端无限延伸的直线(注意这和$R^1$不同)
  - (0,0),简写为Z
- $R^3$的子空间都有哪些？
  - $R^3$本身
  - 过零点两端无限延伸的直线(注意这和$R^1$不同)
  - 过零点的无限大平面
  - (0,0,0)

## 通过矩阵构造向量子空间

$$
A=\begin{bmatrix}
1 & 3  \\
2 & 3  \\
4 & 1  \\
\end{bmatrix}
$$

- 各列属于$R^3$，这两列的任何线性组合(数乘和加法)应该在子空间中,称这个子空间为列空间,记作C(A)，在三维空间中这个列空间就是一个平面，过这两个列向量及(0,0,0)
  ![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170122/203313042.png)

# 第六讲：列空间和零空间

## 列空间

- 上一讲提到两种子空间，平面P和直线L。$P \bigcup L$不是一个子空间，$P \bigcap L$是一个子空间

- 对任意子空间S、T,$S \bigcap T$是一个子空间

- 举个栗子
  
  $$
  A=\begin{bmatrix}
1 & 1 & 2  \\
2 & 1 & 3  \\
3 & 1 & 5  \\
4 & 1 & 5  \\
\end{bmatrix}
  $$

- C(A)是$R^4$子空间，将这三个列向量做线性组合可以得到子空间

- 下面将子空间与线性方程组联系起来

- 现提出两个问题：Ax=b对任意b是否都有解?b怎样才能使x有解？
  
  - 前者回答是，否，因为四个方程，三个未知数，等同于3个列向量的线性组合无法填充整个$R^4$空间，**即列空间无法填充整个四维空间**
  - 后者回答，显然b=(0,0,0,0)是一个答案，b=(1,2,3,4)显然也是一个答案,即先写出任意解(x1,x2,x3)，计算出的b就是使x有解的b，**等同于只有b在A的列空间内，x有解**

- 如果我们去掉第三列，我们依然可以得到相同的列空间，因为这三列并不是线性无关，第三列是前两列之和，此时我们称前两列为主列,所以此栗中的列空间是一个二维子空间

## 零空间

- 零空间(null space)与列空间完全不同，A的零空间包含Ax=0的所有解x
- 列空间关心A,零空间关心x(在b=0的情况下)，在上面那个栗子中，列空间是四维空间的子空间,零空间是三维空间的子空间
  
  $$
  \begin{bmatrix}
1 & 1 & 2  \\
2 & 1 & 3  \\
3 & 1 & 5  \\
4 & 1 & 5  \\
\end{bmatrix}
\begin{bmatrix}
X_1 \\
X_2 \\
X_3 \\
\end{bmatrix}
\begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
\end{bmatrix}
  $$
- 显然零空间包含(0,0,0)，(1,1,-1),这两个向量确定一条直线(c,c,-c)，所以这条直线就是零空间
- 为什么零空间可以称为空间(满足向量空间的自封闭性?):即证明对Ax=0的任意两个解，他们的线性组合依然是解。因为：......矩阵乘法可以展开......分配率......

$$
\begin{bmatrix}
1 & 1 & 2  \\
2 & 1 & 3  \\
3 & 1 & 5  \\
4 & 1 & 5  \\
\end{bmatrix}
\begin{bmatrix}
X_1 \\
X_2 \\
X_3 \\
\end{bmatrix}
\begin{bmatrix}
1 \\
2 \\
3 \\
4 \\
\end{bmatrix}
$$

- 我们更换了b，解为(1,0,0),有其他解吗，如果存在，这些解能构成子空间吗？
- 显然不构成，因为解中不包含(0,0,0),不满足向量空间的基本条件，如本例，两个解(1,0,0),(0,-1,1),但这两个向量的线性组合不通过原点，无法组成向量空间。所以讨论解空间或者说零空间，前提是b=0
- 列空间和零空间是两种构造子空间的方法
  - 从几个向量通过线性组合来得到子空间
  - 从一个方程组，通过让x满足特定条件来得到子空间

# 第七讲：主变量、特解

## 主变量

- 如何用算法解Ax=0
- 举个栗子:
  
  $$
  A=\begin{bmatrix}
1 & 2 & 2 & 2  \\
2 & 4 & 6 & 8  \\
3 & 6 & 8 & 10  \\
\end{bmatrix}
  $$
- 第三行是第一行加第二行，他们线性相关，这将在之后的消元中体现出来
- 消元不改变方程的组，因为消元改动列空间,不改动解空间
- 第一次消元之后,第一列只有第一行的主元不为零

$$
A=\begin{bmatrix}
1 & 2 & 2 & 2  \\
0 & 0 & 2 & 4  \\
0 & 0 & 2 & 4  \\
\end{bmatrix}
$$

- 此时因为第二列第三列线性相关，第二行的主元到了第三列,继续消元
  
  $$
  A=\begin{bmatrix}
1 & 2 & 2 & 2  \\
0 & 0 & 2 & 4  \\
0 & 0 & 0 & 0  \\
\end{bmatrix}=U
  $$
- 如果我们将非0元素和0分开，会得到一个阶梯线，阶梯数是主元(非0)数，在本例中是2，我们称之为矩阵的秩(消元后剩下几个方程)，主元所在的列叫主列(1,3)，其余的列是自由列(2,4)
- 现在我们可以解Ux=0,并进行回代
- 自由列所对应的解为自由变量x2,x4，可以任意选择，选定之后主列对应的**主变量**x1,x3可以通过回代解出

## 特解

- 在本例中假如取x2=1,x4=0,可以得到x=(-2,1,0,0),而(-2,1,0,0)乘任意实数依然是解，这样就确定了一条直线，但这条直线是解(零)空间吗?不是。因为我们有两个自由变量，可以确定不止一条直线，例如取x2=0,x4=1,可以得到x=(2,0,-2,1)
- 所以算法是先消元，得到主列和自由列，然后对自由变量分配数值(1,0)，完成整个解(-2,1,0,0),再对自由变量取另外一组值(0,1)，再得到一组完全解(2,0,-2,1)。
- 两次对自由变量取特殊值(其中一个为1，剩下的都是0，不能全为0，那样得到的完整解也全为0)得到的两组解称为**特解**，根据特解我们可以得到解空间：两组特解的线性组合,a\*(-2,1,0,0)+b\*(2,0,-2,1)
-  秩r代表主变量即主元的个数，只有r个方程起作用，m*n的A矩阵有n-r个自由变量

## 简化行阶梯形式

- U还能进一步简化 
  
  $$
  U=\begin{bmatrix}
1 & 2 & 2 & 2  \\
0 & 0 & 2 & 4  \\
0 & 0 & 0 & 0  \\
\end{bmatrix}
  $$
- 在简化行阶梯形式(reduced row echelon form RREF)中，主元上方也全是0
  
  $$
  U=\begin{bmatrix}
1 & 2 & 0 & -2  \\
0 & 0 & 2 & 4  \\
0 & 0 & 0 & 0  \\
\end{bmatrix}
  $$
- 而且需将主元化为1,因为b=0,所以第二行可以直接除以2
  
  $$
  U=\begin{bmatrix}
1 & 2 & 0 & -2  \\
0 & 0 & 1 & 2  \\
0 & 0 & 0 & 0  \\
\end{bmatrix}=R
  $$
- 简化行阶梯形式以最简形式包含了矩阵的所有信息
- 单位矩阵位于主行与主列交汇处
- 最终得到一个极简的方程组:Rx=0(列可以随便交换位置)，F代表自由列
  
  $$
  R=\begin{bmatrix}
I & F \\
0 & 0 \\
\end{bmatrix}
  $$
  
  其中I为单位矩阵(主列)，F(自由列对应的矩阵),R有r行，I有r列，F有n-r列

## 零空间矩阵

- 零空间矩阵，它的各列由特解组成，记作N，可以看出若有a个自由变量，则N有a列，若无自由变量，则N不存在，x只有唯一解或无解
  
  $$
  R*N=0
  $$
  
  $$
  N=\begin{bmatrix}
-F \\
I  \\
\end{bmatrix}
  $$
- 整个方程可以写成
  
  $$
  \begin{bmatrix}
I & F \\
\end{bmatrix}
\begin{bmatrix}
x_{pivot} \\
x_{free}  \\
\end{bmatrix}=0
  $$
  
  $$
  x_{pivot}=-F
  $$

## 最后举个栗子过一遍算法

- 原矩阵
  
  $$
  A=\begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
2 & 6 & 8 \\
2 & 8 & 10 \\
\end{bmatrix}
  $$
- 第一遍消元
  
  $$
  A=\begin{bmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
0 & 2 & 2 \\
0 & 4 & 4 \\
\end{bmatrix}
  $$
- 第二遍消元(进行一次行交换使得第二个主元在第二行)
  
  $$
  A=\begin{bmatrix}
1 & 2 & 3 \\
0 & 2 & 2 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}=U
  $$
- 显然r=2,1个自由变量,令自由变量为1，得到特解x
  
  $$
  x=\begin{bmatrix}
-1 \\
-1 \\
1 \\
\end{bmatrix}
  $$
- 零空间就是cx,一条直线，这个x为零空间的基
- 接下来继续化简U
  
  $$
  U=\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}=R=
\begin{bmatrix}
I & F  \\
0 & 0  \\
0 & 0  \\
\end{bmatrix}
  $$

$$
F=\begin{bmatrix}
1 \\
1 \\
\end{bmatrix}=U
$$

$$
x=\begin{bmatrix}
-F \\
I  \\
\end{bmatrix}=N
$$

# 第八讲：可解性与解的结构

## 可解性

$$\begin{cases}
x_1+2x_2+2x_3+2x_4=b_1\\
2x_1+4x_2+6x_3+8x_4=b_2\\
3x_1+6x_2+8x_3+10x_4=b_3\\
\end{cases}
$$

- 写成增广矩阵形式：
  
  $$
  \left[\begin{array}{c c c c|c}
1 & 2 & 2 & 2 & b_1 \\
2 & 4 & 6 & 8 & b_2 \\
3 & 6 & 8 & 10 & b_3 \\
\end{array}\right]
  $$
- 消元得到:
  
  $$
  \left[\begin{array}{c c c c|c}
1 & 2 & 2 & 2 & b_1 \\
0 & 0 & 2 & 4 & b_2-2b_1 \\
0 & 0 & 0 & 0 & b_3-b_2-b_1 \\
\end{array}\right]
  $$
- 第一列和第三列为主列，第二列和第四列是自由列
- 可解性：有解时b需要满足的条件？易得条件为b必须在A的列空间里
- **如果A各行的线性组合得到0，b需要满足什么条件？那么b中元素的同样组合必然也是零**
- 如何求Ax=b的所有解？
  - 第一步：求一个特解，将所有自由变量设为0，求所有主变量，在例子中，$x_2和x_4$设为0，可以解得$x_1和x_3$分别为-2、1.5
  - 第二步：完整的解为一个特解加上零空间中任意向量
  - $Ax_{particular}=b   \\   Ax_{nullspace}=0   \\   A(x_{particular}+x_{nullspace})=b$
  - 在此例中，特解为(-2,0,1.5,0),零空间中的解为(-2,1,0,0)和(2,0,-2,1)
  - 完整解为：
    
    $$
    x_{complete}=
\begin{bmatrix}
-2 \\
0 \\
1.5 \\
0 \\
\end{bmatrix}+
c_1\begin{bmatrix}
-2 \\
1\\
0 \\
0 \\
\end{bmatrix}+
c_2\begin{bmatrix}
2 \\
0 \\
-2 \\
1 \\
\end{bmatrix}
    $$
  - 其图像如果在以4个解为轴的四维空间中画出，是一个平面，类似于子空间从零点平移过特解点

## 解的结构

- 现在考虑秩为r的m*n矩阵，r<=m，r<=n ，r取满秩时的情况,r=min(m,n)
- 列满秩：r=n<m，此时没有自由变量 ，**N(A)={0}**,Ax=b的解只有特解一个(b在列空间内)，或者无解。此时R的形式为
  
  $$
  R=\begin{bmatrix}
I \\
0 \\
\end{bmatrix}
  $$
- 行满秩：r=m<n，此时消元时不会出现零行，对任意b，Ax=b有解，共有n-r即n-m个自由变量,此时r的形式为
  
  $$
  R=\begin{bmatrix}
I & F \\
\end{bmatrix}
  $$
- r=m=n时，A为可逆矩阵，R=I，N(A)={0},Ax=b对任意b有解，解唯一

## 一个网友从向量空间角度的解释

> 当向量所占的维数r等于向量的个数n又等于母空间m的维数的时候。这些向量就可以组合成母空间内任意的向量了，即无论b为何值一定有解，但由于必须要所有的向量共同组合才能到达整个母空间任意坐标点，所以每个向量的伸缩必须时特定的量，即x只有一组解。
> 当向量所占的维数r等于母空间m的维数的时候小于向量的个数n时，即A中的部分向量伸缩组合就可以到达母空间的任意坐标点。那么这里就存在着自由向量了，无论b取空间里的什么位置，你可以先随意伸缩你的自由向量得到一个新向量，然后通过那部分可以完全到达母空间的向量与这个新向量一起进过特定的收缩得到向量b。只要自由向量的伸缩量改变那么其它向量的收缩量也要跟着改变，那么X就有无穷多组解。（用x的表达公式来描述就是你可以用A中部分向量（m个主元向量）伸缩组合得到b(此为特解）并且再通过m个主元向量与另外n-m个自由向量随意组成0向量，就可以得到无穷多个x组了）
> 当向量所占维数等于向量的个数小于母空间的维数时，即A中的向量无论怎么伸缩组合只达到母空间中的一个子空间。那么当b在这个子空间时那么A通过特定的伸缩可以到达这一坐标点即X有一组解（这里由于没有自由向量所以没有多解的情况，不要存在b只占子空间部分维数留另外的给自由向量的想法，b在r的每个方向都有值，0也是值。就拿子空间为3维空间举例，如果b只在xy平面内，Z仍然需要进行收缩，缩为0，不是自由的）。如果b没在这一子空间内，那么无论A中向量如何收缩都不能得到即无解（同样拿三维举例，如果A中的向量只在xy平面那么如果b为（1 2 3）你如何收缩取得？）
> 当向量所占的维数小于向量的个数小于母空间的个数时，即A中的向量只能覆盖母空间的一个子空间但在这子空间有自由向量，那么如果b在这个子空间内那么情况和第二点相同，X有无穷多组解；如果b在子空间之外，X无论如何收缩都不能达到，无解。

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