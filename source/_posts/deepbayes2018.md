---
title: 深度贝叶斯习题
date: 2018-09-22 10:26:48
tags: [bayes,math,machinelearning]
categories: 数学
mathjax: true
html: true
password: kengbi
---

Deep-Bayes 2018 Summer Camp的习题
发现自己代码能力果然弱......
<!--more--> 

# Deep<span style="color:green">|</span>Bayes summer school. Practical session on EM algorithm
- 第一题就是应用EM算法还原图像，人像和背景叠加在一起，灰度值的概率分布形式已知，设计人像在背景中的位置为隐变量，进行EM迭代推断。
- 具体说明在官网和下面的notebook注释中有，实际上公式已经给出，想要完成作业就是把公式打上去，可以自己推一下公式。

One of the school organisers decided to prank us and hid all games for our Thursday Game Night somewhere.

Let's find the prankster!

When you recognize [him or her](http://deepbayes.ru/#speakers), send:
* name
* reconstructed photo
* this notebook with your code (doesn't matter how awful it is :)

__privately__ to [Nadia Chirkova](https://www.facebook.com/nadiinchi) at Facebook or to info@deepbayes.ru. The first three participants will receive a present. Do not make spoilers to other participants!

Please, note that you have only __one attempt__ to send a message!


```python
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
DATA_FILE = "data_em"
w = 73 # face_width
```

### Data

We are given a set of $K$ images with shape $H \times W$.

It is represented by a numpy-array with shape $H \times W \times K$:


```python
X = np.load(DATA_FILE)
```


```python
X.shape # H, W, K
```




    (100, 200, 1000)



Example of noisy image:


```python
plt.imshow(X[:, :, 0], cmap="Greys_r")
plt.axis("off")
print(X[1,:,0])
```

    [255. 255.  41. 255.   0.  51. 255.  15. 255.  59.   0.   0.   0. 255.
       0. 255. 255.   0. 175.  74. 184.   0.   0. 150. 255. 255.   0.   0.
     148.   0. 255. 181. 255. 255. 255.   0. 255. 255.  30.   0.   0. 255.
       0. 255. 255. 206. 234. 255.   0. 255. 255. 255.   0. 255.   0. 255.
       0. 255. 255. 175.  30. 255.   0.   0. 255.   0. 255.  48.   0.   0.
       0. 232. 162. 255.  26.   0.   0. 255.   0. 255. 173. 255. 255.   0.
       0. 255. 255. 119.   0.   0.   0.   0.   0.   0. 255. 255. 255. 255.
       0.   0. 248.   5. 149. 255.   0. 255. 255. 255.   0. 108.   0.   0.
     255.   0. 255. 255. 255.   0.   0. 193.  79.   0. 255.   0.   0.   0.
     233. 255.   0.  65. 255. 255. 255.   0. 255.   0.   0.   0. 255.  58.
     226. 255.   0. 242. 255. 255.   0. 255.   4. 255. 255.  97. 255.   0.
       0. 255.   0. 255.   0.   0.   0. 255.   0.  43. 219.   0. 255. 255.
     255. 166. 255.   0. 255.  42. 255.  44. 255. 255. 255. 255. 255. 255.
     255. 255.  28.   0.  52. 255.  81. 104. 255. 255.  48. 255. 255. 255.
     102.  25.  30.  73.]
    


![png](http://ojtdnrpmt.bkt.clouddn.com/blog/180924/2L2I5jDbKE.png?imageslim)


### Goal and plan

Our goal is to find face $F$ ($H \times w$).

Also, we will find:
* $B$: background  ($H \times W$)
* $s$: noise standard deviation (float)
* $a$: discrete prior over face positions ($W-w+1$)
* $q(d)$: discrete posterior over face positions for each image  (($W-w+1$) x $K$)

Implementation plan:
1. calculating $log\, p(X  \mid d,\,F,\,B,\,s)$
1. calculating objective
1. E-step: finding $q(d)$
1. M-step: estimating $F,\, B, \,s, \,a$
1. composing EM-algorithm from E- and M-step


### Implementation


```python
### Variables to test implementation
tH, tW, tw, tK = 2, 3, 1, 2
tX = np.arange(tH*tW*tK).reshape(tH, tW, tK)
tF = np.arange(tH*tw).reshape(tH, tw)
tB = np.arange(tH*tW).reshape(tH, tW)
ts = 0.1
ta = np.arange(1, (tW-tw+1)+1)
ta = ta / ta.sum()
tq = np.arange(1, (tW-tw+1)*tK+1).reshape(tW-tw+1, tK)
tq = tq / tq.sum(axis=0)[np.newaxis, :]
```

#### 1. Implement calculate_log_probability
For $k$-th image $X_k$ and some face position $d_k$:
$$p(X_k  \mid d_k,\,F,\,B,\,s) = \prod_{ij}
    \begin{cases} 
      \mathcal{N}(X_k[i,j]\mid F[i,\,j-d_k],\,s^2), 
      & \text{if}\, (i,j)\in faceArea(d_k)\\
      \mathcal{N}(X_k[i,j]\mid B[i,j],\,s^2), & \text{else}
    \end{cases}$$

Important notes:
* Do not forget about logarithm!
* This implementation should use no more than 1 cycle!


```python
def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k, F, B, s) for all images X_k in X and
    all possible face position d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(W-w+1, K)
        ll[dw, k] - log-likelihood of observing image X_k given
        that the prankster's face F is located at position dw
    """
    # your code here
    H = X.shape[0]
    W = X.shape[1]
    K = X.shape[2]
    w = F.shape[1]
    ll = np.zeros((W-w+1,K))
    for k in range(K):
        X_minus_B = X[:,:,k]-B[:,:]
        XB = np.multiply(X_minus_B,X_minus_B)
        for dk in range(W-w+1):
            F_temp = np.zeros((H,W))
            F_temp[:,dk:dk+w] = F
            X_minus_F = X[:,:,k] - F_temp[:,:]
            XF = np.multiply(X_minus_F,X_minus_F)
            XB_mask = np.ones((H,W))
            XB_mask[:,dk:dk+w] = 0
            XF_mask = 1-XB_mask
            XB_temp = np.multiply(XB,XB_mask)
            XF_temp = np.multiply(XF,XF_mask)   
            Sum = (np.sum(XB_temp+XF_temp))*(-1/(2*s**2))-H*W*np.log(np.sqrt(2*np.pi)*s)
            ll[dk][k]=Sum    
    return ll
```


```python
# run this cell to test your implementation
expected = np.array([[-3541.69812064, -5541.69812064],
       [-4541.69812064, -6741.69812064],
       [-6141.69812064, -8541.69812064]])
actual = calculate_log_probability(tX, tF, tB, ts)
assert np.allclose(actual, expected)
print("OK")
```

    OK
    

#### 2. Implement calculate_lower_bound
$$\mathcal{L}(q, \,F, \,B,\, s,\, a) = \sum_k \biggl (\mathbb{E} _ {q( d_k)}\bigl ( \log p(  X_{k}  \mid {d}_{k} , \,F,\,B,\,s) + 
    \log p( d_k  \mid a)\bigr) - \mathbb{E} _ {q( d_k)} \log q( d_k)\biggr) $$
    
Important notes:
* Use already implemented calculate_log_probability! 
* Note that distributions $q( d_k)$ and $p( d_k  \mid a)$ are discrete. For example, $P(d_k=i \mid a) = a[i]$.
* This implementation should not use cycles!


```python
def calculate_lower_bound(X, F, B, s, a, q):
    """
    Calculates the lower bound L(q, F, B, s, a) for 
    the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1)
        Estimate of prior on position of face in any image.
    q : array
        q[dw, k] - estimate of posterior 
                   of position dw
                   of prankster's face given image Xk

    Returns
    -------
    L : float
        The lower bound L(q, F, B, s, a) 
        for the marginal log likelihood.
    """
    # your code here
    K = X.shape[2]
    ll = calculate_log_probability(X,F,B,s)
    ll_expectation = np.multiply(ll,q)
    q_expectation = np.multiply(np.log(q),q)
    dk_expection = 0
    for k in range(K):
        dk_expection += np.multiply(np.log(a),q[:,k])
    L = np.sum(ll_expectation)-np.sum(q_expectation)+np.sum(dk_expection)
    
    return L
```


```python
# run this cell to test your implementation
expected = -12761.1875
actual = calculate_lower_bound(tX, tF, tB, ts, ta, tq)
assert np.allclose(actual, expected)
print("OK")
```

    OK
    

#### 3. Implement E-step
$$q(d_k) = p(d_k \mid X_k, \,F, \,B, \,s,\, a) = 
\frac {p(  X_{k}  \mid {d}_{k} , \,F,\,B,\,s)\, p(d_k \mid a)}
{\sum_{d'_k} p(  X_{k}  \mid d'_k , \,F,\,B,\,s) \,p(d'_k \mid a)}$$

Important notes:
* Use already implemented calculate_log_probability!
* For computational stability, perform all computations with logarithmic values and apply exp only before return. Also, we recommend using this trick:
$$\beta_i = \log{p_i(\dots)} \quad\rightarrow \quad
  \frac{e^{\beta_i}}{\sum_k e^{\beta_k}} = 
  \frac{e^{(\beta_i - \max_j \beta_j)}}{\sum_k e^{(\beta_k- \max_j \beta_j)}}$$
* This implementation should not use cycles!


```python
def run_e_step(X, F, B, s, a):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k, F, B, s, a).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(H, w)
        Estimate of prankster's face.
    B : array shape(H, W)
        Estimate of background.
    s : float
        Eestimate of standard deviation of Gaussian noise.
    a : array, shape(W-w+1)
        Estimate of prior on face position in any image.

    Returns
    -------
    q : array
        shape (W-w+1, K)
        q[dw, k] - estimate of posterior of position dw
        of prankster's face given image Xk
    """
    # your code here
    ll = calculate_log_probability(X,F,B,s)
    K = X.shape[2]
    for k in range(K):
        max_ll = ll[:,k].max()
        ll[:,k] -= max_ll
        ll[:,k] = np.exp(ll[:,k])*a
        denominator = np.sum(ll[:,k])
        ll[:,k] /= denominator
    q = ll
    return q
```


```python
# run this cell to test your implementation
expected = np.array([[ 1.,  1.],
                   [ 0.,  0.],
                   [ 0.,  0.]])
actual = run_e_step(tX, tF, tB, ts, ta)
assert np.allclose(actual, expected)
print("OK")
```

    OK
    

#### 4. Implement M-step
$$a[j] = \frac{\sum_k q( d_k = j )}{\sum_{j'}  \sum_{k'} q( d_{k'} = j')}$$
$$F[i, m] = \frac 1 K  \sum_k \sum_{d_k} q(d_k)\, X^k[i,\, m+d_k]$$
$$B[i, j] = \frac {\sum_k \sum_{ d_k:\, (i, \,j) \,\not\in faceArea(d_k)} q(d_k)\, X^k[i, j]} 
      {\sum_k \sum_{d_k: \,(i, \,j)\, \not\in faceArea(d_k)} q(d_k)}$$
$$s^2 = \frac 1 {HWK}   \sum_k \sum_{d_k} q(d_k)
      \sum_{i,\, j}  (X^k[i, \,j] - Model^{d_k}[i, \,j])^2$$

where $Model^{d_k}[i, j]$ is an image composed from background and face located at $d_k$.

Important notes:
* Update parameters in the following order: $a$, $F$, $B$, $s$.
* When the parameter is updated, its __new__ value is used to update other parameters.
* This implementation should use no more than 3 cycles and no embedded cycles!


```python
def run_m_step(X, q, w):
    """
    Estimates F, B, s, a given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    q  :
        q[dw, k] - estimate of posterior of position dw
                   of prankster's face given image Xk
    w : int
        Face mask width.

    Returns
    -------
    F : array, shape (H, w)
        Estimate of prankster's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1)
        Estimate of prior on position of face in any image.
    """
    # your code here
    K = X.shape[2]
    W = X.shape[1]
    H = X.shape[0]
    
    a = np.sum(q,axis=1) / np.sum(q)

    F = np.zeros((H,w))
    for k in range(K):
        for dk in range(W-w+1):
            F+=q[dk][k]*X[:,dk:dk+w,k]
    F = F / K
            
    
    B = np.zeros((H,W))
    denominator = np.zeros((H,W))
    for k in range(K):
        for dk in range(W-w+1):
            mask = np.ones((H,W))
            mask[:,dk:dk+w] = 0
            B += np.multiply(q[dk][k]*X[:,:,k],mask)
            denominator += q[dk][k]*mask
    denominator = 1/denominator
    B = B * denominator
    
    s = 0
    for k in range(K):
        for dk in range(W-w+1):
            F_B = np.zeros((H,W))
            F_B[:,dk:dk+w]=F
            mask = np.ones((H,W))
            mask[:,dk:dk+w] = 0
            Model = F_B + np.multiply(B,mask)
            temp = X[:,:,k]-Model[:,:]
            temp = np.multiply(temp,temp)
            temp = np.sum(temp)
            temp *= q[dk][k]
            s += temp
    s = np.sqrt(s /(H*W*K))          
    
    return F,B,s,a
    
    
```


```python
# run this cell to test your implementation
expected = [np.array([[ 3.27777778],
                      [ 9.27777778]]),
 np.array([[  0.48387097,   2.5       ,   4.52941176],
           [  6.48387097,   8.5       ,  10.52941176]]),
  0.94868,
 np.array([ 0.13888889,  0.33333333,  0.52777778])]
actual = run_m_step(tX, tq, tw)
for a, e in zip(actual, expected):
    assert np.allclose(a, e)
print("OK")
```

    OK
    

#### 5. Implement EM_algorithm
Initialize parameters, if they are not passed, and then repeat E- and M-steps till convergence.

Please note that $\mathcal{L}(q, \,F, \,B, \,s, \,a)$ must increase after each iteration.


```python
def run_EM(X, w, F=None, B=None, s=None, a=None, tolerance=0.001,
           max_iter=50):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    w : int
        Face mask width.
    F : array, shape (H, w), optional
        Initial estimate of prankster's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    a : array, shape (W-w+1), optional
        Initial estimate of prior on position of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.

    Returns
    -------
    F, B, s, a : trained parameters.
    LL : array, shape(number_of_iters + 2,)
        L(q, F, B, s, a) at initial guess, 
        after each EM iteration and after
        final estimate of posteriors;
        number_of_iters is actual number of iterations that was done.
    """
    H, W, N = X.shape
    if F is None:
        F = np.random.randint(0, 255, (H, w))
    if B is None:
        B = np.random.randint(0, 255, (H, W))
    if a is None:
        a = np.ones(W - w + 1)
        a /= np.sum(a)
    if s is None:
        s = np.random.rand()*pow(64,2)
    # your code here
    LL = [-100000]
    for i in range(max_iter):
        q = run_e_step(X,F,B,s,a)
        F,B,s,a = run_m_step(X,q,w)
        LL.append(calculate_lower_bound(X,F,B,s,a,q))
        if LL[-1]-LL[-2] < tolerance :
            break
    LL = np.array(LL)
    return F,B,s,a,LL
        
    
```


```python
# run this cell to test your implementation
res = run_EM(tX, tw, max_iter=10)
LL = res[-1]
assert np.alltrue(LL[1:] - LL[:-1] > 0)
print("OK")
```

    OK
    

### Who is the prankster?

To speed up the computation, we will perform 5 iterations over small subset of images and then gradually increase the subset.

If everything is implemented correctly, you will recognize the prankster (remember he is the one from [DeepBayes team](http://deepbayes.ru/#speakers)).

Run EM-algorithm:


```python
def show(F, i=1, n=1):
    """
    shows face F at subplot i out of n
    """
    plt.subplot(1, n, i)
    plt.imshow(F, cmap="Greys_r")
    plt.axis("off")
```


```python
F, B, s, a = [None] * 4
LL = []
lens = [50, 100, 300, 500, 1000]
iters = [5, 1, 1, 1, 1]
plt.figure(figsize=(20, 5))
for i, (l, it) in enumerate(zip(lens, iters)):
    F, B, s, a, _ = run_EM(X[:, :, :l], w, F, B, s, a, max_iter=it)
    show(F, i+1, 5)
```


![png](http://ojtdnrpmt.bkt.clouddn.com/blog/180924/KJ8A3g5k9C.png?imageslim)


And this is the background:


```python
show(B)
```


![png](http://ojtdnrpmt.bkt.clouddn.com/blog/180924/E23FKdKe42.png?imageslim)


### Optional part: hard-EM

If you have some time left, you can implement simplified version of EM-algorithm called hard-EM. In hard-EM, instead of finding posterior distribution $p(d_k|X_k, F, B, s, A)$ at E-step, we just remember its argmax $\tilde d_k$ for each image $k$. Thus, the distribution q is replaced with a singular distribution:
$$q(d_k) = \begin{cases} 1, \, if d_k = \tilde d_k \\ 0, \, otherwise\end{cases}$$
This modification simplifies formulas for $\mathcal{L}$ and M-step and speeds their computation up. However, the convergence of hard-EM is usually slow.

If you implement hard-EM, add binary flag hard_EM to the parameters of the following functions:
* calculate_lower_bound
* run_e_step
* run_m_step
* run_EM

After implementation, compare overall computation time for EM and hard-EM till recognizable F.

