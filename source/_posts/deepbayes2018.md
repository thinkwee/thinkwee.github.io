---
title: deepbayes2018
date: 2018-09-22 10:26:48
tags:
---

Deep-Bayes 2018 Summer Camp
习题
<!--more-->

# Deep<span style="color:green">|</span>Bayes summer school. Practical session on EM algorithm

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
    


![png](task_em_files/task_em_9_1.png)


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


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-8-7e147f60fb2e> in <module>()
          4        [-6141.69812064, -8541.69812064]])
          5 actual = calculate_log_probability(tX, tF, tB, ts)
    ----> 6 assert np.allclose(actual, expected)
          7 print("OK")
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in allclose(a, b, rtol, atol, equal_nan)
       2254 
       2255     """
    -> 2256     res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
       2257     return bool(res)
       2258 
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in isclose(a, b, rtol, atol, equal_nan)
       2330     y = array(y, dtype=dt, copy=False, subok=True)
       2331 
    -> 2332     xfin = isfinite(x)
       2333     yfin = isfinite(y)
       2334     if all(xfin) and all(yfin):
    

    TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


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
```


```python
# run this cell to test your implementation
expected = -12761.1875
actual = calculate_lower_bound(tX, tF, tB, ts, ta, tq)
assert np.allclose(actual, expected)
print("OK")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-10-eb49d6900e27> in <module>()
          2 expected = -12761.1875
          3 actual = calculate_lower_bound(tX, tF, tB, ts, ta, tq)
    ----> 4 assert np.allclose(actual, expected)
          5 print("OK")
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in allclose(a, b, rtol, atol, equal_nan)
       2254 
       2255     """
    -> 2256     res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
       2257     return bool(res)
       2258 
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in isclose(a, b, rtol, atol, equal_nan)
       2330     y = array(y, dtype=dt, copy=False, subok=True)
       2331 
    -> 2332     xfin = isfinite(x)
       2333     yfin = isfinite(y)
       2334     if all(xfin) and all(yfin):
    

    TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


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


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-12-98d2b4c2edac> in <module>()
          4                    [ 0.,  0.]])
          5 actual = run_e_step(tX, tF, tB, ts, ta)
    ----> 6 assert np.allclose(actual, expected)
          7 print("OK")
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in allclose(a, b, rtol, atol, equal_nan)
       2254 
       2255     """
    -> 2256     res = all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
       2257     return bool(res)
       2258 
    

    ~\Anaconda3\lib\site-packages\numpy\core\numeric.py in isclose(a, b, rtol, atol, equal_nan)
       2330     y = array(y, dtype=dt, copy=False, subok=True)
       2331 
    -> 2332     xfin = isfinite(x)
       2333     yfin = isfinite(y)
       2334     if all(xfin) and all(yfin):
    

    TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


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


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-14-e44e2bfe2fd1> in <module>()
          7  np.array([ 0.13888889,  0.33333333,  0.52777778])]
          8 actual = run_m_step(tX, tq, tw)
    ----> 9 for a, e in zip(actual, expected):
         10     assert np.allclose(a, e)
         11 print("OK")
    

    TypeError: zip argument #1 must support iteration


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
    
```


```python
# run this cell to test your implementation
res = run_EM(tX, tw, max_iter=3)
LL = res[-1]
assert np.alltrue(LL[1:] - LL[:-1] > 0)
print("OK")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-6b1860dc5143> in <module>()
          1 # run this cell to test your implementation
          2 res = run_EM(tX, tw, max_iter=3)
    ----> 3 LL = res[-1]
          4 assert np.alltrue(LL[1:] - LL[:-1] > 0)
          5 print("OK")
    

    TypeError: 'NoneType' object is not subscriptable


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


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-18-5c3ec9450804> in <module>()
          5 plt.figure(figsize=(20, 5))
          6 for i, (l, it) in enumerate(zip(lens, iters)):
    ----> 7     F, B, s, a, _ = run_EM(X[:, :, :l], w, F, B, s, a, max_iter=it)
          8     show(F, i+1, 5)
    

    TypeError: 'NoneType' object is not iterable



    <Figure size 1440x360 with 0 Axes>


And this is the background:


```python
show(B)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-19-ba7381968102> in <module>()
    ----> 1 show(B)
    

    <ipython-input-17-1c6656dd6e56> in show(F, i, n)
          4     """
          5     plt.subplot(1, n, i)
    ----> 6     plt.imshow(F, cmap="Greys_r")
          7     plt.axis("off")
    

    ~\Anaconda3\lib\site-packages\matplotlib\pyplot.py in imshow(X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, shape, filternorm, filterrad, imlim, resample, url, hold, data, **kwargs)
       3203                         filternorm=filternorm, filterrad=filterrad,
       3204                         imlim=imlim, resample=resample, url=url, data=data,
    -> 3205                         **kwargs)
       3206     finally:
       3207         ax._hold = washold
    

    ~\Anaconda3\lib\site-packages\matplotlib\__init__.py in inner(ax, *args, **kwargs)
       1853                         "the Matplotlib list!)" % (label_namer, func.__name__),
       1854                         RuntimeWarning, stacklevel=2)
    -> 1855             return func(ax, *args, **kwargs)
       1856 
       1857         inner.__doc__ = _add_data_doc(inner.__doc__,
    

    ~\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py in imshow(self, X, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent, shape, filternorm, filterrad, imlim, resample, url, **kwargs)
       5485                               resample=resample, **kwargs)
       5486 
    -> 5487         im.set_data(X)
       5488         im.set_alpha(alpha)
       5489         if im.get_clip_path() is None:
    

    ~\Anaconda3\lib\site-packages\matplotlib\image.py in set_data(self, A)
        647         if (self._A.dtype != np.uint8 and
        648                 not np.can_cast(self._A.dtype, float, "same_kind")):
    --> 649             raise TypeError("Image data cannot be converted to float")
        650 
        651         if not (self._A.ndim == 2
    

    TypeError: Image data cannot be converted to float



![png](task_em_files/task_em_35_1.png)


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
