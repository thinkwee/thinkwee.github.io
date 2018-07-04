---
title: '机器学习入门:神经网络基础'
date: 2017-02-10 12:20:17
categories: 机器学习
tags:
  - code
  - machinelearning
mathjax: true
photos: http://ojtdnrpmt.bkt.clouddn.com/blog/20170210/225753048.png
html: true
---

***
以简单的点集分类和泰坦尼克号作为例子，学习BP神经网络


<!--more-->

# <font size=5 >模型</font>

## <font size=4 >模型结构</font>

- 就以文章头部那张图片为例，机器学习(监督学习部分)是找到一个模型，通过已知的条件和结果，调整模型中各个参数的值，最后得到一个模型，对于某一类问题，将条件输入模型，可以输出(预测)较为正确的结果，神经网络的模型结构就如图所示，左边的input就是条件，右边的output就是结果
- 模型至少有2个层，输入层与输出层中间可能有多个层，每一层包含若干个节点，每一个节点是一个函数，函数的输入是上一层所有节点的输出通过路径加权再减去这个节点本身的阈值，输出再作为下一层每一个个节点输入的一部分。如果某个节点有输出，则叫做被激活，节点所代表的函数就是激活函数，路径上的加权系数以及阈值就是网络参数。当一个神经网络构建好且学习好后，输入一组条件，每一层的节点依次被上一层节点激活输出，直到最后一层的节点输出结果。机器学习学习的成果就是正确的网络参数，一个神经网络的数据模型也可以看成一组网络参数。在本例中我们研究分类问题，最后的输出节点输出这个条件下属于某一类的概率。

## <font size=4 >网络参数</font>

- 节点的输出可以写成
  $$
  y=f(\sum_{i=1}^nw_ix_i-\theta )
  $$
  其中n是连接到这个节点的路径数，$w_i$是路径上的权值，$\theta$是阈值,$f$是激活函数。阈值的理解:可以看成上一层有一个哑节点，它的输出恒为-1，这个哑节点连接过来的权值$w_{n+1}$就是阈值。阈值和权值统称为网络参数。

- 网络参数的学习其实就是根据这次训练的结果和理想结果对比，将参数增加一个$\Delta$值，进行调整。
  $$
  \Delta w_i=\eta (y-y_0)x_i
  $$
  上式即最简单的感知机模型(无隐层)的权重调整函数，其中$\eta$称为学习率。
  
## <font size=4 >激活函数</font>

- 激活函数有许多种，比如最简单的阶跃函数(加权输入大于阈值则输出1否则无输出)，或者Sigmoid函数、tanh函数。输出节点一般用Softmax函数。

- Sigmoid函数有一种很好的性质:$f'(x)=f(x)(1-f(x))$

  tanh函数也有一种很好的性质:$tanh'(x)=1-tanh^2(x)$

  可以看到这两种函数都可以通过本身计算出导函数，而在多层前馈神经网络中，我们需要利用负梯度来调整网络参数，计算负梯度需要利用导数，这样的性质能方便推导公式。


# <font size=5 >损失函数</font>

## <font size=4 >损失评估</font>

- 每一次训练后，我们需要知道这一次训练是否降低了与目标之间的误差，这个误差的量化就利用损失函数
- 损失评估可以防止BP神经网络过拟合，例如早停策略:将数据分成训练集和验证集，训练集用来计算梯度，更新网络参数，验证机用来评估误差，若训练集误差降低而验证集误差升高，则停止训练以防止过拟合，同时返回具有最小验证集误差的网络参数。

## <font size=4 >交叉熵损失</font>

- 交叉熵损失的计算公式
  $$
  L_i=-log(\frac {e^f_{yi}} {\sum_je^{f_j}})
  $$
  f是最后一层的节点的输出，可以看到括号里实际上是将最后一层输出的每一类的概率指数函数归一化，再求log。假如某一行训练元素，正确分类到a类，结果输出a类的概率低，则括号里的值接近0,$L_i$就会趋近无穷，即损失太多，反之，括号里的值接近1，$L_i$趋近0，即几乎无损失。

  一般还会加入正则化项
  $$
  L=\frac1N \sum_iL_I+ \frac12 \lambda \sum_k\sum_lW_{k,l}^2
  $$








## <font size=4 >均方误差</font>

- 均方误差的衡量就很简单暴力
  $$
  E_k=\frac12\sum_{j=1}^l(y_0j^k-y_j^k)^2
  $$
  $\frac 12$是为了方便做负梯度计算


# <font size=5 >BP算法</font>

## <font size=4 >负梯度</font>

- 梯度下降法或最速下降法是求解无约束最优化问题的一种常见方法，是迭代算法，每一步需要求解目标函数的梯度向量。

- BP算法以目标的负梯度方向对网络参数进行调整，目标函数即损失函数，即

 $$
 \Delta w=-\eta \frac {\partial L}{\partial w}
 $$

  这样求出来的意义是，调整值是损失函数的负梯度乘学习率，即让损失减小的方向调整

- 对于均方误差和交叉熵损失两种损失函数，求出来的负梯度不同，但都可以通过他们的性质优化，以交叉熵损失为例，求出来的负梯度为：

  ![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170213/194245109.jpg)

  其中$W_1,b_1$是第一层网络参数，$W_2,b_2$是第二层网络参数

## <font size=4 >更新公式</font>

- 对每一次学习先得到预测结果，再得出损失函数，从而计算出负梯度，乘上学习率并正则化，就是最终这一次学习到的网络参数调整值


# <font size=5 >代码实现</font>

## <font size=4 >初始化与可视化</font>

```python
class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x:predict(model,x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()
```



## <font size=4 >计算损失</font>

- 其中probs即指数归一化后的各类别概率
- probs[range(num_examples), y]即每一个预测结果中实际正确类别的预测概率，用log还原指数并累加即交叉熵损失data_loss,之后再正则化

```python
# Helper function to evaluate the total loss on the dataset
def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss
```



## <font size=4 >逆向误差传输</font>

```python
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))

    return model
```



## <font size=4 >预测</font>

```python
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
```




# <font size=5 >学习率，正则化</font>

## <font size=4 >梯度下降的学习速率</font>

## <font size=4 >正则化</font>

# <font size=5 >点集分类结果</font>
![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170212/223945007.JPG)

# <font size=5 >泰坦尼克号预测调参</font>

# <font size=5 >泰坦尼克号预测结果</font>

# <font size=5 >参考</font>

>[IMPLEMENTING A NEURAL NETWORK FROM SCRATCH IN PYTHON – AN INTRODUCTION](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
>[CS231n Convolutional Neural Networks or Visual Recognition---optimization](http://cs231n.github.io/optimization-1/#gd)
>[CS231n Convolutional Neural Networks or Visual Recognition---neural-networks](http://cs231n.github.io/neural-networks-3/#gradcheck)
>[龙心尘](http://blog.csdn.net/longxinchen_ml/article/details/50521933)




