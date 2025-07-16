---

title: Easy Reinforcement Learning Notes
date: 2019-09-23 18:16:27
categories: NLP
tags:

- reinforcement learning
- deep learning
mathjax: true
html: true

---

<img src="https://i.mji.rip/2025/07/16/d67b97f7a9ddae88afabc277d249b281.png" width="500"/>

rl study note, minimalist style

*   Q-learning
*   Sarsa
*   Sarsa($\lambda$)
*   DQN
*   Double DQN
*   DQN with Prioritized Experience replay
*   Dueling DQN
*   Policy Gradient

<!--more-->

{% language_switch %}

{% lang_content en %}

Definition
==========

*   agent: An agent, which, in a certain state, takes an action based on some policy to reach the next state
*   s:status, status
*   a: action, action
*   r:reward, reward
*   s,a: Can be called a step. The behavior of the agent can be described as a series of steps. In reinforcement learning, the behavior of the agent can be represented by DP (decision process), where s is the node state in the DP and a is the transition path between states.
*   Q: Q-table, Q(s,a) refers to the value of action a under state s (probability during inference), estimated by the expected reward

Q-learning
==========

*   Random Q-table, initial state, start iteration, $\epsilon$ Greedy
    
*   Taking action a in state s, observing reward r and state $s \prime$
    
*   Key Iteration Formula:
    
    $$
    Q(s,a) = Q(s,a) + \alpha [r + \gamma max_{a \prime}Q(s \prime,a \prime) - Q(s,a)] \\
    $$
    
*   Q's update includes two parts, one of which is naturally the reward (if this step is rewarded), and the other is the time difference, or TD-error, which is the difference between reality and estimation. Here, reality refers to the reward obtained after taking the best action in the new state, while estimation refers to all our rewards except for the actual reward obtained in the final step. For the intermediate steps, the estimated reality reward is obtained using the Q-table values, so the update of the Q-table should be to add (reality - estimation), making the Q-table closer to reality.
    
*   It is noteworthy that only when the last step is rewarded (if there is only one reward at the end), is the reality truly the reward; otherwise, it is still estimated using the Q-table.
    
*   Here, the update of the Q-table is only related to future states and actions. Initially, the updates of all steps except the last one with actual rewards are uncertain (because reality also uses the Q-table for estimation, only the last step is truly real), but after the first iteration, the value update of the last step is determined (knowing how to approach the treasure is known), and unlike LSTM's time series, it does not update from the last time step backwards with BPTT, but updates the transition value of a state (which action is better), and this state may appear at multiple time steps in each iteration (or it is unrelated to the time step). When updating the state adjacent to this state, the estimated reality from the Q-table will be more accurate, and gradually the entire algorithm reaches convergence.
    

Sarsa
=====

*   Q-learning is off-policy because of the gap between reality and estimation, we optimize towards reality, while the actual actions are based on the estimated reality from the Q-table, i.e., asynchronous (off).
    
*   Sarsa is on-policy, differing from Q-learning in algorithmic terms:
    
    *   Q-learning: Select action based on Q-table - Execute action, observe reward and new state - Update Q-table - Update state
    *   Sarsa: Execute action, observe reward and new state - update state - select action on the new state based on the Q-table, and select an action based on the Q-table before the iteration
*   It can be seen that Sarsa has changed the order of steps in the learning algorithm; what effect does this change bring? The effect is to bring the selection of new states and actions forward before the Q-table update, so that when the Q-table is updated, the part of the gap that is realized does not need to be estimated with $max_{a \prime}Q(s \prime,a \prime)$ but is directly estimated using the new state and action:
    
    $$
    Q(s,a) = Q(s,a) + \alpha [r + \gamma Q(s \prime,a \prime) - Q(s,a)] \\
    $$
    
*   That is, both reality and the estimate use the same strategy (determined actions, rather than selecting the maximum value based on given states), and then still use this gap to update.
    
*   In the updating process, Sarsa follows through on its promises (because the reality is adopting new states and actions, the agent will definitely update accordingly), while Q-learning is more optimistic (because the reality is adopting the maximum value under the new state, but the actual action taken may not necessarily be the one with the maximum value, with $\epsilon$ disturbance). Sarsa is more conservative (wishing to do well at every step), while Q-learning is more aggressive (rushing in first and worrying about it later).
    

Sarsa($\lambda$)
================

*   naive version of Sarsa can be seen as Sarsa(0), because the Q-table is updated with every step, i.e., $Q(s \prime,a \prime)$ will only bring the value update of the previous step $Q(s,a)$ . If we consider the updates for all steps, and steps closer to the final reward have higher weights while steps farther from the final reward have lower weights, then the hyperparameter adjusting this weight is $\lambda$ , with a weight of 0 meaning not considering previous steps, i.e., naive Sarsa.
    
*   The specific implementation involves adding a trace matrix E (with each element corresponding to a step (s, a)) to save the weights of all steps in the current path. The closer to the final reward, the higher the weight. Therefore, with each step taken, the corresponding element in the matrix is incremented by 1, and then the corresponding trace matrix value is used as a weight to multiply onto the reward to update the Q-table (here, all steps and the entire E matrix are multiplied together). Afterward, the matrix values are decayed by multiplying with the update decay factor $\gamma$ and the weight hyperparameter $\lambda$ . Clearly, when $\gamma$ is 1, all states receive the same update (since all steps and the entire E matrix are multiplied together); when $\gamma$ is 0, only the executed step element is incremented by 1, and the entire E matrix is set to 0, so only the executed step receives an update, i.e., naive Sarsa. By expanding the iteration of the E matrix, one can obtain an expansion similar to naive Sarsa, except that an E(s,a) is added during the decay to record the distance of a step from the reward.
    
    $$
    \delta = r + \gamma Q(s \prime,a \prime) - Q(s,a) \\
    E(s,a) = E(s,a) + 1 \\
    Q = Q + \alpha \delta E \\
    E = \gamma \lambda E \\
    update  \ \ s,a \\
    $$
    
*   This E-matrix is the eligibility trace. For a certain step, if it is executed, its value is slightly increased, and then it decays slowly until it is executed again. If it is executed multiple times in the short term, its value will rise too high, at which point a threshold can be set to limit the increase in eligibility. This value can be interpreted as the contribution of this step to finding the final reward in this iteration.
    

DQN
===

*   Q-learning + deep neural network
*   Neural networks are used to parameterize the update process of the Q-table, inputting a state vector to output the value of all actions. The training of the neural network replaces the simple iterative update. This can solve the dimensionality disaster problem caused by an excessive number of states.
*   It is not easy to train by simply replacing this way, DQN introduces two techniques
    *   experience replay
    *   fix q
*   Examine the input, output, and loss of the neural network
    *   Two neural networks are involved, one participating in training (evaluation network), and the other replicating the parameters trained by the first network to generate ground truth, i.e., the target network
    *   The network input for training is a state represented as an s-dimensional feature vector, and the output is an a-dimensional value vector obtained from each action, with the loss being the mean squared error between this vector and the a-dimensional ground truth vector.
*   DQN also includes a memory matrix, with dimensions \[number of memory items, $2 * s + 2$ \], where each item contains the reward, action, old state, and new state, i.e., all the information of a single step. The memory only saves the last few memory items executed steps.
*   Where do the input and output of the neural network come from? In fact, it samples a batch of data from memory, inputs the old state to the evaluation network to obtain the model output, inputs the new state to the target network to obtain the ground truth, and then calculates the loss.
*   Every so often, the target network replicates the evaluation network; during this period, the target network remains unchanged, i.e., fix q
*   Experience replay
*   It is noteworthy that we do not directly use the value vector output by the evaluation network to calculate the loss, as no action has been taken yet. Therefore, we first take actions based on the value vector output by the network, update the value corresponding to these actions in the value vector with rewards, and then use this updated value vector to participate in the loss calculation.

Double DQN
==========

*   Double DQN solves the DQN overestimate problem
*   The difference from DQN is that the evaluation network not only accepts an old state to produce an output $q_{eval}$ , but also accepts a new state to produce an output $q_{eval4next}$ , and then selects an action to update $q_{eval}$ based on $q_{eval4next}$ , rather than selecting an action to update based on $q_{eval}$ itself.

DQN with Prioritized Experience replay
======================================

*   Priorly, it was to randomly sample a segment of memory from the memory bank, which would lead to an excessive number of training steps for the model (random sampling is difficult to ensure reaching the final reward)
*   Memory should naturally be allocated priority, with higher priority having a greater sampling probability
*   Priority can be measured by the TD-error value, with larger errors naturally requiring more sampling and optimization
*   Given the priority distribution, if one wishes to obtain a sampling sequence that satisfies this priority distribution, then this is a Monte Carlo problem, which can be solved using MCMC, importance sampling, or the SumTree mentioned in the paper

Dueling DQN
===========

*   Dueling DQN refines the input-output relationship of the internal network in the original DQN:
    
    $$
    Q(s,a;\theta,\alpha,\beta) = Value(s;\theta,\beta) + Advantage(s,a;\theta,\alpha)
    $$
    
*   The value brought by the state and the value (advantage) brought by actions on that state have been split
    
*   Why consider states separately? Because some states are unrelated to actions, and no action will bring about a change in value regardless of what action is taken
    

Policy Gradient
===============

*   Translated Text: The previously introduced methods are all based on value reinforcement learning, i.e., querying the value brought by each action at a certain state and selecting the action with the highest value to execute
    
*   Policy gradient（策略梯度）directly outputs actions from the policy network input state, skipping the step of calculating value, and is a method based on the policy
    
*   Strategy networks do not calculate a certain loss for backpropagation but propagate back based on rewards, and the updated algorithm is as follows:
    
    $$
    \theta = \theta + \alpha \nabla _{\theta} \log \pi _{\theta} (s_t, a_t) v_t \\
    $$
    
*   It should be noted the following key points:
    
    *   The PG algorithm neural network here is responsible for receiving the state as input, the reward as the gradient adjustment value, the actual action executed as the gold label, and outputs an action vector. The entire network itself is a policy $\pi$ (input state, output action probability distribution)
    *   Afterward, the entire model still selects actions based on action probabilities (network outputs) using a policy (network), transitions states, and observes the environment to receive rewards, just like DQN
    *   The first question is, where does this strategy gradient come from? As can be seen, the gradient is still the derivative of the loss with respect to the parameters, $- \nabla _{\theta} \log \pi _{\theta} (s_t, a_t) v_t$ . Where does the loss come from? It is actually the probability of executing an action multiplied by the reward, which can be considered as the cross-entropy between the action probability distribution and the one-hot vector of the executed action, multiplied by the reward (cross-entropy itself has a similar lookup effect). Let's look at its original meaning: the probability of executing an action multiplied by the reward. The network itself is just a strategy. What kind of strategy is a good strategy? If the good reward is obtained after executing the action selected by this strategy, then this is a good strategy network. That is, a strategy network that can select good actions is a good network. It is obvious that the objective function of the network should be the probability of executing an action (the actions recognized by the network) multiplied by the reward (the good actions recognized by the reward).
    *   The second question is, for an environment where rewards can be given at any time, the agent can take a step and update the strategy network once according to the reward. But what about the situation where only the last step can yield a reward? In fact, we adopt round updates, whether it is an environment where rewards can be given at every step or only at the last step, we record all the steps within a round (s, a, r) (except for the last step, the rewards for the other steps are 0 or -1). Then, replace the reward for each step with the cumulative reward and multiply it by a decay coefficient to make the reward decay with the episode steps. Note that unlike discrete value-based learning, policy-based PG can still calculate an action probability to compute the loss even for steps without rewards, so even in the absence of rewards (or negative rewards), it can make the strategy network optimize to reduce the probability of bad actions.

Actor-Critic
============

*   Clearly, strategy-based methods abandon values to gain an advantage (continuous actions), but also bring disadvantages (after each step, there is no value left to estimate immediate rewards, and only round updates can be performed). A natural thought, then, is to combine these two points: choose a strategy-based network as the actor to train the strategy; choose a value-based network as the critic to provide values to the strategy network and estimate immediate rewards.



{% endlang_content %}

{% lang_content zh %}

# 定义

- agent：智能体，智能体处在某一状态下，依据某种策略(policy)，采取一个动作，到达下一个状态
- s:status，状态
- a:action，动作
- r:reward，奖励
- s,a：可以称之为一步。智能体的行为可以描述为一系列步。强化学习里agent的行为可以用DP（决策过程）表示，s是DP里的节点状态，a是状态之间的转移路径
- Q：Q表，Q(s,a)即状态s下执行动作a的价值（infer时的可能性），用奖励期望来估计

# Q-learning

- 随机Q表，初始化状态，开始迭代，$\epsilon$贪心
- 在状态s采取动作a，观察到奖励r和状态$s \prime$
- 关键迭代公式：
  
  $$
  Q(s,a) = Q(s,a) + \alpha [r + \gamma max_{a \prime}Q(s \prime,a \prime) - Q(s,a)] \\
  $$
- Q的更新包括两部分，一部分自然是奖励（假如这一步得到奖励的话），另一部分是时间差值，即TD-error，是现实和估计之间的差值。这里的现实是在新状态采取了最好动作后得到的奖励，这里的估计是指我们所有的奖励，除了最终步得到的真实奖励，其余中间步都是用Q表值来估计现实奖励，因此Q表的更新应该是加上（现实-估计），即让Q表更加贴近现实。
- 值得注意的是，只有最后一步得到奖励时（假如我们只有终点一个奖励），现实才真的是现实的奖励，否则还是用Q表估计的。
- 这里Q表的更新只与未来的状态和动作有关，在最开始，应该除了真正有奖励的最后一步，其余步骤的更新都是不确定的（因为现实也是用Q表估计的，只有最后一步现实才是现实），但第一次迭代之后最后一步的价值更新是确定的（在宝藏边上还是知道怎么走向宝藏），且与LSTM那种时间序列不同，它不是从最后一个时间步往前BPTT，而是更新了一个状态的转移价值（取哪个动作好），这个状态可能出现在每一次迭代的多个时间步上（或者说和时间步无关），接下来与该状态相邻的状态更新时，用Q表估计的现实就会准确一些，慢慢的整个算法达到收敛。

# Sarsa

- Q-learning是off-policy的，因为存在着现实和估计的差距，我们朝着现实优化，而实际采取的是根据Q表估计的现实，即异步的(off)。
- Sarsa是on-policy的，与Q-learning在算法上的区别：
  - Q-learning:根据Q表选动作-执行动作观察到奖励和新状态-更新Q表-更新状态
  - Sarsa：执行动作观察到奖励和新状态-更新状态-根据Q表在新状态上选动作，在迭代之前先来一次根据Q表选动作
- 可以看到Sarsa更改了学习算法的步骤顺序，这种更改带来了什么效果？效果就是将新状态和新动作的选取提前到Q表更新之前，这样Q表更新时，差距里现实的部分不用$max_{a \prime}Q(s \prime,a \prime)$来估计，而直接用新状态和新动作：
  
  $$
  Q(s,a) = Q(s,a) + \alpha [r + \gamma Q(s \prime,a \prime) - Q(s,a)] \\
  $$
- 也就是说现实和估计采用的是同一策略（确定的动作，而不是给定状态选最大价值），然后依然使用这个差距来更新。
- 在更新的过程中，sarsa说到做到（因为现实采用的就是新状态和新动作，agent一定会按照这样更新），而Q-learning则比较乐观（因为现实采用的是新状态下的最大价值，但实际走不一定会采取最大价值的行动，有$\epsilon$的扰动）。Sarsa更为保守（每一步都想走好），而Q-learning更为激进（先rush到再说）。

# Sarsa($\lambda$)

- naive版本的Sarsa可以看成是Sarsa(0)，因为每走一步就更新了Q表，即$Q(s \prime,a \prime)$只会带来上一步$Q(s,a)$的价值更新。假如我们考虑对所有步的更新，且离最终奖励近的步权重大，离最终奖励远的步权重小，那么调整这个权重的超参就是$\lambda$，权重为0就是不考虑之前的步数，即naive Sarsa。
- 具体实现是添加一个trace矩阵E（矩阵中每一个元素对应一步(s,a)）保存所有步在该次路径中的权重，离最终奖励越近，权重越大，因此每走一步，执行的那一步元素对应矩阵值加1，然后用对应的trace矩阵值作为权重乘到奖励上来更新Q表（这里是所有的步和整个E矩阵乘起来），之后矩阵值会衰减一下，乘以更新衰减因子$\gamma$和权重超参$\lambda$。显然$\gamma$为1时，就是所有的状态得到了一样的更新（因为是所有步和整个E矩阵相乘）；当$\gamma$为0时，除了执行步元素加了1，然后整个E矩阵都置0，因此只有执行步得到了更新，即naive Sarsa。将E矩阵的迭代展开，就可以得到与naive sarsa一样的展开，只不过衰减的时候加了一个E(s,a)来记录某一步距离奖励的距离。
  
  $$
  \delta = r + \gamma Q(s \prime,a \prime) - Q(s,a) \\
E(s,a) = E(s,a) + 1 \\
Q = Q + \alpha \delta E \\
E = \gamma \lambda E \\
update  \ \ s,a \\
  $$
- 这个E矩阵就是eligibility trace。对于某一步，如果被执行了，就增加一点值，之后慢慢衰减，直到又被执行。假如短期内被执行多次，就会上升到过高值，这时可以设置阈值来限制eligibility的增加。这个值可以解释为该步在该次迭代中对于找到最终奖励的贡献程度。

# DQN

- Q-learning +  deep neural network
- 神经网络用于把Q-table的更新过程参数化，输入一个状态向量，神经网络输出所有动作的价值，用神经网络的训练来替代简单的迭代更新。这样可以解决状态数过多导致的维度灾难问题。
- 直接这么替换不容易训练，DQN引入两个技巧
  - experience replay
  - fix q
- 先看神经网络的输入输出和损失
  - 有两个神经网络，一个网络参与训练（评价网络），一个网络只复制另一个网络训练得到的参数，用来生成ground truth，即目标网络
  - 参与训练的网络输入是状态，表示为s维特征向量，输出是各个动作获得的a维价值向量，损失是这个向量和ground truth的a维向量之间的均方误差。
- DQN还包含一个记忆矩阵，维度是[记忆条数，$ 2 * s + 2$ ]，每一条包含奖励、动作，老状态和新状态，即一步的所有信息。记忆只保存最近记忆条数次执行步。
- 之后神经网络的输入输出从哪里来？其实是从记忆中采样一个Batch的数据，将老状态输入评价网络得到model output，将新状态输入目标网络得到ground truth，之后计算损失。
- 每隔一段时间目标网络才会复制评价网络，在此期间目标网络都是固定不变的，即fix q
- 而输入是从最近的记忆当中抽取的，即experience replay
- 值得注意的是我们并不直接把评价网络输出价值向量用于计算损失，因为还没有采取动作。因此我们要先根据网络输出的价值向量采取动作，将价值向量里这些动作对应的价值用奖励更新，之后这个更新过的价值向量再参与损失计算。

# Double DQN

- Double DQN解决DQN over estimate的问题
- 与DQN不同之处在于，评价网络不仅接受老状态产生一个输出$q_{eval}$，还接受新状态产生一个输出$q_{eval4next}$，之后依据$q_{eval4next}$选取动作更新$q_{eval}$，而不是根据$q_{eval}$本身选取动作来更新。

# DQN with Prioritized Experience replay

- 之前是从记忆库中随机采样一段记忆，这个随机采样会导致模型训练步数过多（随机很难保证到达最终奖励）
- 那么自然而然想到记忆应该分配优先级，优先级高的采样概率大
- 优先级可以用TD-error值来衡量，error大的自然要多采样多优化
- 已知优先级分布，希望得到满足该优先级分布的一个采样序列，那么这就是一个蒙特卡洛问题了，可以用MCMC，可以用Importance sampling，也可以用论文里提到的SumTree

# Dueling DQN

- Dueling DQN细化了原始DQN的内部网络输入输出关系：
  
  $$
  Q(s,a;\theta,\alpha,\beta) = Value(s;\theta,\beta) + Advantage(s,a;\theta,\alpha)
  $$
- 即拆分成了状态带来的价值和动作在该状态上带来的价值（advantage）
- 为什么要单独考虑状态？因为有些状态是与动作无关的，无论采取什么动作都不会带来价值的改变

# Policy Gradient

- 之前介绍的都是基于值的强化学习方法，即在某状态查询各个动作带来的价值，选择最大价值动作执行
- policy gradient（策略梯度）通过策略网络输入状态，直接输出动作，跳过了计算价值的步骤，是基于策略的方法
- 策略网络并不计算某种损失进行反向传播，而是依据奖励来反向传播，更新的算法如下：
  
  $$
  \theta = \theta + \alpha \nabla _{\theta} \log \pi _{\theta} (s_t, a_t) v_t \\
  $$
- 需要注意以下几个关键点：
  - 这里的PG算法神经网络只负责接收状态作为输入，奖励作为梯度调整值，实际执行的动作作为gold label，输出一个action vector，整个网络本身就是一个策略$\pi$（输入状态，输出动作概率分布）
  - 之后整个模型依然像DQN那样，借助策略（网络）按动作概率（网络输出）选择动作，转移状态，观察环境得到奖励
  - 那么第一个问题，这个策略梯度是怎么来的？可以看到梯度依然是损失对参数求导的形式，$- \nabla _{\theta} \log \pi _{\theta} (s_t, a_t) v_t$，哪来的损失？实际上是执行动作的概率乘以奖励，可以看作是动作概率分布和执行动作one-hot向量之间的交叉熵（交叉熵本来就有类似look up的效果）乘以奖励。我们就看其本来的含义：执行动作的概率乘以奖励，网络本身只是一个策略，什么样的策略是好策略？假如通过这个策略选出来的动作执行之后得到了好的奖励，那么这是一个好的策略网络，也就是能选出好动作的策略网络是好网络，显然网络的目标函数就应该是执行动作的概率（网络认可的动作）乘以奖励（奖励认可的好动作）。
  - 第二个问题，对于随时能给出奖励的环境，agent可以走一步，根据奖励更新一次策略网络，那对于那些只有最后一步能够得到奖励的该咋办？事实上我们采取的是回合更新，无论是每一步都能给出奖励的环境还是只有最后一步有奖励的环境，我们都将一个回合内的所有步(s,a,r)都记录下来（除了最后一步，其余步的奖励都是0或者-1），之后每一步的奖励替换成累加奖励，并乘以一个衰减系数使得奖励随episode steps衰减。注意不同于离散的基于值的学习，基于策略的PG在没有得到奖励的那些步也能算出一个动作概率来计算损失，因此即便是无奖励（或者负奖励），也能使得策略网络优化去让不好的动作概率降低。

# Actor Critic

- 显然基于策略的方法抛弃了值，获得了优势（连续动作），也带来了劣势（每执行一步之后没得值可以用来估计即时奖励，只能进行回合更新），那么一个自然而然的想法就是结合这两点，选择一个基于策略的网络作为actor，训练出策略；选择一个基于值的网络作为critic，用来给策略网络提供值，估计即时奖励。

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