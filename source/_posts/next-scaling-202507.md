---
title: What is the Next Step for Scaling in the Era of RL for LLM?
date: 2025-07-15 17:16:01
categories: LLM
tags:
  - rl
  - scaling
  - llm
  - agent
refplus: true
mathjax: true
---

[![pV1R7X8.md.webp](https://s21.ax1x.com/2025/07/16/pV1R7X8.md.webp)](https://imgse.com/i/pV1R7X8)

When the redundant designs we added in the pre-LLM era have been deleted by the bitter lesson, we are ready to scale up. In the era of RL for LLM, what should be the next scaling up?

<!--more-->

{% language_switch %}

{% lang_content en %}
# Towards General Reasoning
- I've been following recent developments on Agents and the line of work on Incentive Training for Reasoning sparked by DeepSeek. A few days ago, the release of Kimi K2 caught my attention, particularly its section on general reasoning:

{% blockquote [[Kimi K2](https://moonshotai.github.io/Kimi-K2/)] %}
Going beyond verifiable rewards, our general RL system uses a self-judging mechanism where the model acts as its own critic, providing scalable, rubric-based feedback for non-verifiable tasks.
{% endblockquote %}

- It was at this moment that I realized the dots from the recent papers I had been reading started to connect—just as Steve Jobs once said, “You can't connect the dots looking forward; you can only connect them looking backwards. So you have to trust that the dots will somehow connect in your future.”
- This post organizes those thoughts, as illustrated in the diagram above: the scaling of large models has never stopped—we have continued to scale up the knowledge and capabilities learned from next token prediction, in various ways and from different angles.

# Next Token Prediction
<img src="https://i.mji.rip/2025/07/16/5c71b7331c275cbc784bd315744eb000.webp" width="800"/>

- At the top, *Next token prediction is all you need*. A general consensus seems to be that knowledge is primarily acquired during pretraining. Post-training mainly serves to steer/incentivize capabilities. Continuing pretraining remains the preferred way to encode knowledge into the model. Trying to add/edit/delete knowledge during post-training is extremely challenging, and research on knowledge editing has remained relatively toy-level.
{% blockquote [[Life Cycle of LLM Knowledge](https://llmknowledgelifecycle.github.io/AAAI2025_Tutorial_LLMKnowledge/)] %}
Knowledge Is Stored Messily in LLMs
{% endblockquote %}

- Once we clarify the roles of pretraining and post-training, we find that both are worth scaling. When the scaling of pretraining hits a bottleneck, OpenAI proposed *inference-time scaling*—which actually relies on post-training to elicit capabilities. In this sense, it is really scaling up post-training.

# Scaling Up Human Feedback with Reward Models
<img src="https://i.mji.rip/2025/07/16/4bdc98917e64f45c74bb2272c5225fae.webp" width="800"/>

- Moving to the lower left is the RLHF path proposed by OpenAI. Besides highlighting the importance of human feedback (beyond objective correctness, there's also the distinction between hard-to-verify good vs bad), I believe it importantly demonstrates how reward models can be used to scale up human feedback. It's unrealistic to have humans annotate massive amounts of model rollouts, but we can use a small amount of high-quality human data to train a reward model, which can then provide feedback at scale for policy model rollouts. This is essentially a tradeoff: sacrificing precision (human-labeled quality) for quantity (reward model scalability). A reward model trained on a small dataset is sufficient to guide a strong policy—because posing a question is easier than solving it (Discriminator-Generator Gap). Once this cold start is done, OpenAI productized it (ChatGPT), enabling continuous data collection and spinning up a data flywheel.

# Verifiable Rewards
<img src="https://i.mji.rip/2025/07/16/983b3a2577db47a72f6f30615fff2430.webp" width="800"/>

- Moving rightward, we see RLVR—DeepSeek’s exploration of end-to-end RL with outcome rewards, undertaken in the broader effort to replicate OpenAI’s o1 model. DeepSeek’s work has offered me three insights:
    - From DeepSeekMath to DeepSeekR1, it showed that when the pretrained checkpoint is strong enough, RL still has enormous potential—not only can it solve math proofs, it can also elicit general reasoning capabilities;
    - Simple, rule-based rewards can directly be used to train language models. This expands the range of possible environments for RL with LLMs, enabling them to serve as general-purpose models across tasks;
    - With GRPO/Rule Reward, DeepSeek removed the need for a critic model and a reward model, making the approach extremely simple. The early media narrative focused on lower costs, but I believe the greater potential lies in higher efficiency scaling.

# LLMs Implicitly Contain Reward Models
<img src="https://i.mji.rip/2025/07/16/51c2cec1e457b7dbdfea261bacda5363.webp" width="800"/>

- Below RLHF, we see the realization that LLM-based policy models inherently contain reward models:
    - The intuition behind DPO is straightforward: if I follow a pipeline of [train reward model on high-quality human preference data → train policy model using reward model], then surely there exists a way to directly train the policy model using high-quality preference data. DPO mathematically proves this. Although it overlooks on-policy vs off-policy distinctions, DPO offers a key insight: in RLHF post-training, an LLM policy model might also be a reward model;
    - There's a paper not included in the diagram—PRIME: Process Reinforcement through Implicit Rewards by Tsinghua—which extends the implicit reward concept from DPO into outcome-reward tasks, extracting process reward signals. While PRIME is not central to this post, it’s very interesting, and a future combining outcome + process rewards could be promising;
    - Finally, we have *Generalist Reward Models: Endogenous Reward*. Next token prediction on massive human-approved corpora is itself a way of learning a reward function. This continues DPO’s idea: LLMs are both policy and reward models, not just in post-training, but even during pretraining.

# LLMs Implicitly Contain Verifier Models
<img src="https://i.mji.rip/2025/07/16/41910a45eba55627de00243f82207fdc.webp" width="800"/>

- Further right, mirroring the idea of Secret Reward Model under RLHF, is the RLVR counterpart: in RLVR, the LLM itself serves as a verifier model. This includes our recent work NOVER and several related papers. The motivation is straightforward: RLVR depends on verifiable tasks, but what if we only have freeform ground truth, which can’t be rule-based verified? A natural idea is to use the model's perplexity on those ground truth samples as reward. For incentive training, we can condition on the reasoning trajectory and compute perplexity. The idea is simple, but echoing Secret Reward Model, it supports a broader claim: whether RLHF or RLVR, alignment or incentive training, LLMs themselves are sufficient feedback signal extractors. In RL terms, LLMs are good enough as both policy and reward.

# Scaling Up Reinforcement Learning
<img src="https://i.mji.rip/2025/07/16/3f4d3619d0bf4112fbc0386139871937.webp" width="800"/>

- At the bottom, we see the community’s recent efforts in scaling RL for LLMs:
    - DeepSeek-GRM emphasizes that reward models are worth scaling up;
    - ProRL suggests that RL training itself is worth scaling, with potential to surpass the pretrained ceiling;
    - POLAR argues that reward models should not only be scaled up, but done so with pretraining-level scale.

# Converging to a Single Point
<img src="https://i.mji.rip/2025/07/16/298e8cda183da68a8d7f44739a1cde8e.webp" width="800"/>

- Looking back at the road we've traveled, we see everything converges to one point:
    - Reward models should be scaled up
    - Post-training should be scaled up
    - LLMs themselves are reward models
- --> We only need to scale up the LLM itself! It is both policy and reward, and RL enhances its capabilities in both roles. A stronger reward provides better guidance for policies tackling harder problems. The simplest form accommodates the widest range of data and compute. Once tasks and data are well-prepared, everything clicks into gear, spinning faster and leveraging bigger levers. This is the insight conveyed by Kimi K2’s section on general reasoning. As Hyung Won Chung’s slide suggests—less structure, more performance. We began by adding various models and structures, and now we’re removing them one by one:

<img src="https://i.mji.rip/2025/07/16/23a1a1ebeb4308046cb91071b0afa865.webp" width="800"/>

{% blockquote [[Hyung Won Chung's tweet on "Less Structure"](https://x.com/hwchung27/status/180067631291665659)] %}
As a community we love adding structures but a lot less for removing them. We need to do more cleanup.
{% endblockquote %}

# The Next Step
<img src="https://i.mji.rip/2025/07/16/ecb64f8e8fd05a37beaa567f2e1c4fff.webp" width="400"/>

- So what should we scale next? And how?
- One seemingly obvious answer is: from training to inference, to agentic, to multiagent—the next step is scaling up multiagent. (Grok 4 Heavy may currently be the strongest multiagent model/framework; multiagent has also become one of 2025’s hottest AI terms.)

{% blockquote [[Grok 4](https://x.ai/news/grok-4)] %}
We have made further progress on parallel test-time compute, which allows Grok to consider multiple hypotheses at once. We call this model Grok 4 Heavy, and it sets a new standard for performance and reliability.
{% endblockquote %}

- But why? Just because the terms are newer? My view is: we’re not scaling the terms or paradigms themselves. When new paradigms emerge, we’re keen to build around them—but ultimately, the benefits are internalized by the model itself. Scaling laws remain faithful to data and compute, only now they come in different forms.
- When identifying the next direction for scaling, it's not just about agentic or multiagent formats—but where the data comes from to support such scaling. Some scattered thoughts:
    - Synthetic data is inevitable, but I suspect it will show patterns similar to the Discriminator-Generator Gap, enabling tradeoffs to continually produce harder and better data;
    - In RL contexts, data may also appear as environments. A well-defined environment can theoretically generate near-infinite data, meaning that the next phase is not just scaling *amount*, but scaling *difficulty*—an environment that assigns harder goals to stronger agents.
{% blockquote [[Kevin's tweet on "Stop Working on RL"](https://x.com/_kevinlu/status/1942977315031687460)] %}
Why you should stop working on RL research and instead work on product // The technology that unlocked the big scaling shift in AI is the internet, not transformers
{% endblockquote %}
    - I agree with the training → inference → agentic → multiagent scaling roadmap, not just because those terms aim for higher intelligence, but because that path makes LLMs increasingly *useful*. And usefulness brings a key benefit: people are willing to provide more and diverse data to use it.
    - For multiagent, I'm particularly interested in heterogeneous, personal multiagent systems. LLMs have nearly exhausted all knowledge on the internet, which reflects an average of humanity’s collective information. But for individuals, each person’s micro-world continues to generate virtually infinite data and environments. Assigning each person an agent, and allowing society to mirror itself with a society of agents evolving through this data, may be how multiagent scaling becomes possible.

# Citation
If you found the topics in this blog post interesting and would like to cite it, you may use the following BibTeX entry:
```bibtex
@article{next_scaling_202507,
  author = {Wei Liu},
  title = {What is the Next Step for Scaling in the Era of RL for LLM?},
  year = {2025},
  month = {7},
  url = {https://thinkwee.top/2025/07/15/next-scaling-202507/#more},
  note = {Blog post}
}
```

{% endlang_content %}

{% lang_content zh %}

# 迈向通用推理
- 最近一直在关注Agent以及由DeepSeek引发的一系列Incentive Training for Reasoning工作。直到前几天Kimi K2的发布，瞅了一眼其general reasoning的报告部分，如下：

{% blockquote [[Kimi K2](https://moonshotai.github.io/Kimi-K2/)] %}
Going beyond verifiable rewards, our general RL system uses a self-judging mechanism where the model acts as its own critic, providing scalable, rubric-based feedback for non-verifiable tasks.
{% endblockquote %}

- 这时察觉近期阅读的一系列研究串了起来，就像乔布斯说的，“You can't connect the dots looking forward; you can only connect them looking backwards. So you have to trust that the dots will somehow connect in your future.”
- 本文将其整理了一下，就像上图所示：大模型的scaling从未停止，我们一直以不同的方式，从不同角度继续scale up从next token prediction里学习到的知识与能力。

# 下一个token预测
<img src="https://i.mji.rip/2025/07/16/5c71b7331c275cbc784bd315744eb000.webp" width="800"/>

- 最上方，Next token prediction is all you need。应该是大家的一点共识是，知识主要从预训练获取。后训练更多的是对能力的steering/incentivize。将知识训进模型优先的选择还是continue pretraining。如果在后训练去做知识的增删改查会非常困难，知识编辑之类的研究也一直处于比较toy的状态。
{% blockquote [[Life Cycle of LLM Knowledge](https://llmknowledgelifecycle.github.io/AAAI2025_Tutorial_LLMKnowledge/)] %}
Knowledge Is Stored Messily in LLMs
{% endblockquote %}

- 在分清楚预训练和后训练的角色之后，我们发现两部分都值得scale up。预训练的scaling遇到瓶颈之后，OpenAI提出来inference time scaling，但其实inference time scaling依赖后训练对能力的激发，这实际上是在scale up后训练。

# 用奖励模型scale up人类反馈
<img src="https://i.mji.rip/2025/07/16/4bdc98917e64f45c74bb2272c5225fae.webp" width="800"/>

- 往左下，即OpenAI提出的RLHF路线。这条路线除了证明human feedback的重要性（在客观正确之上，还有hard-to-verify好坏的差别）之外，我个人认为还有一点很重要，即用reward model去scale up human feedback。让人类打标海量的model rollouts是不现实的，但是我们可以用少量的高质量人工数据训练一个奖励模型，然后奖励模型可以给海量的policy model rollouts提供反馈。这实际上是一种tradeoff，牺牲精度（人类标注的质量）换取数量（reward model标注的scalability）。用少量数据训练的reward model足以指导强大的policy，因为提出问题总比解决问题简单（Discriminator-Generator Gap）。当这个冷启动完成之后，OpenAI将其产品化（ChatGPT），从而可以不断地收集新的用户数据，完成数据飞轮。

# 可验证奖励
<img src="https://i.mji.rip/2025/07/16/983b3a2577db47a72f6f30615fff2430.webp" width="800"/>

- 往右，是RLVR，即DeepSeek在全员复现OpenAI o1的背景下，探索出的一条路径：借助端到端的RL with outcome reward，可以将推理能力内生化到模型当中。DeepSeek给我带来的启发有三：
    - 从DeepSeekMath到DeepSeekR1，证明了pretrained checkpoint足够好的情况下，RL的挖掘空间还很大，不仅仅是可以做数学证明，还能激发出通用推理能力；
    - 基于规则的简单奖励也可以直接用来训练语言模型，这一下拓宽了RL for LLM环境的范围，使得LLM作为一个通用模型能够借助强化学习胜任各种任务；
    - 借助GRPO/Rule Reward，DeepSeek一下子拿掉了critic model和reward model，变得极度简洁。极度简洁带来的好处早期媒体都报道为降低成本，但我想更大的潜力在于能够以更高的效率scale up。

# LLM隐含了奖励模型
<img src="https://i.mji.rip/2025/07/16/51c2cec1e457b7dbdfea261bacda5363.webp" width="800"/>

- RLHF往下，是人们发现基于LLM的policy model本身就隐含了奖励模型。
    - DPO的直觉非常简单，我使用[高质量人类标注偏好数据训练奖励模型-->用奖励模型训练策略模型]的pipeline，那么肯定能用一种方式将这个pipeline直接表达为用高质量人类标注偏好数据训练policy model。DPO数学上证明了这一点。虽然其忽视了on-policy/off-policy的区别，但DPO提供了一种思想，在RLHF的后训练中，一个LLM policy model可能本身也是一个reward model；
    - 中间有一篇我没有在图中标出的paper，即清华的PRIME: Process Reinforcement through Implicit Rewards。基于DPO提出的implicit reward思想，我们可以将其在outcome reward任务中训练，然后提取出process reward signals。虽然PRIME不在本文讨论的主线里，但是非常有意思，未来outcome+process reward的结合可能也是一条路径；
    - 最后就是最近的Generalist Reward Models：Endogenous Reward。在海量人类（认为正确的）语料上进行next token prediction本身就是在学习一个reward function。这延续了DPO的想法，不仅仅在后训练，在预训练上LLM就可能同时是policy & reward model。

# LLM隐含了验证模型
<img src="https://i.mji.rip/2025/07/16/41910a45eba55627de00243f82207fdc.webp" width="800"/>

- 再往右，是RLVR下方的和Secret Reward Model互为镜像的观点：对于RLHF，LLM本身就是一个reward model；对于RLVR，LLM本身就是一个verifier model。这里包括了我们最近提出的工作NOVER，以及一系列相关工作。这一系列paper的动机和观察很简单：RLVR依赖可验证的任务，如果我只有freeform ground truth，无法rule-based verify，那怎么办呢？一个很直觉的想法就是用policy model在这些ground truth上的ppl作为奖励，如果是incentive training，那就用conditioned on reasoning trajectory的ppl作为奖励。想法很简单，但这和Secret Reward Model遥相呼应证明了一个观点：无论是RLHF还是RLVR，无论是alignment还是incentive training，LLM本身就足以成为一个足够好的反馈信号提取器。在强化学习的语境下，LLM本身就是足够好的policy & reward。

# Scale up强化学习
<img src="https://i.mji.rip/2025/07/16/3f4d3619d0bf4112fbc0386139871937.webp" width="800"/>

- 最后下方，是大家最近在RL for LLM里发力的一件事，也就是如何scaling RL:
    - DeepSeek-GRM强调Reward Model是值得scale up的；
    - ProRL认为RL training是值得scale up的，并且有希望突破pretrained ceiling；
    - POLAR说Reward Model不仅值得scale up，而且值得用预训练的规格scale up

# 汇聚一点
<img src="https://i.mji.rip/2025/07/16/298e8cda183da68a8d7f44739a1cde8e.webp" width="800"/>

- 这时我们回过头看走来的路，就会发现一切回归到了一个点：
    - Reward Model应该scale up
    - 后训练应该scale up
    - LLM本身就是Reward Model
- -->我们只需要scale up LLM本身！它既是policy，也是reward，而RL能够增强它作为两者的能力，越强的reward也能在更难的问题上给予policy更好的指导。最简洁的形态能够实现对数据和算力的最大包容。只要将任务、数据整理恰当，一切就如同齿轮一下能够转起来，越转越快，撬动更大的杠杠。这就是上图里Kimi K2关于general reasoning部分的insight。正如Hyung Won Chung那张ppt所言，less structure, more performance，我们最开始添加了各种模型，各种架构，然后我们再一一删掉：

<img src="https://i.mji.rip/2025/07/16/23a1a1ebeb4308046cb91071b0afa865.webp" width="800"/>

{% blockquote [[Hyung Won Chung's tweet on "Less Structure"](https://x.com/hwchung27/status/180067631291665659)] %}
As a community we love adding structures but a lot less for removing them. We need to do more cleanup.
{% endblockquote %}

# 下一步
<img src="https://i.mji.rip/2025/07/16/ecb64f8e8fd05a37beaa567f2e1c4fff.webp" width="400"/>

- 那下一步，我们该scale什么？以何种方式scale up？
- 一个看似显然的观点是，从training到inference，再到agentic，再到multiagent，下一步我们应该scaling up multiagent（Grok 4 Heavy应该是最近最强的multiagent模型/框架，multiagent也是2025 AI最火热的词之一）。

{% blockquote [[Grok 4](https://x.ai/news/grok-4)] %}
We have made further progress on parallel test-time compute, which allows Grok to consider multiple hypotheses at once. We call this model Grok 4 Heavy, and it sets a new standard for performance and reliability.
{% endblockquote %}

- 但是为什么？因为这些名词一个比一个新吗？我的想法是，scale up的不是这些名词，这些范式。新的范式兴起时，我们热衷于为其添砖加瓦，但最终都被模型所内化，让模型受益于scaling。Scaling law依然忠实于数据和算力，只不过数据和算力以不同的形式呈现。
- 在关注下一个scaling的方向时，我们关注它是以agentic/multiagent的形式出现，更要关注支持这些形式来做scaling的数据从哪来。个人的一些零碎的想法是：
    - 合成数据不可避免，但我猜想合成数据也会出现类似Discriminator-Generator Gap的情况，让我们能够利用一些tradeoff来不断产生更难、更好的数据。
    - 在强化学习的语境下，数据还可能以环境的形式出现。良好定义的环境理论上能产生近似无限的数据，因此下一阶段不仅仅是scale数量，更要scale难度，例如一个能够对越优秀的agent分配更难目标的环境。
{% blockquote [[Kevin's tweet on "Stop Working on RL"](https://x.com/_kevinlu/status/1942977315031687460)] %}
Why you should stop working on RL research and instead work on product // The technology that unlocked the big scaling shift in AI is the internet, not transformers
{% endblockquote %}
    - 我赞同training-->inference-->agentic-->multiagent的scaling路线，不仅仅因为这些名词被创造时就aim更高的智能，而是因为这个路线代表LLM能够越来越有用。有用的一个好处在于，人们愿意为LLM提供更多的、更不同的数据来使用它。
    - 对于MultiAgent，我感兴趣的是heterogeneous的，personal的multiagent系统。LLM几乎耗光了互联网上所有知识，但这些知识是大众的，是人类全体信息的期望。对于个人而言，每个人所处的小世界都在不断地产生近乎无限的数据和环境。为每个人分配一个agent，让人类社会拥有一个agent社会镜像，在无限的人类活动和个人环境下进行进化，是我认为multiagent实现下一个scaling的可能性。

# 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:
```bibtex
@article{next_scaling_202507,
  author = {Wei Liu},
  title = {What is the Next Step for Scaling in the Era of RL for LLM?},
  year = {2025},
  month = {7},
  url = {https://thinkwee.top/2025/07/15/next-scaling-202507/#more},
  note = {Blog post}
}
```

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