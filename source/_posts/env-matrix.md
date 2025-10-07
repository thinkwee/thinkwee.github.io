---
title: Scaling the Environment 
date: 2025-07-17 20:50:32
categories: LLM
tags:
  - rl
  - scaling
  - llm
  - agent
refplus: true
mathjax: true
---

<img src="https://i.mji.rip/2025/07/18/2e9daed2f82508d5db2c449cbb90188a.png" width="400"/>

What I Talk About When I Talk About Scaling the Environment?
<!--more-->

{% language_switch %}

{% lang_content en %}

# Scaling Environments

<img src="https://i.mji.rip/2025/07/18/1844572f3d9fd6443aa0d80ced17728a.gif" width="400"/>


- In the previous blog post, we mentioned the importance of scaling environments in the era of RL for LLMs.
- Similar to the pre-LLM era where we scaled the quantity and quality of data, in the RL era, we scale the difficulty of environments.
- We can evaluate the difficulty of an environment from three dimensions:
    - **Generation Difficulty**: The difficulty of collecting new problems/goals/tasks within the environment
    - **Solving Difficulty**: The difficulty of the problems assigned to the agent within the environment
    - **Verification Difficulty**: The difficulty of verifying whether the agent’s output is correct after completing a task
- These difficulties determine how easy it is to build an environment, and whether the constructed environment is sufficient to train powerful agents.
- A precise terminological distinction is that verification typically refers to checking if a prediction matches a ground truth using a verifier model; whereas a reward model generally evaluates the quality of a prediction without having ground truth. The former emphasizes consistency checking, the latter emphasizes obtaining a ground truth. For the sake of discussion, we do not strictly differentiate between the two in this post. After all, if it is hard to obtain ground truth, then it is naturally hard to verify. For tasks that are difficult to verify, under the discussion of a general reward model, they also correspond to tasks that are difficult to model a reward function for.
- Each dimension can be categorized as either simple or difficult. This binary classification mainly emphasizes relative difficulty—e.g., generation being harder than solving, or verification being easier than generation. This classification helps us clarify the goals and direction when scaling environments. Under this classification, we can form an environment matrix with eight subspaces.
- It’s worth noting that we ignore two subspaces where verification is harder than solving:
    - Because most RL-applicable problems exhibit a generator-discriminator/verificator gap, i.e., it is easier to judge whether a policy is good than to get a good policy.
    - If getting a policy is easy (e.g., proposing a mathematical conjecture or making a future prediction), but verification is hard (requiring higher intelligence or a long time), then in that domain, the problem that needs solving is verification itself.
    - From this perspective, verification becomes the policy that needs to be learned, while "policy generation" becomes more akin to problem generation.
    - Therefore, if such subspaces are to be solved with AI methods, they can be categorized into other subspaces. However, for completeness, we still illustrate them here and mark them in gray to indicate their objective existence.

# First Layer: Generation, Solving, and Verification Are All Easy

<img src="https://i.mji.rip/2025/07/18/416ea48192da73fc891db6780832f390.png" width="500"/>

- Tasks in this category are simple in every aspect. Typically, humans already have mature and robust templates or rules for them—e.g., unit conversion, spelling correction, arithmetic, etc.
- Strategies for such tasks can be written using rules and do not require complex AI systems to learn.
- Interestingly, although these are the simplest environment settings, LLMs are not necessarily the best strategies here—in fact, they do not need to be. For example, with arithmetic for countless attempts, we have no theoretical guarantee that a language model never make mistakes, whereas a calculator will never make mistakes. Or for arithmetic on billion-digit numbers, the language model’s context cannot hold it all, but a simple big integer algorithm can handle it easily.
- Does this imply that language models are not good? Of course not. Rather, it means that for different types of problems, we need differently designed intelligences. Language models can simply solve such problems by calling a calculator or writing code. When simple problems challenge the robustness of high-level intelligence, high-level intelligence can use induction, reasoning to organize lower-level intelligences to solve them.

# Second Layer: Either Solving or Generation Is Very Difficult

<div class="image-row" style="display: flex; justify-content: center; align-items: center; gap: 20px; background: white; padding: 20px;">
<img src="https://i.mji.rip/2025/07/18/347cf0be1465655fa6bd861e550a1f7a.png" width="500" style="margin: 0; display: block;"/>
<img src="https://i.mji.rip/2025/07/18/5152a5ca7844a1f2ea7c837178dc2407.png" width="500" style="margin: 0; display: block;"/>
</div>

<style>
.image-row p {
  display: flex !important;
  margin: 0 !important;
  padding: 0 !important;
}
</style>

- This layer corresponds to most current RL research for LLMs. Two representative directions are RLHF and RLVR:
  - **RLHF** corresponds to scenarios where generating high-quality problems (data collection) is very difficult. For product-grade LLMs, we need to collect real-world queries from actual user logs rather than relying on simple datasets for preference learning. Therefore, constructing challenging tasks/goals is very difficult. Initially, the challenge of RLHF seemed to be verification difficulty, but a series of works have shown that with high-quality human preference data, reward models can indeed learn accurate human preferences. Good data consists of two parts: good questions and good model answers (not just good answer, but good rollouts from better trained policy llm). All this depends on deploying on-policy RL into product-grade LLMs and achieving data flywheels.
  - **RLVR** corresponds to scenarios where solving problems is very difficult. It was only two years after RLHF became common in LLM post-training that the RLVR paradigm emerged. Before RLVR was applied to mathematics, there was no shortage of math problems or easy verification for their results. However, when the base model's capabilities were insufficient and the search space was not optimized, it was hard to explore strategies in early-stage RL that would yield positive feedback. It’s like a monkey typing on a keyboard—while it could theoretically type Shakespeare, we don’t know how long it would take. But if RL starts from a strong base, it's like a PhD in literature typing, making the probability of generating literature much higher. People now realize the importance of pretraining a strong base model for RLVR, and some mid-training efforts are also emerging.

# Third Layer: Only Verification Is Easy / Only Generation Is Easy

<div class="image-row" style="display: flex; justify-content: center; align-items: center; gap: 20px; background: white; padding: 20px;">
<img src="https://i.mji.rip/2025/07/18/7be89f0c106e83509ee5926096aa6d75.png" width="500" style="margin: 0; display: block;"/>
<img src="https://i.mji.rip/2025/07/18/b8d96a4a7d43b700e2fa37ba101ca386.png" width="500" style="margin: 0; display: block;"/>
</div>

- This layer corresponds to directions we are about to explore. It is hard, but as more effort is invested, these environments will be gradually constructed to train more advanced intelligence.
  - **Only Verification Is Easy**: Both generation and solving are difficult. A typical example would be the highest-difficulty math problems. Math problems with standard answers are always easy to verify, and currently, most high-difficulty math problems can be solved by LLMs. To further improve intelligence, we need even harder math problems. But where do we collect them? That’s the difficulty—requiring the smartest human minds to continuously produce more difficult (but solvable) problems to train the models. This process is clearly unsustainable and cannot scale up. Humanity's final mathematical frontier can be updated annually, but it will become thinner and thinner, and the fact that problems must be solvable by humans limits the upper bound of this type of intelligence. If AI generates and solves the problems, it violates the Generator-Verifier Gap. Therefore, constructing this type of environment is resource-constrained—specifically, limited by human intellectual resources.
  - **Only Generation Is Easy**: Both verification and solving are difficult. The main challenge here lies in verification—tasks that are subjective, require semantic understanding, lack unified evaluation standards, or have high time/labor verification costs, such as artistic/literary creation, policymaking, education, and healthcare. In these areas, we have a vast number of problems to solve, but it is very difficult to determine whether AI has solved them. A key feature of this subspace is human participation. AI will become part of human civilization, participating in and influencing social activities, receiving feedback from human society. This is a far more challenging direction—optimizing a system that includes both humans and AI.

# Final Layer: Expert-Level / Superhuman

<img src="https://i.mji.rip/2025/07/18/f1060d46cd95ba9f11ecbbbf1b9f9f7a.png" width="500"/>

- This subspace is difficult in all dimensions. We cannot take shortcuts by leveraging one dimension being easier than the others to train intelligence. I currently cannot give an example of this subspace, but it must exist. Perhaps at this level, AI will develop AI, regulate AI, and leverage AI.

# Citation
If you found the topics in this blog post interesting and would like to cite it, you may use the following BibTeX entry:
```bibtex
@article{next_scaling_202507,
  author = {Wei Liu},
  title = {Scaling the Environment},
  year = {2025},
  month = {7},
  url = {https://thinkwee.top/2025/07/17/env-matrix/#more},
  note = {Blog post}
}
```

{% endlang_content %}

{% lang_content zh %}

# Scaling 环境

<img src="https://i.mji.rip/2025/07/18/1844572f3d9fd6443aa0d80ced17728a.gif" width="600"/>


- 在上一篇博文中，我们提到在RL for LLM时代，scaling环境的重要性。
- 类似前LLM时代，我们scale数据的数量和质量，在RL时代，我们scale环境的难度。
- 我们可以从三个维度来衡量环境的难度：
    -   生成难度：生成一个环境并在环境中收集新的问题/目标/任务时的难度
    -   解决难度：在环境中，智能体解决问题需要多高的智能水平
    -   验证难度：智能体完成任务之后，验证这个智能体的交付结果是否正确的难度
- 这些难度决定了环境是否容易构建，以及构建出来的环境是否足以训练强大的智能体。
- 一个严谨的术语上的区别是，验证往往指给定ground truth，verifier model会判断prediction是否和ground truth一致；而reward model通常会在没有ground truth的情况下直接判断prediction的好坏。前者强调判断一致，后者强调得到ground truth。本文为了方便讨论，不严格区分两者，毕竟如果难以得到ground truth的话，自然也就难以验证。而对于难以验证的问题，在general reward model的讨论下，也对应着难以建模reward model。
- 每个维度都可以分为简单和困难，这种二分类主要是强调相对难度，即生成比解决困难，或者验证比生成容易。这种划分方式可以帮助我们梳理scaling环境时的目标和方向。在这种划分方式下，我们可以得到环境矩阵，其包含八个子空间。
- 值得注意的是，我们忽略了验证比解决困难的两个子空间，
    -   因为大部分适用于RL的问题都存在generator-discriminator/verificator gap，即判断策略好不好比得到一个最优策略容易。
    -   如果得到策略容易（比如提出一个数学猜想，或者对未来做一些预测），但是很难验证（需要高智力水平或者长时间来验证），那么在这个领域，需要解决的问题是完成验证，而不是提出猜想/策略。
    -   从这个视角来看，完成验证反而成为了需要学习的策略，而“得到策略”更像是生成问题
    -   因此这类子空间，如果要用AI的方式解决，则可以归为其他子空间，但在本文中，我们依然将其画出来，并标为灰色，代表其客观存在。

# 第一层，生成、解决、验证均简单

<img src="https://i.mji.rip/2025/07/18/416ea48192da73fc891db6780832f390.png" width="500"/>

- 这类任务无论从哪个角度都很简单，一般而言人类早已形成成熟的，鲁棒的模板或者规则，例如单位换算、拼写纠错、加减法等等
- 这类任务的策略可以用规则编写而成，无需使用复杂的AI系统学习
- 有趣的是虽然这是最简单的环境设置，但是大语言模型不一定是最好的策略，或者说，无需在该设置下成为最好的策略。例如加减法，进行无数次加减法，我们没有理论保障语言模型不出错，但计算器永远不会出错，又或者进行上亿位数字的加减法，语言模型的context无法装下，但是一个简单的大整数算法就能处理
- 这难道说明语言模型不行吗？当然不是，而是对于不同类型的问题，我们需要不同设计的智能。语言模型也可以简单的通过function call调用计算器或者写代码来解决这些问题。当简单的问题挑战了高级智能的鲁棒性时，高级智能可以通过归纳、推理，使用低级智能来解决这类问题。

# 第二层：解决或者生成很困难

<div class="image-row" style="display: flex; justify-content: center; align-items: center; gap: 20px; background: white; padding: 20px;">
<img src="https://i.mji.rip/2025/07/18/347cf0be1465655fa6bd861e550a1f7a.png" width="500" style="margin: 0; display: block;"/>
<img src="https://i.mji.rip/2025/07/18/5152a5ca7844a1f2ea7c837178dc2407.png" width="500" style="margin: 0; display: block;"/>
</div>

- 这一层对应着当前LLM大部分的RL研究，两个代表性方向就是RLHF和RLVR
  - RLHF对应着生成新的问题（收集数据）很困难的情况。对于产品级别的LLM，我们需要从实际的用户使用日志中收集真实世界的query，而不是在简单的数据集上完成偏好学习。因此构建有挑战的任务/目标非常困难。RLHF一开始的挑战似乎在于很难验证，但一系列工作表明如果有了高质量的人类偏好数据，reward model是足以学习到准确的人类偏好。好的数据包含两部分，好的提问，好的模型的回答（不仅仅是好的回答，而且是训练得到的更好的policy model的rollouts），这一切依赖于将on-policy RL部署到产品级LLM里，通过数据飞轮实现。
  - RLVR对应着解决问题很困难的情况。在LLM post-training普遍使用RLHF两年之后，才出现RLVR范式。在RLVR应用于数学领域之前，数学题的资源并不缺乏，数学题结果也非常容易验证，但是在基座模型能力不够强的情况下，搜索的空间没有经过优化，我们难以在RL早期探索出得到正向反馈的策略。类似猴子敲键盘，虽然理论上可以敲出莎士比亚著作，但不知敲到猴年马月。但如果从一个好的基座开始RL，就如同让一个文学博士敲键盘，敲出著作的概率要大大增加。现在人们意识到了pre-train一个优秀底座对于RLVR的重要性，很多mid-training的工作也开始兴起。

# 第三层：仅仅验证容易/仅仅生成容易
<div class="image-row" style="display: flex; justify-content: center; align-items: center; gap: 20px; background: white; padding: 20px;">
<img src="https://i.mji.rip/2025/07/18/7be89f0c106e83509ee5926096aa6d75.png" width="500" style="margin: 0; display: block;"/>
<img src="https://i.mji.rip/2025/07/18/b8d96a4a7d43b700e2fa37ba101ca386.png" width="500" style="margin: 0; display: block;"/>
</div>

- 这一层对应着我们接下来探索的方向。虽然很难，但我相信随着人们投入的增加，这两类环境会被逐渐构筑起来，训练更高水平的智能
  - 仅仅验证容易：生成和解决都很困难。一个典型的例子是，最顶尖难度的数学题。拥有标准答案的数学题永远容易验证，而当前世界上绝大部分高难度数学题也能被LLM解决，如果我们想要进一步提升智能，就需要更高难度的数学题。从何处收集？这是一个难题，需要人类最聪明的头脑不断的产生更难的（且人类已经解出）的难题，然后训练模型完成。这个过程显然不可持续也无法scale up。人类最后的数学防线每年都可以更新，但会越来越单薄，而且人类可解的前提限制了这一类智能的上限。如果是AI出题、AI解题，则违背了Generator-Verifier Gap。因此这类环境构建的难度在于资源（人类的智力资源）
  - 仅仅生成容易：验证和解决都很困难。这类任务主要的难点在于验证，即一些主观的/需要语义判断的/验证标准不统一的/需要高额时间成本或者人力成本验证的任务，例如艺术/文学创作、政策制定、教育和健康。在这些领域我们有海量的问题需要解决，但是很难判断AI能否解决这些问题。这个子空间一个重要的特征就是人的参与。AI会作为人类文明的一部分，参与并影响人类的社会活动，从人类社会获得反馈。这是更加难以探索的一个方向，将人与AI作为一个整体的系统去优化。

# 最后一层：专家级别/超出人类级别
<img src="https://i.mji.rip/2025/07/18/f1060d46cd95ba9f11ecbbbf1b9f9f7a.png" width="500"/>

- 这个子空间无论从哪个维度来看，都很难。我们没法借助一个维度比另一个维度更加简单的特点来取巧去训练智能。我暂时无法给出这个子空间的例子，但其必然存在。也许到了这个层次，AI会发展AI， 监管AI，利用AI。

# 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:
```bibtex
@article{next_scaling_202507,
  author = {Wei Liu},
  title = {Scaling the Environment},
  year = {2025},
  month = {7},
  url = {https://thinkwee.top/2025/07/17/env-matrix/#more},
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
