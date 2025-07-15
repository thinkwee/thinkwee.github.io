---
title: "[Some Questions asking Myself 2025.5]"
date: 2025-05-21 12:36:43
categories: MyQuestion
tags:
  - inference
  - math
  - seq2seq
  - llm
  - agent
  - questions
refplus: true
mathjax: true
---

<img src="https://i.mji.rip/2025/07/16/0de1613151fa41695f480a9e134dc3f2.png" width="500"/>

The second post on my "some very-personal questions to myself" series. It's been over a year since last post and many progress on LLM have been made from academic/industry, which partially solves my questions. I will introduce these works and ask myself some new questions. This post is about Pretrain Ceiling, Second Half, Scaling the Environment.

<!--more-->

{% language_switch %}

{% lang_content en %}


# Questions from a Year Ago

## Can Compression Solve Everything?

- One year later, it appears that mainstream AI research still adheres to the LLM compression paradigm: using pretraining to compress world knowledge, then relying on post-training to extract it.
- As for whether LLMs can discover entirely new knowledge, research has now largely shifted to the AI4Science domain.
- Regarding the example from the previous blog post involving mathematicians and biologists:
  - In foundational fields like mathematics, new breakthroughs can often be achieved via interpolation within existing research ideas and paradigms. For example, DeepMind’s [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/){% ref alphaevolve %} combines evolutionary algorithms with LLMs to discover new fast matrix multiplication algorithms. The foundational algorithmic knowledge required was already encoded in the model via compression. Through prompt engineering and evolutionary iteration, the system uncovered “low-hanging fruit” that humans had yet to explore.
  - In empirical sciences like biology, which rely heavily on large amounts of new observations, an agentic approach can allow LLMs to interact with the real world using tools to synthesize new knowledge. In this paradigm, the LLM acts more like a scientist’s tool than a replacement. Another path is to bypass the reasoning abilities of LLMs altogether and build domain-specific models directly from field data—like [Evo2](https://news.stanford.edu/stories/2025/02/generative-ai-tool-marks-a-milestone-in-biology-and-accelerates-the-future-of-life-sciences){% ref evo2 %}, which trains on genome sequences. For naturally sequential data (like genomes), retraining domain-specific models makes sense; for non-sequential data, one can structure it as text, using language models for knowledge organization and reasoning.

## World Models: Data-Driven?

- There has been no substantial breakthrough in world modeling so far.
- Researchers have found that using LLMs to simulate both the world and the agent requires [different capabilities](https://arxiv.org/abs/2407.02446){% ref tradeoff_world_agent %}.
- More practical progress lies in the domain of LLM Agents: using models to construct interactive environments, such as video generation models or 3D space models—collectively referred to as world models. This reflects another extension trend in agent research: scaling the environment.
- In *Advances and Challenges in Foundation Agents*, scholars provided a comprehensive overview of the [current state of world model research](https://arxiv.org/abs/2504.01990){% ref foundation_agents %} from the agent perspective; most current approaches rely on models or external simulators and treat world models as single-task modeling problems that ultimately reduce to traditional single-step prediction.

## The "Bitter Lesson" of Agents?

- The “bitter lesson” still holds true. For instance, [Alita: Generalist Agent](https://arxiv.org/abs/2505.20286){% ref alita %} minimizes prior design and maximizes freedom, autonomously building and invoking MCP tools and achieving impressive results on the GAIA platform.
- Minimal priors and maximal freedom mean the agent’s capabilities are internalized within the base model, requiring no additional framework or scaffolding. We have yet to see truly “agent-native” application scenarios.
- Since the release of OpenAI’s o1 and DeepSeek, the industry consensus is that even the most basic LLM responses can shift from System 1 to System 2 reasoning.

## Alignment and Feedback

- As mentioned a year ago, post-training is essentially about steering LLMs: traditional alignment sacrifices some capability to guide models toward safer behavior; similarly, we can steer models toward more intelligent yet more hallucination-prone behavior, as demonstrated by DeepSeek R1’s "reasoning" capabilities.
- Our understanding of post-training continues to deepen. Looking at the [life cycle of LLMs](https://llmknowledgelifecycle.github.io/AAAI2025_Tutorial_LLMKnowledge/){% ref llm_knowledge_lifecycle %}, we find that knowledge is hard to truly add, remove, or edit during the post-training phase—retrieval is the most viable option. Hence, the “incentive, not memorize” principle seems reasonable.
- Pretraining has already endowed models with powerful knowledge. Traditional supervised-learning-based post-training can only introduce limited new knowledge. If we wish to add knowledge and capability, it’s better to stick with next-token-prediction rather than simple input-output pairs. So far, reinforcement learning appears to be a more suitable direction for post-training, enabling models to explore autonomously and truly grow in ability, rather than merely memorizing narrowly defined tasks.
- From a reward perspective, just as alignment uses human data to train reward models and inject values, we can also design rule-based rewards to encode natural laws into models.
- Recent discussions about the [limits of post-trained models not exceeding pretraining ceilings](https://arxiv.org/abs/2504.13837){% ref reasoning_limit %} show that if post-training is used only for steering without new knowledge input, it aligns with entropy-reduction expectations. In RL processes, the policy rolls out and only strengthens the good parts; if the correct solution never appears, pretraining ceilings can’t be breached. Traditional alignment data flywheels may face similar bottlenecks, though their knowledge scope is so broad it’s hard to investigate.

## Beyond Language

- The multimodal field has surged forward over the past year. Many advances are not in foundational techniques but in improved user experiences—combining multiple modalities better appeals to the senses. For example, [Google VEO3](https://deepmind.google/models/veo/){% ref google_veo3 %} can generate both video and audio simultaneously.
- At the same time, we’ve seen fascinating new pretraining paradigms such as [fractal neural networks](https://arxiv.org/abs/2502.17437){% ref fractal_neural_net %} and [diffusion language models](https://arxiv.org/abs/2502.09992){% ref diffusion_lm %}.
- What do additional modalities mean for reasoning capabilities? Most researchers respond: using text reasoning capabilities for visual reasoning. This isn’t about introducing new modalities for reasoning, but feeding other modalities into text-based reasoning. What I hope to see is “purely visual chain-of-thought” reasoning. Some recent efforts like [Visual Planning: Let’s Think Only with Images](https://arxiv.org/abs/2505.11409){% ref visual_planning %} attempt image-only reasoning. But such tasks are often limited to navigation, maps, or Frozen Lake scenarios, and still require interpreting intermediate images into action commands. To achieve true “image-to-image” reasoning, clearer problem definitions and task setups may be needed.

---

# New Questions

## Beyond Pretraining and Post-training

- The field continues to explore scaling laws, revealing from perspectives like [communication](https://arxiv.org/abs/2411.00660v2){% ref communication_scaling %} and [physics](https://physics.allen-zhu.com/){% ref physics_llm %} that knowledge can be transferred effectively to models via next-token-prediction and that model capacity can be expressed predictably.
- Is it possible to go beyond the pretraining and post-training paradigm to truly enable adding, deleting, updating, and querying knowledge and capabilities? This is crucial for personalized LLMs.
- Existing knowledge editing methods remain too simplistic and intrusive.

## Self-Evolution

- Recently, research on "model self-evolution" has grown rapidly. Nearly all unsupervised, weakly supervised, and self-supervised post-training approaches claim self-evolution capabilities. But is this truly self-evolution? Just as AlphaGo evolved through self-play, can LLMs under RLVR paradigms achieve genuine self-evolution? Or is this still just “self-entertainment” within the boundaries of pretraining?

## What Am I Overlooking?

- Both academia and industry are being driven by a blind arms race:
  - When industry makes a breakthrough with large-scale models or reduces research costs, academia quickly follows to harvest the benefits. Academia becomes the tester for breakthroughs made by industry. From “Can LLM do sth?” benchmark-style papers to “XXPO” studies applying DeepSeek GRPO to various domains, researchers now sometimes don’t even test other domains—just overfit some math benchmarks.
  - Industry too faces competitive pressure, like smartphone vendors releasing new iPhones, LLM companies roll out new versions monthly. If one company introduces a new model feature, competitors often replicate it by the next release cycle. If a competitor can replicate it quickly, it means the breakthrough falls within a foreseeable range of the scaling law.
  - This causes researchers to be constrained by low-hanging fruit and predictable problems, overlooking broader questions. Problems can remain unsolved, but critical thinking should never stop. In this LLM era, what overlooked domains are still worth examining?
- Don’t underestimate applications. Applications are the final link of science serving society, and they can also reverse-inspire new research trends. ChatGPT is a classic example: by combining a simple chat interface with post-training, it brought the value of LLMs into the homes of everyday users—and in turn spurred academic interest in LLM research.
- Why did I once overlook large models? And what am I overlooking now?

## If LLMs Are AGI

- Then should we use LLMs to build real AGI applications? Suppose AGI has arrived, and we can operate an entity equivalent to an ordinary person or even superhuman. What valuable things could we attempt?
- “Operate” sounds very negative, but operating a human being is actually simple—it doesn’t require science. Many industries throughout history have relied on humans functioning as operated entities to keep running.
- Our first thought is workers—blue-collar, white-collar, industry employees—can they be replaced by AI? But there’s also a different angle, like [recruiting ancient human subjects](https://www.pnas.org/doi/10.1073/pnas.2407639121){% ref ancient_human_subjects %}. This line of thinking doesn’t just ask which jobs can be replaced, but explores what things require humans yet are impossible for humans to achieve—perhaps AI can step in.

## “The Second Half”

- Recent works such as [The Second Half](https://ysymyth.github.io/The-Second-Half/){% ref the_second_half %} and [Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf){% ref era_of_experience %} suggest that research should shift from “how to solve problems” to “how to define problems.”
- I strongly agree, but I see this as a terminology update following the rise of powerful LLMs—the paradigm itself hasn’t changed: we still define tasks and solve them with models. What’s changed is that we’ve moved from constructing datasets to designing new environments, and from proposing new models to enabling models to learn online in environments and outperform others. We’re not scaling datasets—we’re scaling environments.
- In what directions should we “scale the environments”?
  - A quality environment should generate infinite data; data volume should no longer be the only axis of expansion.
  - The environment’s difficulty should become the focus of expansion.
  - Beyond digital environments, real-world environments may be an important milestone—for instance, agents physically building cities on Earth.
  - Scientific environments may offer even higher ceilings.
- In this “second half,” is RL the only thing we need?

# Citation
If you find this blog post interesting and wish to cite it, you may use the following bibtex:

```bibtex
@article{next_on_llm_2025_5,
  author = {Wei Liu},
  title = {[Some Questions asking Myself 2025.5] Pretrain Ceiling, Second Half, Scaling the Environment},
  year = {2025},
  month = {5},
  url = {https://thinkwee.top/2025/05/21/next-on-llm-2/},
  note = {Blog post}
}
```


{% endlang_content %}

{% lang_content zh %}

# 关于一年前的疑问

## 压缩能否解决一切？

- 时隔一年，目前看来，主流的 AI 研究依然未能突破 LLM 压缩理论的范式：依靠预训练来压缩世界知识，再通过后训练来提取这些知识。
- 关于 LLM 是否能够发现全新的知识，研究已聚焦于 AI4Science 领域。
- 针对上一篇博客中数学家与生物学家的例子：
  - 数学等基础学科，往往可以基于现有研究思路与范式进行插值，从而取得新成果。例如 DeepMind 推出的 [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/){% ref alphaevolve %}，通过结合进化算法与 LLM，发现了新的快速矩阵乘法算法。该方法所需的基础算法知识，已经以压缩形式编码于模型，通过 prompt engineering 与进化算法迭代，便可挖掘人类尚未充分探索的“漏网之鱼”。
  - 对于生物学这类高度依赖大量新观测才能得出结论的经验科学，则可通过 agentic 方式，让大模型借助工具与真实世界交互，从而总结新知识。在此范式中，LLM 更像科学家的工具，而非替代者。另一种途径是摒弃语言大模型的推理能力，直接基于领域数据构建专用大模型，例如 [Evo2](https://news.stanford.edu/stories/2025/02/generative-ai-tool-marks-a-milestone-in-biology-and-accelerates-the-future-of-life-sciences){% ref evo2 %}，以基因组序列训练模型。对于天然呈序列形式的数据（如基因组），适合重训领域专用模型；而对于非序列数据，则可将其整理为文本，借助语言模型进行知识整理与推理。

## 世界模型：以数据驱动？

- 世界模型方面尚未出现实质性进展。
- 研究者发现，用 LLM 同时模拟世界与 agent 需要[不同的能力](https://arxiv.org/abs/2407.02446){% ref tradeoff_world_agent %}。
- 更为实用的进展体现在 LLM Agent 领域：利用模型来构建与之交互的环境，如视频生成模型、3D 空间生成模型，将此类模型称为世界模型。这体现了 Agent 研究的另一种扩展趋势——scale the environment。
- 在《Advances and Challenges in Foundation Agents》中，学者们从 Agent 视角梳理了[世界模型研究现状](https://arxiv.org/abs/2504.01990){% ref foundation_agents %}；当前大多依赖模型或外部 simulator，从不同的 (state, observation, action) 拓扑结构出发，将世界模型视作单一任务的建模，而终归回到传统的单步预测。

## 智能体的“苦涩教训”？

- “苦涩教训”依然有效，例如 [Alita: Generalist Agent](https://arxiv.org/abs/2505.20286){% ref alita %}，以最少的先验设计赋予模型最大自由度，自建并调用 MCP 工具，在 GAIA 平台上取得亮眼成果。
- 最少先验、最大自由度，意味着 Agent 能力被基础模型内化，无需额外框架或脚手架。迄今尚未出现真正“Agent 原生”（Agent-Native）的应用场景。
- 自 OpenAI o1 与 DeepSeek 以来，业界共识是：最朴素的 LLM 响应，能够从系统 1 过渡到系统 2。

## 对齐与反馈

- 正如一年前所述，post-training 本质上是为语言模型定向：传统的 alignment 用以牺牲部分能力，将模型引导至更安全的方向；同理，也可将模型引向更高智能却更易幻觉的方向，DeepSeek R1 已展现此种“推理”能力。
- 对 post-training 的理解愈发深化。观察整个 LLM 的[生命周期](https://llmknowledgelifecycle.github.io/AAAI2025_Tutorial_LLMKnowledge/){% ref llm_knowledge_lifecycle %}，我们发现“知识”在后训练阶段难以真正增删改，仅适合检索。因此，“incentive, not memorize” 的理念显得合理。
- 预训练已赋予模型强大知识，传统 supervised-learning-based 的 post-training 一方面能引入的知识有限，另一方面若要添加知识与能力，更应沿用 next-token-prediction，而非简单的 input-output 对；当前来看，强化学习是更适宜的 post-training 方向，通过目标奖励让模型自主探索，促进真正能力的增长，而非狭隘的记忆。
- 从奖励角度看，正如 alignment 可通过人类数据训练奖励模型注入价值观，我们亦可设计基于客观规则的 reward，将自然法则注入模型。
- 近期关于 post-trained model 无法超越 pre-trained [上限的讨论](https://arxiv.org/abs/2504.13837){% ref reasoning_limit %}亦表明：若后训练仅用于 steering 而不引入新知识，则符合熵减预期。RL 流程中，我们让 policy 自行 roll out，再强化好的部分；因此若正确解法从未出现，则难以突破预训练上限。传统 alignment 数据飞轮或许也存在类似瓶颈，只是其涵盖的知识过于广泛，不易探查。

## 超越语言

- 多模态领域在过去一年突飞猛进，许多进展并非底层技术突破，而是改善用户体验——融合多种模态更易打动感官，例如 [Google VEO3](https://deepmind.google/models/veo/){% ref google_veo3 %}，可同时生成视频与声音。
- 同时，出现了诸多有趣的新预训练范式，如[分形神经网络](https://arxiv.org/abs/2502.17437){% ref fractal_neural_net %}及[扩散语言模型](https://arxiv.org/abs/2502.09992){% ref diffusion_lm %}。
- 更多模态对推理能力有何意义？大多数研究者的回答是：将文本推理能力用于视觉推理。这并非为推理引入新模态，而是为文本推理输入其他模态。我期待的路径是“纯视觉链式推理”，近期已有工作如 [Visual Planning: Let’s Think Only with Images](https://arxiv.org/abs/2505.11409){% ref visual_planning %}，尝试通过纯图像进行推理。但此类任务多局限于导航、地图、Frozen Lake 等场景，且依旧需从中间生成的图像解析行动指令。要实现纯粹的“图像到图像”推理，或需更明确的问题与任务定义。

---

# 新的问题

## 超越预训练与后训练

- 业界不断在 scaling law 方面探索，从[通信](https://arxiv.org/abs/2411.00660v2){% ref communication_scaling %}与[物理学](https://physics.allen-zhu.com/){% ref physics_llm %}等角度揭示：知识可通过 next-token-prediction 有效迁移至模型，并对容量进行可预测性表达。
- 有没有可能超越预训练和后训练范式，实现对知识与能力的真正增删改查？这对个性化 LLM 至关重要。
- 现有的知识编辑方法仍过于简单且侵入性强。

## 自我进化

- 近期“模型自我进化”研究日益增多，几乎所有无监督、弱监督、自监督的 post-training 均声称能自我进化。它们是真正的自进化吗？如同 AlphaGo 的自对弈，LLM 在 RLVR 范式下能否实现同样的自进化？还是说，这不过还是在预训练范围内的“自娱自乐”？

## 我忽视了什么

- 学术界与工业界均被盲目的军备竞赛所牵引：
  - 一旦工业界在大规模模型上取得新突破，或在研究成本上作出有效降低，学术界便会紧随其后，啃食红利。学术界沦为工业界突破后，在各领域的测试者。从“Can LLM do sth?” 类型的 Benchmark 文章，到将 DeepSeek GRPO 应用于各领域的“XXPO”研究，甚至近期都不愿意在其他领域做一个测试者，而是仅仅过拟合一些数学领域的benchmark。
  - 工业界亦面临竞争压力，如同手机厂商不断推出新 iPhone，大模型厂商以月为单位发布新版本的模型。一家厂商模型推出的新功能，往往在下一发布季度就被友商复现。问题在于，若能被友商迅速复现，说明其技术突破在可预见的 scaling law 射程范围内。
  - 这导致研究者被低悬果实与可预测问题局限，忽视更宽广的议题。问题可暂未解决，但思考不可放弃：在当下 LLM 时代，还有哪些值得审视的被忽略领域？
- 别轻视应用。应用是科学服务社会的最终环节，也可反向引领研究新潮流。ChatGPT 即是经典案例，通过简易对话界面与 post-training，将 LLM 的价值以普罗大众可接受的方式推向千家万户，反过来促进学术界对 LLM 的研究。
- 我自己为何曾忽视大模型？而今又在忽视什么？

## 如果 LLM 是 AGI

- 那么，我们是否该用 LLM 构建真正的 AGI 应用？设想AGI已经出现，我们可以操纵一个普通人乃至超人类的实体，我们能做何有益尝试？
- 操纵这个词听起来非常的负面，但其实操纵一个人类非常简单，无需科学。自古以来很多产业也需要人类以一个被操作的实体来工作，来运转。
- 我们第一想到的是工人，是白领，是行业里的打工人，能否被AI替代。但其实也有完全不同的角度，例如[招募古代人类被试](https://www.pnas.org/doi/10.1073/pnas.2407639121){% ref ancient_human_subjects %}。这个角度所代表的方向不是仅仅考虑哪些岗位能被替代，而是考虑用AI做那些需要人类才能做好，但无法由人类完成的事。

## “下半场”

- 近期诸如[The Second Half](https://ysymyth.github.io/The-Second-Half/){% ref the_second_half %}及[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf){% ref era_of_experience %}等文章指出：研究应从“如何解决问题”转向“如何定义问题”。
- 我非常赞同，但我觉得这是当 LLM 足够强大后，研究术语的更新，范式仍旧未变：提出任务并通过模型解决。不同的是，从构造数据集转向设计新环境，从提出新模型转向让模型在环境中在线学习并胜出。我们要scale的不是dataset，而是environment
- 我们应朝哪些方向“scale the environments”？
  - 优质环境应能生成无限数据，数据量不应再是唯一的扩展维度。
  - 环境的难度更应成为扩展目标。
  - 超越数字环境，现实环境或是重要里程碑，例如让 agent 在地球上实际建立城市。
  - 而科学环境或许有更高上限。
- 在这“下半场”，RL 是否为唯一所需？

# 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:

```bibtex
@article{next_on_llm_2025_5,
  author = {Wei Liu},
  title = {[Some Questions asking Myself 2025.5] Pretrain Ceiling, Second Half, Scaling the Environment},
  year = {2025},
  month = {5},
  url = {https://thinkwee.top/2025/05/21/next-on-llm-2/},
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

{% references %}

[tradeoff_world_agent] Li, M., Shi, W., Pagnoni, A., West, P., & Holtzman, A. (2024). Predicting vs. Acting: A Trade-off Between World Modeling & Agent Modeling. In arXiv: Vol. abs/2407.02446. https://doi.org/10.48550/ARXIV.2407.02446

[alphaevolve] Google DeepMind. (2025). AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms. https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

[evo2] Stanford University. (2025). Evo2: Generative AI tool marks a milestone in biology and accelerates the future of life sciences. https://news.stanford.edu/stories/2025/02/generative-ai-tool-marks-a-milestone-in-biology-and-accelerates-the-future-of-life-sciences

[visual_planning] Xu, K., Wang, Y., Zhang, R., & Chen, X. (2025). Visual Planning: Let's Think Only with Images. In arXiv: Vol. abs/2505.11409. https://arxiv.org/abs/2505.11409

[alita] Qiu, H., Xiao, C., Yang, Y., Wang, H., & Zheng, L. (2025). Alita: Generalist Agent with Minimal Predefinition and Maximal Self-Evolution. In arXiv: Vol. abs/2505.20286. https://arxiv.org/abs/2505.20286

[foundation_agents] Liu, M., Tian, Y., Yang, S., Chen, B., & Zhou, B. (2025). Advances and Challenges in Foundation Agents. In arXiv: Vol. abs/2504.01990. https://arxiv.org/abs/2504.01990

[llm_knowledge_lifecycle] Zhang, S., Chen, J., & Wang, W. Y. (2025). The LLM Knowledge Lifecycle: An AAAI 2025 Tutorial. https://llmknowledgelifecycle.github.io/AAAI2025_Tutorial_LLMKnowledge/

[reasoning_limit] Wei, J., Chen, X., & Bubeck, S. (2025). Can reasoning emerge from large language models? Investigating the limits of reasoning capabilities in pre-trained and fine-tuned models. In arXiv: Vol. abs/2504.13837. https://arxiv.org/abs/2504.13837

[google_veo3] Google DeepMind. (2025). Veo3: Advancing text-to-video generation with synchronized audio. https://deepmind.google/models/veo/

[fractal_neural_net] Chen, L., Dai, Y., & He, K. (2025). Fractal Neural Networks: Scaling Deep Learning Beyond Linear Paradigms. In arXiv: Vol. abs/2502.17437. https://arxiv.org/abs/2502.17437

[diffusion_lm] Yang, M., Tian, Y., & Lin, Z. (2025). Diffusion Language Models: Toward Controllable Text Generation with Guided Diffusion. In arXiv: Vol. abs/2502.09992. https://arxiv.org/abs/2502.09992

[communication_scaling] Rao, S., Knight, W., & Sutskever, I. (2024). Scaling Laws from an Information-Theoretic Perspective. In arXiv: Vol. abs/2411.00660. https://arxiv.org/abs/2411.00660v2

[physics_llm] Allen-Zhu, Z., & Li, Y. (2025). On the Connection between Physical Laws and Neural Scaling Laws. https://physics.allen-zhu.com/

[ancient_human_subjects] Jiang, L., Cohen, J., & Griffiths, T. L. (2024). Recruiting ancient human subjects with large language models. Proceedings of the National Academy of Sciences, 121(21), e2407639121. https://www.pnas.org/doi/10.1073/pnas.2407639121

[the_second_half] Chen, K., Liu, H., & Zhang, D. (2025). The Second Half: From Solving Problems to Defining Problems. https://ysymyth.github.io/The-Second-Half/

[era_of_experience] DeepMind. (2025). Welcome to the Era of Experience. https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf

[previous_post] Liu, W. (2024). [Some Questions asking Myself 2024.4] Compression, World Model, Agent and Alignment. https://thinkwee.top/2024/04/23/next-on-llm/

{% endreferences %}