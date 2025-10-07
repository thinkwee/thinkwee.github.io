---
title: (Welcome) to the Era of Wild
date: 2025-10-05 18:40:08
categories: LLM
tags:
  - rl
  - scaling
  - llm
  - agent
refplus: true
mathjax: true
---


<img src="https://i.mji.rip/2025/10/07/0b1adc08119a47115c49c4eaac3887e7.png" width="300"/>

Connecting the dots, (welcome) to the era of wild.

<!--more-->


{% language_switch %}

{% lang_content en %}

Three quarters of 2025 have already passed, and many exciting new developments have occurred in AI this year. Looking back, I feel many dots have once again been strung together:

-	DeepSeek R1
-	ClaudeCode/Gemini CLI/Codex
-	Era of Experience
-	The Second Half
-	RLVR
-	Agentic RL
-	HealthBench
-	GDPval
-	Sora 2 App
-	OpenAI Dev Day

A line that connects them is that AI will no longer be optimized merely toward datasets, toward metrics such as accuracy, but will be directly optimized toward the deep and complex goals in real-world human social activities. We put AI in the Wild, hoping it can directly participate in human social production activities, learn from feedback from the real world, and that such feedback can reach directly to the model, until every next token prediction, rather than being overly diluted by layers of external mechanisms.

# Evolution of Tasks

From the perspective of NLP, we have been making task move from simulation toward the real:

-	In the NLP era, the tasks the model completed were to classify a sentiment label for each text fragment, or to annotate dependency relations in a sentence, whereas humans (except linguists/NLPer) generally do not and do not need to do this
-	In the LLM era, models organize all NLP tasks into the form of prompts for processing, and humans do the same when communicating with language
-	In the Agentic era, models operate tools and produce artifact, and many human professions are precisely to do these things
-	In the Wild era, models and humans will jointly form a society, and task will no longer distinguish between real and simulated

On the surface, Wild AI looks like making application, but because of the change in the nature of task, it differs from traditional vertical domain AI: it is not about collecting NLP/LLM-formatted data from the domain to continue training models, but about deploying AI into the real world and directly optimizing for the domain’s ultimate goals.

From a technical perspective, this can be seen as application with RL, but the exploration of RL in LLM is still long; how to accurately bring back feedback produced by the final objectives to the model may require changes in human-machine interaction patterns, changes in model frameworks, and even paradigms beyond RL.

In the most naive sense, this is still RLHF and not a completely new path. It is precisely this path that helped OpenAI launch ChatGPT, allowing large models to truly reach human users, and the effects brought by real users’ preference feedback later became widely known: they made large models truly understood by the world and opened a new LLM era. And now everyone will race along this path at an unprecedented speed.

# Deep Goals

What are the deep and complex goals and feedback in human social activities? Possible (but not exhaustive) examples include:

-	GDP
-	employment rate
-	average life expectancy
-	crime rate
-	annual profit
-	cutoff score
-	box office
-	scientific discovery
-	h-index
-	global temperature
-	……

Why Does Wild AI Optimize for These Goals?

-	One of reinforcement learning’s most utilitarian advantages is optimizing for non-differentiable objectives, and there exist too many non-differentiable gaps between the model and real-world utility
-	People have discovered the potential of reinforcement learning plus strong priors; it may be possible to optimize for these goals
-	Scaling needs a new story; from training to inference, the next stage of scaling requires new data and new dimensions. The infinitely many environments in the real world, extremely long chains, and goals that currently seem fantastical all provide new soil for that story
-	The First Half of chasing dataset SOTA has become ineffective
-	Large companies have limited patience for long-term investment in AI

Recommendation systems are a field that has long entered the Wild Era: they optimize for GMV, profoundly transform our lives, trigger countless discussions such as information cocoons, and together with mobile internet shape the current landscape of tech company giants. The research directions of academia and industry have gradually diverged. And LLMs/AI entering the Wild Era will go further than recommendation systems in every aspect (or put it another way, be more severe).

# Welcome to the Era of Wild?

I will use the word Welcome cautiously. Goodhart’s law tells us that once a metric becomes a target it is no longer a good metric. The indicators currently defined for measuring human society already have many problems; even without AI’s participation, many pathological phenomena have already occurred in people’s pursuit of these indicators, and AI’s involvement may further accelerate this kind of hacking. In addition, from another perspective, certain individuals or organizations might also realize their objectives through Feedback Hijacking.

On the other hand, previously people were divided over whether AI should be developed first or governed first, but at present the development-first technical path has become somewhat clearer, and companies that develop first will not stop: AI’s transformation of the world will not pause because of debates. Under this premise, the technology for governing AI may actually need to evolve faster.

# The Future

In the new Wild Era, what technologies are still needed to develop AI?

-	Before moving into real society, first achieve truly reliable AI decision mechanisms, rather than still relying on probabilities
-	A new cycle of four major components: model/algo/data/infras. Currently when people do Agentic AI, they are still working within well-cared-for datasets and environments. But upon reaching real environments, current model architectures may not be suitable for extremely long chains, algorithms cannot fully utilize feedback, and the base infrastructure cannot support large-scale efficient training and inference. And when model/data/base infrastructure are all optimized to saturation, people will further go into the real environment to mine the next data treasure, bringing more complex data situations, which in turn will prompt the evolution of model/algorithm/base infrastructure, forming a cycle
-	Simulators: letting AI directly transform the real world is still somewhat radical; how to realistically simulate feedback and simulate the impact caused by AI is indispensable. Simulators can on the one hand conduct rehearsals before real transformation, and on the other hand scale up training, just like reward model
-	New feedback curation standards: like pretraining data, people need various complex strategies to ensure data quality, avoiding bad feedback signals brought by next token. In the real world, feedback signals are more complex and also require more systematic vetting
-	Hard to Verify: currently our progress is almost entirely on the Easy to Verify side within the Generator Verifier Asymmetry; after moving to the real world, we will flip to the other side, and how to mine and utilize feedback in the Hard to Verify domain is also a major topic
-	Collective intelligence: after AI enters human society, countless entities paired with AI will emerge, the entire human society mirrored as an AI society. Will such bottom-up intelligent collectives give rise to new intelligence and new social mechanisms? Will this correspondingly require new evaluation systems?
-	When optimizing for deep goals in real production activities, should evaluation and feedback be safely separated to avoid falling into the Goodhart trap?
-	Bidirectional optimization: maybe we will not only optimize AI for society, but also optimize society for AI
-	Mechanism-driven alignment: current alignment signals such as human preferences are still controlled by humans determining the direction of alignment; if preferences are inverted, the same process can make the model learn badly. In the future is it possible to set environmental mechanisms such that when models interact and experience in society and learn from feedback, no party can dominate the alignment signals, but rather such mechanisms achieve common goals
-	World Sensor: a feedback collector designed for human social activities, bridging the environment and the model

This is an era of confusion, bubbles, temptation, risk, and opportunity coexisting.

(Welcome) to the Era of Wild.

# Citation
If you found the topics in this blog post interesting and would like to cite it, you may use the following BibTeX entry:
```bibtex
@article{wild_era_202510,
  author = {Wei Liu},
  title = {(Welcome) to the Era of Wild},
  year = {2025},
  month = {10},
  url = {https://thinkwee.top/2025/10/05/wild-era/#more},
  note = {Blog post}
}
```

{% endlang_content %}

{% lang_content zh %}



2025年已经过去了四分之三，今年的AI发生了很多激动人心的新进展，回顾这¾年，我感觉很多dots又一次被串联起来:

-   DeepSeek R1
-   ClaudeCode/Gemini CLI/Codex
-   Era of Experience
-   The Second Half
-   RLVR
-   Agentic RL
-   HealthBench
-   GDPval
-   Sora 2 App
-   OpenAI Dev Day



串联起它们的一条线是，AI将不再仅仅面向数据集优化，面向accuracy这样的指标优化，而是将直接面向真实世界人类社会活动里深层次的复杂目标优化。我们 *put AI in the Wild*，希望其能直接参与人类社会生产活动，从真实世界的反馈中学习，而且这种反馈能直达模型，直到 every next token prediction，而不是被层层外部机制过分稀释。


# 任务的演变

从NLP的视角来看，我们一直在让task从模拟向真实迈进:

-   NLP 时代，模型完成的任务是为每个文本片段分类一个情感标签，或者在句子中标注依存关系，而人类（除了语言学家/NLPer）一般不会也无需这么做
-   LLM 时代，模型将所有NLP任务组织为prompt的形式进行处理，人类用语言沟通时会这么做
-   Agentic 时代，模型操作工具，产出artifact，很多人类的职业就是做这些事情
-   Wild 时代，模型和人类将共同组成社会，task将不再区分真实还是模拟



表面上看，*Wild AI* 像是在做application，但由于 task 性质的变化，它与传统垂域 AI 有所区别：不是从垂域里收集NLP/LLM形式的数据继续训练模型，而是将AI投放到真实世界，直接面向垂域最终的目标优化。

如果从技术上看，这可以看成是application with RL，但是RL in LLM的探索还很长，如何把最终目标带来的反馈准确的带回给模型，这其中可能需要人机交互模式的变革，模型框架的变革，甚至超过RL的新范式。

最朴素的来讲，这依然是RLHF，并不是全新的路线。正是这条路线帮助 OpenAI 推出了 ChatGPT，使得大模型真正触达人类用户，而真实用户的偏好反馈带来的效果后来也众所周知了：让大模型真正被世人所理解，开启了新的LLM时代。而现在大家会在这条路线上以前所未有的速度狂飙。



# 深层目标

什么是人类社会活动里深层次的复杂目标和反馈？可能（但不全面）的例子有：

-   GDP
-   就业率
-   平均寿命
-   犯罪率
-   年利润
-   分数线
-   票房
-   科学发现
-   h-index
-   全球气温
-   ......

为什么Wild AI面向这些目标优化？

-   强化学习最功利的一个优势就是，面向不可导的目标优化，而模型到真实世界效用之间存在太多不可导的gap
-   大家发现了强化学习+强先验的潜力，有可能做到面向这些目标优化了
-   Scaling需要新的故事，从training到inference，下一阶段的 scaling 需要新的数据与维度。而真实世界里无限多的环境、极长的链路、当前看来天方夜谭的目标都给故事提供了新的土壤
-   刷数据集SOTA的The First Half失效了
-   大型企业对AI长期投入的耐心有限

推荐系统是一个早已进入 *Wild Era* 的领域：它面向 GMV 优化，深刻改造了我们的生活，引起了无数类似信息茧房之类的讨论，和移动互联网一起塑造了今天的科技公司巨头格局，学术界和工业界的研究方向渐行渐远。而进入 *Wild Era* 的 LLM/AI，其在各个方面上都会比推荐系统更进一步（或者说更加严重）。


# 欢迎来到 *Wild Era*？

我会谨慎使用Welcome这个词。Goodhart's law告诉我们，一项指标一旦变成了目标，它将不再是个好指标。当前人们定义的用于衡量人类社会的各项指标本来就存在诸多问题，即便没有AI参与，人们在追求这些指标的过程中已经发生了很多畸形现象，而AI参与进来，可能会进一步加速这种hacking。此外，从另一个角度来看，某些个人或组织也可能通过Feedback Hijacking实现自己的目的。

另一方面，之前大家对于AI是先发展还是先治理各执一词，但目前来看先发展的技术路线又清晰了一些，先发展的公司也不会停下脚步：AI对于世界的改造不会因为各执一词的讨论而暂停。在这个前提下，如何治理AI的技术反而可能需要更快的进化。

# 未来

在新的Wild Era，发展AI还需要哪些技术？

-   迈向真实社会之前，首先要实现真正可依赖的AI决策机制，而不是依然靠概率
-   新的四大件循环：model/algo/data/infras。当前大家做Agentic AI，依然还是在呵护的很好的数据集和环境中做。而到达真实环境之后，当前的模型可能架构不适用于超长的链路，算法没法充分的利用反馈，基架支撑不起大规模高效训练和推理。而当模型/数据/基架都优化到趋于饱和时，人们就会进一步去真实环境里挖掘下一个数据宝藏，带来更复杂的数据情况，进而接着促使模型/算法/基架进化，形成循环。
-   模拟器：让AI直接改造真实世界还是有些激进，如何真实的模拟反馈，模拟AI带来的影响不可或缺。模拟器一方面能在真实改造之前进行演习，另一方面也能scale up训练，就像reward model一样。
-   新的feedback整理规范：就像预训练数据一样，人们需要各种复杂的策略来保证数据质量，避免next token带来不好的反馈信号。在真实世界，反馈信号更加复杂，也需要更成体系的甄别。
-   Hard to Verify：当前我们的进展几乎都在Generator Verifier Asymmetry里 Easy to Verify的那一侧，到真实世界之后，我们会翻到另一侧，Hard to Verify的领域如何挖掘并利用反馈也是一个大课题。
-   群体智能：AI进入人类社会后，将会出现无数pair with AI的实体，整个人类社会镜像为AI社会，这种自下而上的智能集群是否会涌现新的智能，新的社会机制？是否对应需要新的评估体系？
-   当面向真实生产活动深层次目标优化时，评估和反馈是否应该安全分离，避免步入Goodhart陷阱？
-   双向优化：也许我们不仅仅是optimize AI for society， 也会optimize society for AI。
-   机制驱动的对齐：当前的对齐信号比如人类偏好，依然是人类掌握着对齐的方向，偏好取反，相同的流程，就能让模型学坏。未来是否有可能设置环境机制，当模型在社会中交互、体验，并从反馈中学习时，没有任何一方能主宰对齐信号，而是靠这类机制达成共同目标。
-   World Sensor：面向人类社会活动设计的反馈收集器，架起环境和模型的桥梁。

这是一个迷茫，泡沫，诱惑，风险，机遇并存的时代。

(Welcome) to the Era of Wild.

# 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:
```bibtex
@article{wild_era_202510,
  author = {Wei Liu},
  title = {(Welcome) to the Era of Wild},
  year = {2025},
  month = {10},
  url = {https://thinkwee.top/2025/10/05/wild-era/#more},
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
