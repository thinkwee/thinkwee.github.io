---
title: "[Some Questions asking Myself 2024.4]"
date: 2024-04-23 15:50:02
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

<img src="https://i.mji.rip/2025/07/16/122eedc8f2f91dc50d74b7b244718f69.png" width="500"/>


Some very-personal questions, assumptions and predictions on the future after the large model era. I hope to keep it a habit for writing such future-ask post for every half year to keep me thinking about the "next token" in the AI era. This post is about Compression, World Model, Agent and Alignment.

<!--more-->

{% language_switch %}

{% lang_content en %}

# Is Compression Our Only Path to General Intelligence?

## Is compression all we need?
- The first question is about compression.  
- Large models compress all the textual data in the world into the parameters of a single model, enabling everyone to "extract" information through natural language interaction. This process undoubtedly alleviates knowledge or information asymmetry. For example, a dentist can query an LLM to write code, while a programmer can enhance their paper writing with the assistance of an LLM. Extracting pre-encoded knowledge from LLMs is always beneficial. However, our aspirations go beyond this simple query-based knowledge retrieval. We wonder:  
  - **Can new discoveries emerge from the compressed information/knowledge in these models?** For instance, could a physicist uncover a new law from an LLM? Or could an LLM predict the content of this post? The answer is uncertain: it could be yes or no.  
    - On the affirmative side, mathematicians provide an example—many discoveries in pure theoretical research arise solely from scientists' cognitive processes and prior knowledge. Compression-based large models excel at leveraging past knowledge. If they can effectively simulate the cognitive process of scientists, they might achieve groundbreaking discoveries.  
    - On the negative side, some discoveries require empirical observation. They are "discovered" because someone observes them, such as the identification of new species in biology, which cannot be inferred merely from known information.  
    - Another question worth pondering is whether new discoveries are even necessary. After all, perhaps 99.999% of the world's activities in the next second follow established patterns. A tool that efficiently extracts and applies these patterns can still profoundly impact humanity. While this is true, our pursuit of AGI compels us to strive for more than this pragmatic goal.
  - The core question hinges on **"Is compression all we need?"**{% ref compression_for_agi %} If I could compress all the world's myriad and diverse data into a model, could it predict the future? If the model could accurately simulate the entire world, the answer would be yes—fast-forwarding the simulation would reveal glimpses of the future. But does compression combined with conditional extraction truly equate to simulation?
  - Elon Musk once remarked that the focus should be on the transformation between energy and intelligence. **Is compression the best method for such transformation?** Perhaps it serves as an efficient intermediary between energy and compressed knowledge (instead of intelligence).  
  - Related to this "compression question" is another: **"Is predicting the next token all we need?"** This question probes the limits of procedural and causal knowledge representation.


## World Model: A Data-Driven Approach?

- Regarding world models, a popular concept posits that intelligence comprises several interconnected subsystems (e.g., cognition, memory, perception, and world models), informed by human cognitive priors. The world model specifically refers to our brain's simulation of the world, enabling decision-making without waiting for real-world interaction.  
- The aspiration is to model these subsystems individually. However, most of our data is either unsupervised or end-to-end (holistic rather than divided into subsystems). Unsupervised data poses challenges in enabling all subsystem functionalities (e.g., language model pretraining struggles with instruction-following). End-to-end data might not train all subsystems effectively.  
- If we could segment and organize data to correspond to these subsystems, could we achieve a world model in the form of multi-agent or multi-LM systems?


## Agents

- Could OpenAI's *Bitter Lesson* overshadow many aspects of research on large models? Will agent-based research meet a similar fate? In other words, even after scaling up large models, will the research focus on agents remain irreplaceable? This might depend on whether the most rudimentary outputs of LLMs can transition from "System 1" (intuitive responses) to "System 2" (deliberative reasoning){% ref system2 consciousness_prior %}.  

- If an agent possesses all the actions and information of a human, can we consider it equivalent to a human?


## Alignment and Feedback

- Everything revolves around the **data flywheel**. The objective is to achieve better signals with each update by aligning the model.  
- Alignment demonstrates the importance of improving positive samples rather than focusing on negative samples, distinguishing it significantly from contrastive learning.  
- Alignment{% ref rlfh %} can be beneficial or detrimental, depending on the goal to which the model is aligned.  
- Some interesting questions are:
  - How can we integrate various forms of feedback (human/non-human, textual/other modalities, social/physical)?  
  - By connecting all these feedback types, we might align models with more powerful goals. Moreover, the laws governing this integration could reveal fundamental rules of the world.  
  - Reward models exemplify the energy hidden in tradeoffs: by sacrificing some precision, we gain scalable training, rewarding, and labeling. This tradeoff results in stable improvements. Can we uncover more such "energy" within these processes?  
    - For example, could cascading reward models (like interlocking gears) amplify the reward knowledge encoded by human annotations across datasets?  
  - Similarly, the **alignment tax**{% ref alignment_tax %} represents another tradeoff. Is there latent "energy" in these tradeoffs, where sacrificing A for B leads to overall intelligence gains?


## Beyond Language

- Language is more intricate, reasoned, and abstract than other modalities because it is fundamentally "unnatural"—a construct of human invention.  
- Nonetheless, researchers have identified an elegant objective for language: **predicting the next token**, a goal reflecting the entire history of computational linguistics.  
- Other modalities, like images, videos, and sounds, are "natural," as they convey raw information from the physical world. Could these modalities have objectives as intuitive or powerful as predicting the next token?  
- What implications do multimodal capabilities have for the reasoning abilities of large models?  


## Cite This Post
If you find this post helpful or interesting, you can cite it as:

```bibtex
@article{next_on_llm_2024,
  author = {Wei Liu},
  title = {[Some Questions asking Myself 2024.4] Compression, World Model, Agent and Alignment},
  year = {2024},
  month = {4},
  url = {https://thinkwee.top/2024/04/23/next-on-llm/#more},
  note = {Blog post}
}
```

{% endlang_content %}

{% lang_content zh %}

# 压缩是我们通往通用智能的唯一可能吗？
## 压缩能解决一切吗?
- 这是第一个问题，关于压缩。
- 大模型将世界上所有的的语料压缩到一个模型的参数中，每个人都可以通过自然语言交互来"提取"信息。这个过程无疑缓解了信息或知识的不对称性。例如，一个牙医可以通过查询LLM来编写程序，而程序员也可以通过LLM的协助来提升工作效率。从LLM中提取已编码的现有知识总是有益的，但我们想要肯定不仅仅是这种简单的知识查询，而是更多的可能：
  - 是否可能用现有的压缩信息/知识发现新事物？例如，物理学家能否从LLM中发现新定律？或者LLM能否预测这篇文章？目前没有答案，可能是，也可能不是。
    - 积极的一面是，数学家已经证明了这一点，因为许多纯理论研究发现仅源于科学家的大脑和过去的知识。基于压缩的大模型擅长利用过去的知识，如果它们能有效地模拟科学家的认知过程，就可能发现新的发现。
    - 消极的一面是，一些新发现往往来自经验观察（他们"发现"某物是因为他们看到了它），比如在生物学中，新的物种不可能仅仅从已知物种信息的推理中得出。
  - 另一个值得思考的问题是，新的发现是否真的必要。可能下一秒钟的世界，99.999%的活动都遵循着既定模式。一个能够高效提取和应用这些模式的工具仍然可以深刻影响人类。即便如此，我们追求 AGI（通用人工智能）的理想肯定远大于仅仅挖掘实用模式。
  - 核心问题在于“压缩是否就是我们需要的全部？”{% ref compression_for_agi %}如果我能将世界上千变万化的数据压缩进一个模型，它能否预测未来？如果模型能够准确模拟整个世界，答案就是肯定的——快进模拟将揭示未来的片段。但是，压缩与条件提取的结合真的等同于模拟吗？
  - 埃隆·马斯克曾表示，应关注能量与智能之间的转化。压缩是否是这种转化的最佳方法？或许它充当了能量与压缩知识之间的高效中介，而不是能量与智能的中介。
  - 与此“压缩问题”相关的是另一个问题：“预测下一个标记是否就是我们需要的全部？”这个问题探讨了程序和因果知识表示的极限。


## 世界模型，以数据驱动的方式？
- 对于世界模型，一种流行的说法是基于人类认知的先验将智能分割成几个相互连接的子系统（包含认知/记忆/感知/世界模型）。世界模型包含了我们脑中对于世界的模拟，这样我们不用等到与真实世界交互之后再进行决策。
- 我们期望用模型分别建模这几个子系统，但我们拥有的大多数数据都是无监督的或端到端的（整体而不是分成多个子部分的数据），而无监督的建模难以实现所有部分的功能（例如预训练阶段的语言模型无法实现指令跟随），端到端的数据则不确定能否训练好所有子系统。
- 如果数据可以为所有这些子系统分割和组织，我们能否以多智能体或多LM系统的形式实现世界模型？

## 智能体
- OpenAI的"痛苦教训"可能会掩盖许多关于大模型的研究。智能体是否会面临类似的命运？
- 换句话说，即使在扩大大模型规模后，智能体研究的内容是否仍然不可为LLM所替代？这可能取决于LLM最朴素的响应是否能从系统1过渡到系统2{% ref system2 consciousness_prior %}。
- 如果一个智能体拥有一个人的所有行为和信息，我们能说它就是一个人吗？

## 对齐/反馈
- 一切都与数据飞轮有关。目标是在每次更新中对齐模型后获得更好的信号。
- 对齐{% ref rlfh %}证明我们需要更好的正样本，而不是构建负样本。这是与对比学习的一个显著区别。
- 对齐可以是好的也可以是坏的，这取决于模型对齐的目标。
- 我们如何连接各种反馈，包括人类/非人类、文本/其他模态、人类社会/物理世界。
- 通过链接所有这些反馈我们可以与更强大的目标对齐。更重要的是，连接这些反馈的规律可能揭示世界的规则。
- RLFH使用人类标注训练奖励模型，然后用奖励模型扩大规模训练语言模型，实现了人类标注和模型标注的tradeoff，牺牲了一点奖励的精确度，换来了奖励样本的scale up。RLFH挖掘了这种tradeoff中蕴含着的能量，即牺牲多少质量，换来多少数量，可以带来最终效果的稳定提升。我们是否可能挖掘更多这种能量的可能性？比如级联的reward model，例如齿轮一般，将人类标注的奖励知识不断放大到样本中。
- 类似的，alignment tax{% ref alignment_tax %}也是一种tradeoff，这种tradeoff是否也有牺牲A换来B但最终提升了整体智能的能量？

## 超越语言
- 语言比其他模态更复杂、更具推理性、更抽象，因为它实际上是"非自然的"，是人类创造的。
- 但研究人员为语言找到了一个令人惊叹的目标，它反映了计算语言学的整个历史，即语言模型，也就是预测下一个词。
- 图像/视频/声音等其他模态是自然的，因为它们传达了物理世界的原始信息。这些信息能否有类似或更朴素的目标？
- 更多模态对大模型的推理能力意味着什么？


## 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:

```bibtex
@article{next_on_llm_2024_4,
  author = {Wei Liu},
  title = {[Some Questions asking Myself 2024.4] Compression, World Model, Agent and Alignment},
  year = {2024},
  month = {4},
  url = {https://thinkwee.top/2024/04/23/next-on-llm/#more},
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
[compression_for_agi] Compression for AGI - Jack Rae | Stanford MLSys #76 https://www.youtube.com/watch?v=dO4TPJkeaaU

[system2] LeCun, Y. (2022). A path towards autonomous machine intelligence. version 0.9. 2, 2022-06-27. Open Review, 62(1), 1-62. 

[consciousness_prior] Bengio, Y. (2017). The consciousness prior. arXiv preprint arXiv:1709.08568.

[rlfh] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

[alignment_tax] Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., ... & Kaplan, J. (2021). A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861. 

{% endreferences %}