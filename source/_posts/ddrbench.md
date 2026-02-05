---
title: Deep Data Research -- Database as Hunting Ground, LLMs as Hunters
date: 2025-11-30 12:08:36
categories: LLM
tags:
  - llm
  - agent
  - reasoning
  - deep-learning
refplus: true
mathjax: true
---

<img src="https://i.mji.rip/2026/02/05/5a6bce13485a7772889a39fe9236f051.png" width="700"/>

Introducing DDR-Bench.
[*Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models*](https://arxiv.org/abs/2602.02039).

<!--more-->

{% language_switch %}

{% lang_content en %}

# What is Deep Data Research?

<img src="https://i.mji.rip/2026/02/05/7213512cd0803c7b680dca4daeee7f22.png" width="1000"/>


We introduce **Deep Data Research (DDR)**, a task where LLMs autonomously dive into databases to explore insights they deem important — **no pre-defined questions, no explicit targets, no interaction limit, just fully autonomous Data→Insights**.

Unlike traditional QA or coding benchmarks, DDR evaluates whether models can **proactively set investigative goals** and extract meaningful insights from complex databases, mimicking how expert data scientists work in practice.

Please checkout our [project page](https://huggingface.co/spaces/thinkwee/DDR_Bench) and [arXiv paper](https://arxiv.org/abs/2602.02039). 


# Highlights

<img src="https://i.mji.rip/2026/02/05/b446049d1cdb0ed7e937dac0310cf050.png" width="1000"/>


- **Verifiable Evaluation**: Checklist-based assessment extracted from unstructured reports, validated by 50+ domain experts
- **Three Diverse Domains**: Electronic Health Records (MIMIC-IV), Sport & Exercise Psychology (GLOBEM), Annual Financial Reports (10-K SEC filings)
- **Highest Autonomy**: No pre-set questions or targets — LLMs decide what to investigate
- **Minimalist Design**: Built for Agentic LLMs with simple ReAct prompts and minimal toolset (2 MCP servers, 6 functions)
- **Long-Horizon**: No limit put on the number of interactions, Agentic LLMs decide when to stop.

# Key Findings

<img src="https://i.mji.rip/2026/02/05/88c411853335b23ad0c6c4777fa4b305.png" width="1000"/>


- DDR evaluates investigatory intelligence rather than executional intelligence. The former places substantially higher demands on agency, requiring models to autonomously set goals and determine exploration directions.
- Frontier models already exhibit signs of agency, yet long horizon exploration remains the primary bottleneck.
- High quality Deep Data Research behavior emerges from a stable implicit coordination between reasoning and exploration, rather than from a simple accumulation of isolated capabilities.
- Explicit reasoning is often concentrated in the initial interaction rounds and gradually gives way to tool dominated behavior. Part of the reasoning is implicitly embedded in tool parameters and code comments, rather than being expressed through explicit chains of thought.
- Test time scaling analyses from the perspectives of interactions, tokens, and cost show that strong LLMs behave like hunters, patiently exploring before drilling deeply into insights, while exhibiting exceptionally high token efficiency.
- Increasing the reasoning budget can substantially raise the number of reasoning tokens and reduce the number of interaction rounds, but the final performance metrics fluctuate significantly. This indicates a trade off between reasoning depth and interaction frequency, where neither extreme is optimal.
- Effective agency depends on the model’s internal exploration strategy, rather than relying solely on agent modules or parameter scaling. Agent modules primarily reshape interaction patterns, instead of consistently improving deep data research capability.
- Training time factors systematically influence test time scaling behavior. The effects of parameter scale and long context optimization are weaker than those of agentic native training.
- Current SOTA models still struggle to exceed 50% average accuracy, indicating DDR tasks are far from saturated.

# Read More

- Please checkout
 -  [Notion Blog](https://thinkwee.top/2026/02/05/ddrbench/)
 -  [arXiv Paper](https://arxiv.org/abs/2602.02039)
 -  [Project Page](https://huggingface.co/spaces/thinkwee/DDR_Bench)
 -  [Github](https://github.com/thinkwee/DDR_Bench)
 -  [Dataset and Trajectory](https://huggingface.co/collections/thinkwee/ddrbench)

# Citation
If you found the topics in this blog post interesting and would like to cite it, you may use the following BibTeX entry:
```bibtex
@misc{liu2026huntinsteadwaitevaluating,
      title={Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models}, 
      author={Wei Liu and Peijie Yu and Michele Orini and Yali Du and Yulan He},
      year={2026},
      eprint={2602.02039},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.02039}, 
}
```

{% endlang_content %}

{% lang_content zh %}

# 什么是深度数据研究？

<img src="https://i.mji.rip/2026/02/05/7213512cd0803c7b680dca4daeee7f22.png" width="1000"/>


我们提出了**深度数据研究（DDR）**任务，让LLM自主深入数据库，探索它认为重要的洞见——**没有预设问题，没有明确目标，完全自主地从数据到洞见**。

与传统的问答或编程基准不同，DDR评估模型是否能够**主动设定研究目标**，并从复杂数据库中提取有意义的洞见，模拟专业数据科学家的实际工作方式。

请查看我们的[项目页面](https://huggingface.co/spaces/thinkwee/DDR_Bench)和[arXiv论文](https://arxiv.org/abs/2602.02039) 


# 亮点

<img src="https://i.mji.rip/2026/02/05/b446049d1cdb0ed7e937dac0310cf050.png" width="1000"/>


- **可验证评估**：基于非结构化报告提取的检查清单评估，经50+领域专家验证
- **三个多样化领域**：电子健康记录（MIMIC-IV）、运动心理学（GLOBEM）、年度财务报告（10-K SEC文件）
- **最高自主性**：无预设问题或目标——由LLM自行决定探索方向
- **极简设计**：为Agentic LLM构建，简单ReAct提示词和最小工具集（2个MCP服务器，6个函数）
- **长程交互**：没有交互轮数限制，Agentic LLMs决定何时停止并生成洞见。

# 核心发现

<img src="https://i.mji.rip/2026/02/05/88c411853335b23ad0c6c4777fa4b305.png" width="1000"/>

- DDR评估的是“探究式智能”而不是“执行式智能”，前者对于Agency有更高的要求，需要自主设定目标与决定探索方向。  
- 前沿模型虽显露能动性，但长程探索仍然是主要难点。 
- 优秀的Deep Data Research行为源于“推理—探索”的稳定隐式协同，而非单点能力叠加。 
- 显式推理多集中在开局轮次，随后逐步过渡到工具调用主导。部分推理被隐含进工具参数与代码注释中，而非体现在显式思考链。 
- 从interaction, tokens, cost三个角度的test-time scaling分析可以看出，优秀的LLMs像猎手一般，耐心探索，然后深入挖掘洞察，并且对token展现出极高的利用效率。
- 提高推理预算可能会显著增加推理token并减少交互轮次，但最终指标显著波动，显示推理深度与交互频率存在权衡，极端都不最优。 
- 有效能动性依赖模型内在探索策略，而非仅靠agent模块或参数scaling。 Agent模块更多是在重塑交互模式，而非稳定提升deep data research能力。 
- 训练时因素系统性影响推理时刻的scaling表现。 参数规模和长上下文优化的影响不如agentic-native training。
- 目前SOTA模型平均准确率仍难以超过50%，表明DDR任务远未饱和。

# 了解更多

- 请查看我们的
 -  [Notion Blog](https://thinkwee.top/2026/02/05/ddrbench/)
 -  [arXiv Paper](https://arxiv.org/abs/2602.02039)
 -  [Project Page](https://huggingface.co/spaces/thinkwee/DDR_Bench)
 -  [Github](https://github.com/thinkwee/DDR_Bench)
 -  [Dataset and Trajectory](https://huggingface.co/collections/thinkwee/ddrbench)

# 引用
如果觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:
```bibtex
@misc{liu2026huntinsteadwaitevaluating,
      title={Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models}, 
      author={Wei Liu and Peijie Yu and Michele Orini and Yali Du and Yulan He},
      year={2026},
      eprint={2602.02039},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.02039}, 
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
