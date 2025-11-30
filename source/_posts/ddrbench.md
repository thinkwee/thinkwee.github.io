---
title: 'DDR-Bench: Benchmarking Agentic Data Research'
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

<img src="https://i.mji.rip/2025/11/30/4c4c183f8e7b492575bede143c35a1f8.png" width="700"/>

DDR-Bench: LLMs That Hunt Instead of Wait.

<!--more-->

{% language_switch %}

{% lang_content en %}

# What is Deep Data Research?

We introduce **Deep Data Research (DDR)**, a task where LLMs autonomously dive into databases to explore insights they deem important — **no pre-defined questions, no explicit targets, just fully autonomous Data→Insights**.

Unlike traditional QA or coding benchmarks, DDR evaluates whether models can **proactively set investigative goals** and extract meaningful insights from complex databases, mimicking how expert data scientists work in practice.

# Highlights

- **Verifiable Evaluation**: Checklist-based assessment extracted from unstructured reports, validated by 50+ domain experts
- **Three Diverse Domains**: Electronic Health Records (MIMIC-IV), Sport & Exercise Psychology (GLOBEM), Annual Financial Reports (10-K SEC filings)
- **Highest Autonomy**: No pre-set questions or targets — LLMs decide what to investigate
- **Minimalist Design**: Built for Agentic LLMs with simple ReAct prompts and minimal toolset (2 MCP servers, 6 functions)
- **Long-Horizon**: Up to 100 turns and 70,000+ tokens per trajectory

# Key Findings

- **Domain knowledge defines the ceiling** — it determines how deeply a model can reason within a domain
- **Exploration strategy governs whether models approach that ceiling** — reflecting the ability to generate informative hypotheses
- **Cost efficiency determines convergence speed** — advanced architectures achieve higher information gain per token

Current SOTA models still struggle to exceed 50% average accuracy, indicating DDR tasks are far from saturated.

# Read More

For detailed methodology, experimental results, and analysis on test-time scaling and exploration patterns, check out the full write-up: 👉 **[DDR-Bench Notion Blog](https://thinkwee.notion.site/ddrbench)**

{% endlang_content %}

{% lang_content zh %}

# 什么是深度数据研究？

我们提出了**深度数据研究（DDR）**任务，让LLM自主深入数据库，探索它认为重要的洞见——**没有预设问题，没有明确目标，完全自主地从数据到洞见**。

与传统的问答或编程基准不同，DDR评估模型是否能够**主动设定研究目标**，并从复杂数据库中提取有意义的洞见，模拟专业数据科学家的实际工作方式。

# 亮点

- **可验证评估**：基于非结构化报告提取的检查清单评估，经50+领域专家验证
- **三个多样化领域**：电子健康记录（MIMIC-IV）、运动心理学（GLOBEM）、年度财务报告（10-K SEC文件）
- **最高自主性**：无预设问题或目标——由LLM自行决定探索方向
- **极简设计**：为Agentic LLM构建，简单ReAct提示词和最小工具集（2个MCP服务器，6个函数）
- **长程交互**：每条轨迹最多100轮，70,000+ tokens

# 核心发现

- **领域知识决定天花板**——决定了模型在特定领域推理的深度
- **探索策略决定能否接近天花板**——反映了生成有信息量假设的能力
- **成本效率决定收敛速度**——先进架构能以更低成本实现更高的单token信息增益

目前SOTA模型平均准确率仍难以超过50%，表明DDR任务远未饱和。

# 了解更多

详细的方法论、实验结果、测试时扩展和探索模式分析，请查看完整文章：👉 **[DDR-Bench Notion Blog](https://thinkwee.notion.site/ddrbench)**

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
