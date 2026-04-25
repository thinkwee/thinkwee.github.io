---
title: "HiMe: A Personal Health Agent in the Wild"
date: 2026-04-25 11:46:22
categories: LLM
tags:
  - agent
  - health
  - llm
  - ios
  - self-hosted
---

<img src="https://raw.githubusercontent.com/thinkwee/HiMe/main/assets/logo.png" width="500"/>

Introducing HiMe: Towards Personal Health Agentic Intelligence.

<!--more-->

{% language_switch %}

{% lang_content en %}

# What is HiMe?

[HiMe](https://github.com/thinkwee/HiMe) is a self-hosted personal health AI agent. It connects Apple Watch and iPhone HealthKit data to a local backend in real time, then lets the user talk with an agent through Telegram or Feishu. The agent can answer questions about recent health patterns, generate reports, set scheduled or event-triggered checks, and create small personalised pages such as trackers or dashboards.

In short, it is an attempt to move the understanding of wearable data from passive charts toward an interactive health companion that can understand and respond to a person's real-time data stream. It is still research-grade software, not a medical device, and does not constitute medical advice.

# Why Health Agent

The next scaling of agents may come from heterogeneous, personal multiagent systems: the internet has almost been exhausted as an average of humanity, while each person's micro-world still continuously generates data, environments, and feedback. HiMe is a small attempt along this line.

At first glance, personal health looks like an ordinary vertical application. Apple Watch collects heart rate, sleep, steps, workouts, HRV, blood oxygen and many other signals; an app draws charts; a model writes a summary. This version is useful, but I do not think it is the most interesting part.

The more interesting question is: **can a person's long-running health stream become an environment for an agent?**

The difference is subtle. A dashboard assumes the user already knows what to ask. I hope an agent can help discover what is worth asking. A dashboard optimizes for visualization. A health agent may need to care more about timing, memory, evidence, and follow-up. A dashboard is pulled by the user. A health agent might sometimes push back: not as a doctor, but as a sensor-facing companion that notices weak signals before the user opens the app.

# From Data to Environment

In many AI discussions, "environment" still means a benchmark, a simulator, a game, or a carefully wrapped tool-use task. Wearable data gives a different kind of environment:

- It is continuous rather than episodic.
- It is personalized rather than drawn from a public dataset.
- It has delayed and ambiguous feedback.
- It is full of confounders that the sensor cannot observe.
- It is sensitive enough that hallucination matters, but not formal enough to be solved by rule-based verification.

This is where I would place HiMe on the hard-to-verify side. "Your resting heart rate is 5 bpm higher than your 30-day baseline" is easy to verify. "You are probably under-recovered today" is already a semantic interpretation. "You should change tomorrow's training" crosses into advice and should be treated with much more caution. A system like this has to move between facts, hypotheses, and actions, and I would like these three to remain distinguishable.

So the key problem is not "can LLMs summarize health data". Of course they can. The key problem is whether we can build the right loop:

```text
sensor stream -> query -> analysis -> evidence -> memory -> trigger -> user feedback -> better next query
```

If this loop works, a personal health agent may be closer to a **World Sensor** than to a chatbot. It collects fragments of the user's world, compresses them into usable state, and tries to decide when the state deserves attention.

# What I Want to Learn

HiMe is also an early attempt to probe several questions:

- If personal micro-worlds continuously generate data, can personal agents become a real scaling path rather than a toy application?
- Can an agent learn useful long-term context without pretending to be a doctor?
- Can user-specific interfaces become a normal output modality of agents, beyond chat and reports?
- Can grounding make an agent trustworthy enough for sensitive personal domains?
- Can we build AI systems that participate in daily life without immediately falling into metric hacking?

I do not know the answers yet. But personal health seems to be a good testbed because it is simultaneously mundane and difficult. It has data, but the data is incomplete. It has feedback, but the feedback is delayed. It has goals, but the goals are not clean reward functions. It is personal, but not isolated from society. It is useful, but dangerous if overclaimed.

{% endlang_content %}

{% lang_content zh %}

# HiMe 是什么？

[HiMe](https://github.com/thinkwee/HiMe) 是一个自托管的个人健康 AI Agent。它连接 Apple Watch 和 iPhone 的 HealthKit 数据，将数据即时同步到本地后端，然后让用户通过 Telegram 或飞书和 Agent 对话。Agent 可以回答近期健康模式相关的问题，生成报告，设置定时或事件触发的检查，也可以创建一些小的个性化页面，例如 tracker 或 dashboard。

简单来说，它是一个尝试：把可穿戴数据的理解模式从被动图表，变成一个可以理解实时跟人数据流的交互的健康 companion。它仍然是 research-grade software，不是医疗器械，也不构成医疗建议。

# 为什么是健康 Agent

Agent 下一步的 scaling 也许来自 heterogeneous, personal multiagent systems：互联网作为人类平均意义上的知识已经几乎被耗尽，但每个人的小世界依然在持续产生数据、环境和反馈。HiMe 是沿着这条线做的一个小尝试。

表面上，个人健康像是一个普通的垂直应用。Apple Watch 采集心率、睡眠、步数、运动、HRV、血氧等信号；App 画几张图；模型写一段总结。这个版本当然有用，但我觉得它并不是最有意思的部分。

更有趣的问题是：**一个人长期运行的健康数据流，能否成为 Agent 的环境？**

这里的差别很微妙。Dashboard 默认用户已经知道该问什么。我希望 Agent 能帮助发现什么值得问。Dashboard 优化的是可视化。健康 Agent 也许更需要关心时机、记忆、证据和追问。Dashboard 由用户主动拉取。健康 Agent 有时也许应该主动推送：不是作为医生，而是作为一个面向传感器的 companion，在用户打开 App 之前先注意到一些弱信号。

# 从数据到环境

在很多 AI 讨论里，environment 仍然意味着 benchmark、simulator、game，或者一个被精心封装的 tool-use task。可穿戴数据提供了另一种环境：

- 它是连续的，而不是 episodic 的。
- 它是个人化的，而不是来自公共数据集。
- 它的反馈延迟且模糊。
- 它充满传感器无法观测到的混杂因素。
- 它足够 sensitive，因此幻觉很重要；但又不够形式化，无法靠 rule-based verifier 解决。

我会把 HiMe 放在 hard-to-verify 的一侧。“你的静息心率比 30 天 baseline 高了 5 bpm”容易验证；“你今天可能恢复不足”已经是语义解释；“你应该调整明天的训练”则跨入建议，应当更加谨慎。这样的系统需要不断在 facts、hypotheses、actions 之间移动，而我希望这三者不要被混在一起。

所以关键问题不是“LLM 能不能总结健康数据”。当然可以。关键问题是能否建立一个正确的循环：

```text
sensor stream -> query -> analysis -> evidence -> memory -> trigger -> user feedback -> better next query
```

如果这个循环成立，个人健康 Agent 也许更像一个 **World Sensor**，而不是 chatbot。它收集用户世界的碎片，将其压缩为可用状态，并尝试判断什么时候这些状态值得被注意。

# 我想验证什么

HiMe 也是用来探测几个问题的初步尝试：

- 如果个人 micro-world 持续产生数据，personal agents 能否成为真实的 scaling 路径，而不是玩具应用？
- Agent 能否学习长期上下文，同时不假装自己是医生？
- 用户特定的 interfaces 能否成为 Agent 的常规输出模态，而不只是 chat 和 reports？
- Grounding 能否让 Agent 在敏感个人领域达到可用的可信度？
- 我们能否构建一种参与日常生活的 AI 系统，而不立刻落入 metric hacking？

我还不知道答案。但个人健康似乎是一个很好的 testbed，因为它既日常又困难。它有数据，但数据不完整。它有反馈，但反馈延迟。它有目标，但目标不是干净的 reward function。它是个人的，但又不脱离社会。它有用，但过度声称就危险。

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
