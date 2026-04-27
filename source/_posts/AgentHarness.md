---
title: "Lessons from Burning 1.7B tokens: Harnessing Personal Agent"
date: 2026-04-26 09:55:29
categories: LLM
tags:
  - agent
  - health
  - llm
  - harness
---

<img src="https://i.mji.rip/2026/04/27/1b73a417652e5ff2aa969209d181ac3f.png" width="800"/>

Some practical notes on Agent Harness from building HiMe.

<!--more-->

{% language_switch %}

{% lang_content en %}

# Agent Harness for Personal Agents

This post summarizes a few practical lessons and reflections on **Agent Harness** from building **HiMe**. HiMe is a personal health agent I have been working on recently. It is the first open-source agent project that understands real-time wearable data streams and provides health support. Everything is open source: the data pipeline, the Agent Harness, and the iOS/Apple Watch app.

This was also my first relatively large-scale development project with Claude Code. In total, I used **1.74B Claude Code tokens**, mostly on Opus 4.6:

- **uncached input:** 370,000+ tokens
- **output:** 9,350,000+ tokens
- **cache write:** 57,200,000+ tokens
- **cache read:** 1,670,000,000+ tokens

The project ended up with 222 commits, and HiMe v1.0.0 contains more than 50,000 lines of code. Feel free to try it on [GitHub](https://github.com/thinkwee/HiMe).

I will not spend much time defining Agent Harness here. A simple way to think about it is this:

> In an LLM Agent product, **everything around the model that systematically supports the agent is part of the Harness**.

It is the engineering work required to turn a model into a working product. As base models become stronger, our expectations for agents also rise. The harness becomes more complex, then some of that scaffolding gets internalized by the base model, as in the idea that _"LLMs Eat Scaffolding for Breakfast."_ Then we expect even more from agents, build more complex harnesses, and the cycle continues. **Engineers and models co-evolve in this loop.**

One useful way to frame this project is that I am harnessing Claude Code to build a personal agent, while the end goal is to let users harness that personal agent with almost no effort.

> This transfer of control quietly shapes many design decisions.

Most of those 1.74B tokens went into these two layers of harnessing. There are already countless best practices for using Claude Code, so this post focuses on the second layer: how to design a personal agent that users can actually harness, and how to make that experience good.

> For a Personal Agent, what do users expect from the agent, and what kind of Harness does that imply?

I summarized several lessons from the development process. Almost all of them are product and engineering considerations, not algorithmic ones.

## A Harness Should Raise the Floor for Weak Models Without Capping Strong Ones

Not every user will use the strongest model in every scenario. A good Harness should provide reliable support when the model is weak, without imposing unnecessary structural constraints when the model is strong.

> Its job is to **define the environment**, not to prescribe the path.

The first mistake I made was trying to make weak models work by endlessly debugging prompts and taming the agent into a workflow. I hard-coded the system prompt into a process, wrote extremely detailed tool instructions, and even specified how each tool should be used at particular points in the workflow. The result was usually the worst of both worlds:

- weak models still found new ways to fail;
- strong models were reduced to engines for executing fixed scripts.

> A workflow compresses the future into the past. It tries to cover unseen situations with paths that have already been seen. But an agent operates in an open environment.

We should tell the model **the principles, not the exact moves**. In the official HiMe release, I removed all workflow descriptions, all in-context examples, and all concrete "don't do this" examples. I only described the agent's responsibilities. Current models can follow in-context examples well in normal QA, but in agent settings, where tasks are long-horizon and startup prompts are very long, their capabilities often degrade. They tend to memorize the examples instead of generalizing from them.

It is better to leave that feedback loop to the user and let the agent correct its behavior from feedback, instead of hard-coding a specific form of agency from the beginning. Only freeze what is deterministic. For principles that leave room for judgment, only solidify the deterministic parts, such as API libraries.

If a single agent session does not produce the desired result, do not rush to adjust the prompt. First check:

- whether the agent has the tools it needs;
- whether the environment gives it enough feedback;
- whether the task defined by the prompt is simply beyond the model's capability, if the agent still cannot complete it under ideal conditions.

HiMe handles context in two layers: curated context for weak models, and access points to more raw, information-rich context. Wearable-data access is one example. There is a cheap path that covers about 80% of use cases, where the agent directly works with a dataframe containing recent health data. There is also an escape hatch, where the agent can choose to read the full dataset.

> Design the environment for the agent, fallback strategies for the user, and feedback mechanisms between the two.

Under these principles, I spent most of my time making weak models behave reasonably. But once something worked well on a weak model, switching to a stronger model often produced pleasant surprises.

## Beyond Static Tests

HiMe did not have unit tests at the beginning, and problems appeared quickly. New features and new code kept piling in, disrupting the project structure and introducing many breaking changes that had to be reworked. So I added static unit tests. Claude Code is actually quite good with tests: after it finishes writing code, it runs the test suite to make sure the new code at least does not break existing functionality.

Later, though, I found that for an agent project, static tests only cover a small subset of mechanical issues, such as fallback behavior when an LLM provider times out. The issues that really shape the experience are often interaction-layer issues, and those only surface when someone actually uses the system.

So I introduced **User Tests**, where LLMs simulate users.

An LLM agent and an LLM user have completely different perceptual fields:

- a user does not need to understand, and cannot realistically audit, which internal step of the agent went wrong;
- the user only sees the messages sent by the agent or the artifacts it produces;
- the user responds by asking follow-up questions, pushing back, or simply expressing emotions based on what they can see.

> Users are not linters or debuggers.

By letting LLMs simulate users with different personalities and stress-test the agent, I found many issues, including silent feedback loops and poor handling of frequent messages.

## Multi-Agent as an Engineering Necessity

In the vast literature on LLM multi-agent systems, we often see debates about whether multi-agent systems are better than single-agent systems, or theoretical explanations for why they should be better. In HiMe, the reason for using multiple agents is much simpler: it is an engineering decision, not a way to chase better benchmark numbers.

> An agent is an abstraction over responsibility.

A good agent is like a good Python file: it has one responsibility and clear boundaries.

When one agent takes on too many responsibilities, problems accumulate quickly. Monitoring becomes harder, fallbacks become harder to design, and errors become harder to attribute.

HiMe uses a **"sub-agent as a tool"** design:

- the **Chat Agent** is the only main agent and handles all user interaction;
- the **analysis agent** performs read-only data analysis;
- the **management agent** handles memory reads/writes and other system-level operations in the HiMe framework.

Each agent has its own tool set, with read and write permissions separated. The main agent can call a sub-agent as a tool. It only needs to provide the goal; the sub-agent handles the multi-step operation and returns the result.

From a performance perspective, the difference between single-agent and multi-agent designs may not be large for strong models. For weak models, however, it matters a lot for raising the capability floor. Clear responsibilities reduce the amount of context and reasoning the model has to carry in each run.

Although HiMe is multi-agent, it does not use concurrency. In a personal health agent, the priority is long-term, stable support for the user, not maximizing the efficiency or quality of artifact production as in coding agents. A controllable agent loop is closer to a state machine than to a distributed system.

## Keep Maximum Distrust Toward LLMs

We should never assume that a probabilistic model will reliably do exactly what we expect. This is especially important in personal health.

> Prompts drift; code does not.

If a constraint can be expressed in code, do not rely on the prompt for it. Reduce uncertainty wherever possible.

Design multiple layers of fallback across different dimensions. HiMe contains **12 fallback chains**, covering cases such as:

- tool-call failures;
- context overflow;
- provider overload.

Different semantic failure modes use different retry and fallback strategies.

Any system that can enter a loop needs a breaker.

Take responsibility for hallucination. Most agents use web retrieval to keep responses factual, but for personal agents, the more common problem is **unfaithfulness**: for example, the agent may perform no tool call at all, then claim it did and produce a fabricated analysis.

HiMe includes an audit agent that reviews every message the Chat Agent sends to the user. By looking at the tool-call chain, it checks whether the quantitative or qualitative claims in the message are actually supported by query results from the user's real-time health data.

The persistence layer, including data and queues, must outlive the application layer, namely the agent loop. The data in a personal agent is a real user asset. It cannot be lost because the agent temporarily behaves abnormally.

Any agent operation with persistent side effects must be triggered through a chat path explicitly initiated by the user.

## LLMs Are Not Designed for True Proactivity

A defining feature of a Personal Agent is that the user is continuously generating data. Every person is a walking data stream. These streams are unlabeled, noisy, and low-density. Most importantly, LLMs were not designed for stream processing. They expect an complete input and produce an output for every such input. True proactivity would require the LLM to decide, as each token in the data stream arrives, whether it should output a token. That is not a built-in capability of today's general LLMs, unless the model has been specially trained for it, as in full-duplex voice LLMs.

In Personal Health, these streams are wearable data and other health data. Directly feeding these streams into an LLM is possible, but as described above, stream understanding is not a native capability of current general-purpose LLMs. HiMe positions the LLM agent as a data analyst rather than a data analysis engine. This makes good use of the LLM's knowledge, but gives up true stream processing. Early on, I tried letting the LLM read data, inspect the current personal context and observations, decide how long to sleep before the next read, and allow that sleep to be interrupted at any time. The results were poor.

In the end, HiMe implements proactivity through **co-construction between the user and the agent**:

In personal health, truly real-time stream understanding and alerts should not be left to a heavy analysis engine like an LLM. They should be handled by lightweight, specialized tools that can preserve real-time resolution.

When an LLM is needed, the important thing is **personalized proactivity, not higher-frequency proactivity**. At its core, proactivity is a decision about whether to take an action when an observation arrives. In HiMe, this is implemented through user-agent collaboration and layered solidification:

- Do not solidify it. Let the user decide when to run an analysis. This is essentially reactive.
- The user can ask the agent to solidify certain analyses into scheduled tasks.
- The agent can further refine those tasks into event-triggered forms, such as triggering when blood oxygen is too low, or when there has been no data for several consecutive hours.

These are also the two types of proactive tasks implemented by most agent products. HiMe additionally implements Generative Pages: the agent writes new app pages for the user, creating new ways and new moments to interact with the data stream. This is the most customizable form of proactive analysis.

## Generative App

The original motivation for HiMe's generative pages was to approximate the experience of a generative app: **the agent creates new app functionality for the user in real time**. For example, in HiMe, a user can ask the agent to create a running-pace recorder or a yoga workout logging panel.

But HiMe does not actually modify the app. Changing an app in real time is not a safe design. Once an app has been released and distributed, its functionality should not start changing just because AI has been introduced. Instead, HiMe lets the agent generate new pages on the user's self-hosted server. Both frontend and backend reads/writes happen on the server side. The app itself does not change; it only acts as a browser.

Even this already opens up a lot of imagination for future agent products. The highest form of personalization is letting users implement their own needs. Many missing features in agent products are really missing context. For a personal health agent, if the user does not enter the information, the agent will never know whether the user took vitamins today or how much weightlifting they did. Wearables cannot provide that information. But each user can ask the agent to create an interface for recording it. That data then naturally becomes part of the user's personal data stream and gives the agent better context.

## Generative Databases as Memory

HiMe does not have a complex memory mechanism. This is still an area for future exploration. Current high-level methods for agent memory construction and retrieval are mostly case-by-case engineering optimizations, and they do not transfer easily to the personal health agent setting.

> Memory is not just context recall.

In a personal agent, memory should be lifelong modeling of the user. A person is the collection of their contexts; a piece of memory is the part of that context that is useful to the agent's current session, and therefore a part of the person.

For long-term memory, such as user preferences or reusable experience accumulated by the agent, HiMe uses Prompt Markdown. The agent reads and writes Markdown files, which are then incorporated into system prompt construction. At least this makes memory human-auditable and human-editable. In the HiMe dashboard, users can hot-update the agent's prompt files, including memory, at any time.

Another important design choice is that **HiMe abstracts everything as a database**:

- the wearable data stream is a database;
- Memory Markdown is a database;
- past trajectories and activities are databases;
- scheduled tasks are databases;
- agent-generated personalized pages, and the backend data used by those pages, are also databases.

For databases the agent can write to, we still describe principles rather than imposing concrete constraints. How to create databases, how to organize memory, what to use as keys, what schemas to define: these are all left to the agent. In effect, the whole process of memory refinement, indexing, and retrieval is delegated to the model. A weak LLM can build tables in the simplest possible way, using timestamps; a strong LLM has much more room to operate.

Beyond flexibility, another benefit of treating everything as databases is that the agent no longer has to treat context sources as separate, heterogeneous systems. It can freely query, combine, and cross-check different kinds of information.

For strong models, this makes context preparation very natural: they can read all data through a unified interface and build connections across sources.

For example, the agent can query the memory table to see whether the user has fitness goals, then query a generative page, perhaps a workout-tracking dashboard, to read statuses recorded by the user, and finally query the user's wearable data to understand heart-rate changes. It can then combine all of this information into a reasonable suggestion.

## Cache Everything

HiMe is designed to make personal health agents as cost-effective as possible for everyone. Cost had to be considered from the very beginning.

> Cache every reusable component, minimizing both users' token cost and TTFT.

In early designs, we added many dynamic placeholders to the system prompt so the agent could better perceive the current context, but this broke the KV cache. In the final version, we removed all placeholders and let the agent observe context from observations instead of forcing it into the prompt.

All agent loops are organized as multi-turn dialogues. New messages are append-only.

Different agents need different system prompts. We split the system prompt into atomic components, then assemble them in ordered layers for different agents, maximizing reuse of the shared prompt prefix across agents.

The agent's coding tool is based on a Jupyter notebook, which is another form of cache. It does not cache prefixes; it caches variables produced by executed code blocks. The agent therefore does not need to rewrite analysis code in the next tool step. It can directly reuse variables from the previous step and continue the analysis. We also found that this makes the agent more willing to analyze data in depth.

## The Most Important Point

> Ignore all the summaries and lessons above.

The best lessons always come from designing an Agent yourself. Learn from on-policy trials. **Humans are the best self-evolving agents.**

## Several More Things

There are also a few thoughts unrelated to harness.

### Why did we design a pixel cat HiMeow in HiMe?

It was not to imitate Claude Code's little crab and give the product a mascot. We wanted to make the idea of a "digital twin" tangible. We hope a Personal Health Agent can be an avatar of the user, unlike most assistants, which remain assistants. An assistant is "it"; an avatar is "me." When the user's wearable data stream shows a healthy state, HiMeow is healthy too, and vice versa. If the user wants to take care of the cat, they are also taking care of themselves.

Current iOS restrictions around health data are very strict, which makes fully real-time background synchronization difficult. This is why both open-source and closed-source Health apps tend to make certain design compromises. From Apple's perspective, this is a reasonable choice for power consumption and privacy. But if we want users to get the most seamless and timely transfer experience possible, the best approach is to encourage them to open the app and trigger foreground transfer. That is why we designed HiMeow: whenever the user comes back to look at it, all data transfer is triggered and completed at that moment. The process is almost instant, and the user also gets the freshest information.

### Why care so much about the agent's token consumption?

This has also been discussed a lot recently. A fun fact: priced at API rates, my 1.7 billion tokens would cost roughly 1,500 dollars. In reality, I only paid for one month of the Claude Code Plan, 5x Max, and I did not even spend that entire month on HiMe. Under the coding-plan model, the cost was compressed to around 1/15 or lower. This may suggest that the coding-plan model will not last too long. In the long run, token prices will likely move back toward the value they create.

When we build AI agent products, we are using tokens to create value for users. HiMe therefore needs to maximize cost-effectiveness and reduce cost below the performance threshold where user experience starts to suffer.

Across the whole vibe-coding process, the parts that took the most time were actually outside Agent Harness, such as iOS app development and HealthKit data-stream processing. This was not because the code volume was different, but because I was unfamiliar with those two areas. Most of the time went into repeated debugging and explaining requirements. Stronger coding agents did not seem to help much in domains I did not understand; if anything, they amplified my unfamiliarity.

When I installed HiMe for the first time and went out for a walk, I could directly feel health data continuously appearing. I could really feel myself, as an individual, producing a data stream. These streams are different from the human knowledge that large models acquire from the internet. They are another kind of data asset, one that can be used by the user themselves. Most agents process existing assets, such as turning a user's knowledge or requirements into a PPT, a survey, or a piece of code. Agents like HiMe ask a different question: how can we extract personal assets from personal data streams, even when the user does not yet know what they need? There are still many technical gaps here, but it is also a promising direction for the future.

{% endlang_content %}

{% lang_content zh %}

# Personal Agent 的 Agent Harness

这篇文章总结了我在开发 **HiMe** 过程中，对于 **Agent Harness** 的一些实践性理解与反思。HiMe 是我最近在做的个人健康智能体。这是第一个理解实时穿戴设备数据流、提供健康支持的开源智能体项目，从数据管道、Agent Harness、iOS/Apple Watch App，全部开源。

这也是我第一次较大规模使用 Claude Code 进行开发，总共用掉了 Claude Code **1.74B tokens**，大部分是 Opus 4.6：

- **uncached input:** 370,000+ tokens
- **output:** 9,350,000+ tokens
- **cache write:** 57,200,000+ tokens
- **cache read:** 1,670,000,000+ tokens

总共 222 commits，最终 HiMe v1.0.0 版本有 50,000 多行代码。欢迎大家在 [GitHub](https://github.com/thinkwee/HiMe) 上关注，试用，提供反馈。

关于什么是 Agent Harness，这里不再展开，可以简单理解为：

> 在一个 LLM Agent 产品中，除大模型本身之外的所有系统性支撑代码都是 Harness。

它本质上是开发者为“让模型真正跑起来”所付出的全部工程努力。随着基础模型能力的持续提升，我们对 Agent 的预期也在不断提高，相应地，Harness 的复杂度也在上升，然后被基模内化（所谓的 _LLMs Eat Scaffolding for Breakfast_），然后我们对 Agent 有更高的期待，出现更为复杂的 harness，如此循环。**工程师和模型在这个循环中 Co-Evolution。**

一个有意思的视角是：我在 harness Claude Code 来构建 personal agent，而最终目标是让用户 effortless 地 harness 这个 personal agent。

> 这种“控制权的转移”其实决定了很多设计选择。

这 1.74B 的 tokens 主要花在了这两层 Harness 上。对于如何使用 Claude Code 的 best practice 已经数不胜数，而本文主要介绍后者，即如何设计好一个 personal agent，让其能够被用户所 harness，让用户有最佳体验：

> 对于 Personal Agent，用户对于 Agent 的期望是什么，又需要怎样的 Harness？

我总结了几个在开发过程中学习到的 lessons，几乎全是产品设计和工程上的一些考量，不涉及算法问题。

## Harness 提升弱模型的下限，不给强模型设限

不是所有用户在所有场景下都一直用最强的模型。一个有效的 Harness，应当在弱模型条件下提供可靠兜底，同时不给强模型施加不必要的结构性约束。

> 它的职责是**定义环境，而非规定路径**。

我犯的第一个错误就是为了让弱模型能 work，不断地调试 prompt，试图将 Agent 驯化为 workflow：将 system prompt 写死成某种流程，将 tool instruction 写得非常详细，乃至指定了其在 workflow 里的特定用法。而最终的结果往往是两头落空：

- 弱模型依然不断冒出新的问题；
- 强模型则被压缩成了执行固定脚本的引擎。

> workflow 是把未来压缩成过去，试图用已经见过的路径，去覆盖所有未发生的情况。但 Agent 面对的，是一个开放环境。

告诉模型**原则**，而不是具体做什么：在 HiMe 正式版我删除了所有的 workflow 描述以及 in-context examples，以及所有的 "don't do" 式具体例子，仅仅描述 Agent 职责。现有的模型在一般 QA 里能够很好地 follow in-context examples，但在 Agent 这种长程任务以及超长启动 prompt 的场景下，模型常常能力弱化，倾向于记忆 in-context examples 而不会泛化。

我们不如把这个反馈交给用户，让 agent 利用反馈来修正自己的行为，而不是一开始就写死具体的 Agency。只对确定性的东西固化。对于任何有发挥空间的原则，只固化那些确定性的内容，比方说 API 库。

如果一次 agent session 的结果没有达到预期，并不着急调整 prompt，而是先检查：

- 给 agent 提供的工具是否完备；
- 给 agent 提供的环境反馈是否完备；
- 如果一个 Agent 在“理想条件下”都无法完成任务，再考虑 prompt 所定义的任务难度是不是超出了模型能力。

HiMe 会把 context 处理为两类，给弱模型的精加工的 context，以及保留更原始的信息量更全的 context 访问入口。例如对于穿戴数据的访问，既有能 cover 80% 场景的廉价路径（给 Agent 一个预存了近期健康数据的 dataframe 直接操作），也有 escape hatch（Agent 可以选择读取全量数据）。

> 为 Agent 设计环境，为用户设计 Fallback 策略，为两者之间的沟通设计 Feedback 机制。

在这套原则下，我的大多数时间都是在维护弱模型，但当在弱模型上调出一个比较好的效果时，切换到强模型，它常常能够带给我惊喜。

## 超越静态测试

在一开始 HiMe 并没有单元测试，但很快就出现了问题：不断增加的新功能新代码涌入，扰乱了项目的结构，引入了大量需要返工的破坏性修改。然后我写了静态的单元测试。Claude Code 其实对于测试的支持非常好，它会在每次自己写完代码之后执行测试，确保写入的代码至少不破坏已有的功能。

但是后来我发现，对于一个 Agent 项目，仅仅静态测试只能 cover 很小一部分机制性的问题，例如 LLM Provider 超时的回退；但真正影响体验的，往往是交互层的问题，而这些只有在“有人用”的情况下才会暴露。

因此我引入了 **User Test**，让 LLM 模拟 users。

LLM Agent 和 LLM 用户有完全不同的感受野：

- 用户不需要理解，也不可能审计 Agent 内部是哪一个具体步骤出了问题；
- 用户只能看到 Agent 发来的消息或者其产出的 artifact；
- 用户只会根据自己能看到的部分反问、追问，或者仅仅表达情绪。

> 用户不是 Linter 或者 Debugger。

我让 LLM 模拟不同人格的用户，对 Agent 进行压力测试，发现了许多问题，例如静默的反馈链，对于频繁消息的处理等等。

## Multi-Agent，一种工程实践的需要

在如今茫茫多的 LLM Multi-Agent 论文里，我们常常可以看到关于 Multi-Agent 更好还是 Single-Agent 更好的讨论，或者是关于 Multi-Agent 为什么更好的理论分析。而在 HiMe 的实践里，为什么用 Multi-Agent 是非常简单直白的：不是为了追求更好的指标效果，而是工程考虑。

> Agent 本身就是一种职责的抽象。

一个好的 Agent 就和一个好的 Python 文件一样，只负责一个职责且边界清晰。

当一个 Agent 承担过多职责时，问题会迅速积累：监控变得困难，fallback 难以设计，错误也难以归因。

HiMe 采用了 **sub-agent as a tool** 的设计：

- **Chat Agent** 是唯一主 Agent，负责所有和用户的交互；
- **analysis agent** 负责只读数据分析；
- **management agent** 负责记忆读写和其他 HiMe 框架系统性操作。

每一个 Agent 有自己的工具集，读写权限分离。主 Agent 可以把子 Agent 当作工具调用，只需要给出目标，子 Agent 自行完成多步操作返回结果。

从性能上考虑，即便对于强模型来说，Single-Agent 和 Multi-Agent 差别可能不大；但对于弱模型来说，这对于提高下限能力非常重要，清晰的职责可以降低模型在每一次运行时的心智负担。

虽然是 Multi-Agent，但是 HiMe 并没有做并发，因为对于 personal health agent 的场景，重要的是长期稳定地支持用户，而不是像 coding agent 一般最大化产出 artifact 的效率和质量。相比之下，一个可控的 agent loop 更接近一个状态机，而不是一个分布式系统。

## 对 LLM 保持最大的不信任

我们不应该对于概率模型相信其绝对会实现某种预期，在 personal health 的场景里这尤为重要。

> Prompt 会漂移，code 不会。

任何能在代码里表达的约束，不要指望 prompt。尽可能减少不确定性。

设计多层的 fallback，不同维度的 fallback。HiMe 里包含了 **12 条 fallback 链路**，例如：

- 工具调用失败；
- 超 context；
- provider 过载。

按照不同的语义有不同的 retry 和 fallback 策略。

任何可能陷入循环的系统都要有 breaker。

要为 Hallucination 负责。大部分的 agent 会采取 web retrieval 的方式来保证其 response 是 factual 的，但对于 personal agent 而言更常见的是 **unfaithful** 的问题，例如 agent 没有执行任何工具调用，却谎称其完成了调用然后虚假分析。

HiMe 设计了一个审计 agent，审计 Chat Agent 发给用户的每一条消息，结合其工具调用链，检查其消息中的定量或者定性分析是否都有用户的真实健康数据 query 结果作为 evidence 支撑。

持久层（数据、队列）必须比应用层（agent loop）活得久。personal agent 的数据是用户的真实资产，不能因为 agent 的短暂异常而丢失。

所有 agent 执行的带持久化副作用的操作必须由用户明确发起的 chat 路径触发。

## LLM is not designed for Real Proactive

Personal Agent 一个特点就是个人用户在不断产生数据，每一个人都是行走的数据流。这些数据流是没有标签的、noisy 的，且信息密度低的。最重要的是，LLM 并不是为了处理流式数据设计，它对每一个输入期待产生一个输出。想要做到真正的 proactive，则需要 LLM 本身能够在接受数据流里每一个 token 时就决定是否要 output token，这对于当前的 LLM 并不是内置的能力（除非像全双工的语音 LLM 进行特殊的训练）。

在 Personal Health 的场景里，这样的数据流就是可穿戴设备数据以及其他健康数据。直接将这些数据流输入 LLM 也是一种做法，但如上所述，这并不是当前通用 LLM 内置的能力。HiMe 将 LLM Agent 置于数据分析师的角色，而不是数据分析工具，这样能够充分利用 LLM 的知识，却失去了真正处理流式数据的能力。我在一开始尝试过，让 LLM 读取数据，根据当前的 personal context 和数据的 observation 来自行决定 sleep 多久（可随时打断）之后再进行下一次数据读取，但效果很不理想。

最终 HiMe 采取了**用户和 Agent 共建**的形式实现 Proactive 功能：

在个人健康场景，真正的实时数据流理解和警报不应该留给 LLM 这么重的分析引擎，而是用其他轻量的专业工具来分析，保证实时分辨率。

在需要 LLM 的场景，更重要的是**个性化的 proactive，而不是更实时的 proactive**。本质上 proactive 就是一个在每个 observation 到来时是否要采取某种 action 的决策，在 HiMe 里我们采用一种用户和 Agent 协作、分层固化的形式实现：

- 不固化，让用户决定何时要执行分析，这其实就是 reactive。
- 用户可以让 agent 将一些分析固化为定时任务。
- 更进一步，这些任务 agent 可以进一步 refine 为事件触发的形式，例如血氧过低时触发，连续几个小时没有数据时触发等等。

以上也是大部分 agent 产品会实现的两种 proactive tasks，HiMe 还实现了 Generative Pages，即 agent 为用户写出新的 app 页面，以新的方式和新的时机与数据流交互，这也是最高自定义程度的 proactive analysis。

## Generative App

说到 HiMe 的 generative page，其初衷是想要实现 generative app 的效果：**Agent 实时地为用户创建新的 app 功能**，例如在 HiMe 里，用户可以要求 agent 创建一个跑步配速记录仪或者瑜伽锻炼记录面板等等。

但 HiMe 实际上并没有改变 App。实时改变 App 并不是一种安全的做法：App 在上架分发之后其功能不应该随着 AI 的引入再改变。因此 HiMe 实际上是在用户自己 host 的 server 端让 Agent 生成新的页面，前后端的读写都是在 server 端，不改变 App，App 只是承担浏览器的功能。

但是这样已经让未来的 agent 产品非常具有想象力。最大的个性化就是让用户自己实现需求。许多 Agent 产品的功能缺失实际上是 context 的缺失，例如对于个人健康 Agent，如果用户自己不填写，则 Agent 永远不知道用户今天是否有补充维生素，完成了多少量级的举重：穿戴设备无法提供这些信息。但是每一个用户可以让 Agent 完成一个界面，让其输入这些信息。这些数据将会自然而然融入用户的个人数据流，为 Agent 提供更好的 context。

## Generative Databases as Memory

HiMe 并没有设计复杂的记忆机制，这是未来还需要探索的一个领域。当前的所有的高阶 Agent 记忆构建方式和检索方式都是 case by case 的工程优化，很难直接迁移到 personal health agent 的场景。

> memory 不仅仅是 context recall。

在 personal agent 的场景下，memory 应该是对于用户的终身建模：一个人是其上下文的集合，一段 memory 就是对于 agent 当前 session 有用的上下文，也就是人的一部分。

对于长期记忆，例如用户的偏好，或者 Agent 在实践中能够复用的经验，HiMe 采用了 Prompt Markdown 的形式，让 Agent 去读写 Markdown 文件，并接入其 system prompt construction 中。这样至少能够让人类可审计、可更改。在 HiMe 的 dashboard 上，用户是可以随时去热更新 Agent 的 Prompt 文件，包括记忆。

另外一个重点是，**HiMe 将所有的内容都抽象为数据库**：

- 穿戴数据流是一个数据库；
- memory markdown 也是数据库；
- 过去的 trajectory/activity 是数据库；
- 定时任务是数据库；
- agent 生成的 personalised pages，以及这些 pages 用到的后端数据，也都是数据库。

对于 agent 有写权利的数据库，我们依然保持原则性描述而不做具体约束。例如如何创建数据库、如何规划记忆，以什么作为 key，有哪些 schema 等等都交给 agent，相当于记忆的整套 refine、indexing 和 retrieval 都交给了模型自己把控。弱 LLM 可以以最朴素的 timestamp 来建表，而强 LLM 则能有更大的发挥空间。

除了自由度，将一切统一为数据库的另一个好处是，agent 不再区分各种异构的 context 来源，它可以随意地查询、混合、交叉验证各类信息。

对于强模型来说，在准备自己的 Context 时，它可以非常自然地统一读取所有数据，建立关联。

例如 Agent 可以查记忆表判断用户是否有健身的需求，然后查 generative page（可能是一个锻炼记录 dashboard）读取用户自己记录的各种状态，最后查用户的穿戴数据了解心率变化，综合所有这些数据给出合理的建议。

## Cache Everything

HiMe 的设计是为了让所有人以最高性价比使用 personal health agent。成本从一开始就是一个需要考虑的因素。

> HiMe 希望 cache 所有可以复用的部分，最大程度降低用户的 token 成本和 TTFT。

在初期设计里，我们在 system prompt 里加入了很多 dynamic placeholder 以让 agent 更好地感知当前上下文，但是这样破坏了 KV cache。最终版本里我们删掉了所有 placeholder，让 agent 从 observation 里观察上下文，而不是强行注入。

所有的 agent loop 都组织为 multi-turn dialogue 的形式，新的 message 只 append。

不同的 agent 需要不同的 system prompt。我们将 system prompt 拆分成多个原子部分，然后分层保持顺序组装到不同的 agent 上，最大程度利用所有 agent 的共有 prompt head。

agent 的 coding 工具采用了 Jupyter notebook 的形式，这也是另外一种形式的 cache：它不是 cache 前缀，而是 cache 已经处理过的代码块执行后保存的变量。这样 agent 不用在下一个 tool step 重新写分析代码，可以直接复用上一步的分析结果变量继续分析，而且我们发现这样 agent 更倾向于深入分析数据。

## 最重要的一点

> 无视掉以上所有总结和经验。

最好的经验永远来自于自己亲自设计一个 Agent。Learn from on-policy trials，**人是最好的 self-evolving agent。**

## Several More Things

此外还有一些与 harness 无关的感受。

### 为什么在 HiMe 里我们设计了一只像素猫 HiMeow？

这并不是为了模仿 Claude Code 的小螃蟹提供一个吉祥物，而是我们想要具象化“数字孪生”的体验。我们希望 Personal Health Agent 是用户的一个化身，而不像其他的 Assistant 都是助手。助手是它，而化身是我。当用户自己的穿戴数据流表现出健康的状态时，HiMeow 也会健康，反之亦然。如果用户有兴趣照顾好这只猫，同时其实是在照顾好自己。

目前 iOS 的限制对于健康数据的把控非常严格，很难做到完全即时的后台同步。这也是为什么当前能看到的无论是开源还是闭源的 Health 类 App，都在设计上做了一定程度的取巧。从苹果的角度这是为了控制功耗和保护隐私，是非常合理的决策。但如果要让用户尽可能获得无感的及时传输体验，最好的方法就是促使他打开 App 进入前台传输。所以我们设计了 HiMeow：用户只要什么时候回来看一看它，此时就触发并完成了所有数据传输。这个过程几乎是瞬间完成的，同时也能让用户拿到最即时的信息。

### 为什么要节省 Agent 的 Token 消耗？

这也是最近讨论比较多的一点。一个 fun fact 是：我的 1.7 billion tokens 的账单按照 API 价格折合计算大约在 1500 美元左右。但实际上，我只花了一个月的 Claude Code Plan (5x Max），甚至这一个月并没有把所有时间都用在 HiMe 这个项目上。所以，如果按 coding plan 来计算，成本被压到了原来的 1/15 甚至更低。这可能说明：coding plan 模式不会持续太久。未来的 token 价格还是会尽可能地回归到其创造的价值上去。

当我们做 AI agent 产品的时候，也就是在利用 token 为用户创造价值。因此 HiMe 要尽可能去最大化性价比，在不影响用户体验的性能阈值下，去降低成本。

在整个 vibe coding 的流程里，我花的最多时间的是 Agent Harness 以外的部分，例如 iOS App 的开发和 HealthKit 数据流的处理。这并不是因为代码量不同，而是我本身就不熟悉后两者，将大部分时间花在了反复 debug 和描述需求上。更强的 Coding Agent 貌似并没有在我不熟悉的领域帮助到我，反而是放大了我的生疏。

当我第一次装上 HiMe 之后，出去走一走，会直观感受到不断地有健康数据冒出来，我会真实地感受到自己作为一个个体在产生数据流。这些数据流不同于大模型在互联网上获取到的人类知识的期望，它是另一片能够为用户本人所用的数据资产。大部分 Agent 都是在加工资产，例如把用户的知识和需求文本转成 ppt/一段 survey/一份代码，而 HiMe 之类的 Agent 是在考虑如何从个人数据流中提取个人资产，用户自己甚至都不知道需求是什么。这里目前还存在很多技术上的 gap，但也是未来 promising 的一个方向。

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
