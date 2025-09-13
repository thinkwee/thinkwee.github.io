---
title: Towards General Reasoning
date: 2025-09-13 10:56:16
categories: LLM
tags:
  - inference
  - math
  - reasoning
  - llm
  - rl
  - agent
  - questions
refplus: true
mathjax: true
---

<img src="https://i.mji.rip/2025/09/13/6afdb242505675e387a7e2498a292346.png" width="600"/>

How far have we gone towards general reasoning? How far are we from the general reasoning?
<!--more-->

{% language_switch %}

{% lang_content en %}

<img src="https://i.mji.rip/2025/09/13/fc1d7e4647d71b982f9ad9a0cbbbeaa0.png" width="600"/>

Recently our paper *NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning* was accepted to EMNLP. NOVER uses the LLM's perplexity of the ground truth conditioned on the reasoning trajectory as the reward, which extends the RLVR paradigm beyond math and code, enabling learning of general reasoning on arbitrary text-to-text tasks without extra models or verifiers.

<img src="https://i.mji.rip/2025/09/13/d82968f183f79eeca1606f08b6fcf9f6.png" width="600"/>

When we began NOVER in February, most RLVR work focused on mathematical and code reasoning; RLVR research that targets general or hard-to-verify domains was scarce. Nearly six months later many interesting related papers have emerged. Due to limited resources, many ideas in our experiments were not fully validated and were left out of the paper. This post organizes those ideas and surveys recent relevant work to assess how far we have come on general reasoning and how far we still must go. 

# What

<figure>
<img src="https://i.mji.rip/2025/09/13/b019a44cb56556aa079fdfd7c182c4d2.png" width="1000"/>
<figcaption>NOVER extends RLVR to general reasoning like Science Explanation and Proof, Social Reasoning, Creative Writing and Translation</figcaption>
</figure>

NOVER’s target problem is, for tasks whose answers are unstructured natural text and therefore unsuitable for rule-based verifiers, how can we apply RLVR to acquire reasoning ability?

Such cases are common. A math problem often has a unique canonical answer that can be placed in a `\boxed{}`, while explanations of physical or chemical phenomena can legitimately take many different forms (think of long, opinionated answers on Quora-like sites). For entity-level QA we can use exact match, but for long-form generation such as document-level summarization/translation/creative writing, there is no reliable rule (ROUGE, BLEU and similar metrics have long been shown to be unreliable). The same applies across vertical domains (medicine, law, social sciences, psychology, literature, education) where many ground-truth instances are free-form.

A few concepts are easy to conflate; NOVER focuses only on the following: 

- the ground truth exists (this is not unsupervised learning); we do not desire a reward model to directly give a judgment, but rather a verifier that compares ground truth and prediction; 
- the ground truth may be subjective or objective, but the dataset contains at least one reference. 
- Even when the ground truth is objective, rule-based verifiers are often failed: objective GTs can be expressed in many textual forms, otherwise we would simply use exact match as the reward. Moreover, even for math, where answers are seemingly easy to verify (multiple choice labels, numbers, formulas, short text, booleans), model responses vary widely and rule-based verifiers are not robust {% ref compass_verifier %}.

<figure>
<img src="https://i.mji.rip/2025/09/13/7624446cf9788e075ec0a68461713615.png" width="600"/>
<figcaption>Error patterns on easy-to-verify tasks from compass verifier {% ref compass_verifier %}</figcaption>
</figure>

# Why

Why pursue general reasoning? Many non-math/code tasks (creative, humanities, or long-form scenarios) appear to be not a suitable targets for a “reasoning” model. But consider:

- It is still unknown whether RLVR is the ultimate correct paradigm for training reasoning models.
- It is still unknown whether CoT genuinely corresponds to human-style reasoning.
- It is still unknown whether CoTs learned via RLVR truly represent a model’s internal reasoning.

Currently RLVR is better seen as a method to train native CoT outputs, and CoT is simply “say something before the final answer.” That something is not necessarily reasoning (some works find that even repeating the question can increase accuracy); it is model-generated tokens that, when produced before the answer, help the LLM better exploit learned patterns and increase the prediction probability of the next correct answer tokens. 

<figure>
<img src="https://i.mji.rip/2025/09/13/e066a152bb031bf47c99411fb95f17f6.png" width="400"/>
<figcaption>DataAlchemy's show that CoT's gains can arise from reuse and interpolation of patterns near the training distribution {% ref is_cot_mirage %}</figcaption>
</figure>


From that practical viewpoint, producing a bit more text that improves answer quality is no harm (users often try a “deeper thinking” mode in chat systems expecting better answers). 

Another reason to study general reasoning is that, **for an LLM, task difficulty is tied to verification difficulty**. We aim to keep pushing the frontier of problems that models can solve: some tasks that are hard for humans (e.g., olympiad math) might still be learnable by models if supplied with a correct, sufficiently informative reward. Conversely, tasks whose rewards are hard to formalize are harder for models to learn. 

# What it actually is

What we need for RLVR on free-form text is a good verified signal, which is actually a reward function that measures semantic agreement between ground truth and model prediction. That is exactly what we pursue in the Natural Language Generation. The most basic target is cross-entropy (ppl). **From this perspective NOVER essentially moves the SFT loss into the RLVR setting, and recent work shows SFT and RL differences are often not large.**

Although NOVER used ppl, perplexity may not be optimal. We can arrange verified signals along an axis from fine to coarse granularity: the coarser the signal, the more information is lost and the sparser the reward becomes. On this axis three main approaches appear:

- Perplexity-based signals.
- Rubrics / checklists.
- Trained verifier models that yield binary (0/1) rewards.

<figure>
<img src="https://i.mji.rip/2025/09/13/59636a40c655c608a6733bb59d27039c.png" width="800"/>
<figcaption>The Axis of Verified Signals</figcaption>
</figure>

Compared with binary rewards, ppl provides a denser signal, extends naturally to free-form text, and avoids reward saturation; but it loses the absolute correctness signal, i.e., the model never observes a strict correct/incorrect label and we cannot use pass@k-style metrics to assess sample difficulty. Rubrics/checklists sit between these extremes: they are more fine-grained than binary rewards but still sparser than ppl. High-quality rubrics typically require sample-wise, human expert annotation. Several recent works explore rubric-style solutions{% ref ace_rl checklists_are_better ticking_all_the_boxes rubric_anchors rubrics_as_rewards %}. Baichuan-M2 in particular develops a fairly detailed Verifier System that functions as a model-driven environment, with a Patient Simulator (data generator) and a Rubrics Generator (rollout evaluator) {% ref baichuan_m2 %}.

<figure>
<img src="https://i.mji.rip/2025/09/13/6a376729962f2a3b2ae838d70e7aa97f.png" width="600"/>
<figcaption>Baichuan-M2's Verifier System{% ref baichuan_m2 %}</figcaption>
</figure>


Rubrics also enable controlled synthetic data generation for debiasing reward models{% ref robust_reward_modeling  %}, so the reward model focuses on true causal factors and resists hacks stemming from format, length, or tone. OpenAI’s Deliberative Alignment can be seen as an outcome-RL approach that uses safety-oriented rubrics {% ref deliberative_alignment %}.

# How

<figure>
<img src="https://i.mji.rip/2025/09/13/1ae265ffdbc0c5855689ef5c6b64c909.png" width="400"/>
<figcaption>NOVER's reward is derived from the policy model's conditional ppl of the ground truth given the reasoning trajectory</figcaption>
</figure>

NOVER applies a crude but direct approach: for a rollout, compute the policy model’s conditional ppl of the ground-truth answer given the rollout's reasoning trajectory as the reward.

<figure>
<img src="https://i.mji.rip/2025/09/13/2746aac29b4bcba4045197579be651f6.png" width="600"/>
<figcaption>The idea of reasoning advantage (RA).</figcaption>
</figure>

The idea of reasoning-ppl based improvements has appeared before. A short NeurIPS 2024 LanGame workshop paper called this notion reasoning advantage (RA), essentially the relative change in reasoning ppl compared to a no-reasoning baseline. That paper used RA for data selection, which is essentially keeping CoT examples with high RA for SFT, so it can be viewed as an offline-RL style method {% ref on_reward_functions %}.

Fortuitously, I experimented with relative reasoning ppl in NOVER and later found the LANGame writeup: it is an intuitive and reasonable design. 

<figure>
<img src="https://i.mji.rip/2025/09/13/165b17d45b447fe0d39c0960d5557f99.png" width="600"/>
<figcaption>The idea of longPPL.</figcaption>
</figure>

Another related refinement on ppl is longPPL which measures ppl on a context-dependent subset of tokens: longPPL subtracts the ppl without long context from the ppl with long context, thereby focusing evaluation on tokens that truly depend on long-range context {% ref what_is_wrong_with_perplexity %}. RA shares the same spirit: we want the reward to come from those tokens in the ground truth that genuinely require CoT reasoning. 

More Interestingly, in GRPO a simple group normalization makes relative ppl improvements and absolute ppl effectively equivalent on advantage calculation, so absolute reasoning ppl itself is a solid reward signal.

But applying ppl directly has issues. 

- First, ppl is numerically unstable: advantage estimates vary across batches and exhibits length bias. NOVER converted ppl into in-group quantiles to produce more stable rewards. QRPO applies quantile transforms more rigorously: it maps rewards to quantiles of the base-policy reward distribution across the dataset, making the partition function tractable and enabling numerically stable pointwise rewards even in offline RL {% ref quantile_reward_policy_optimization %}.
- Which model should be used to compute ppl? In principle a stronger external model could be a more accurate verifier, but the gap between large and small model cause problems, which is similar to bad distillation results when using DPO to train small models from GPT-distilled labels. NOVER uses the policy model itself to compute ppl, which saves extra models and eases scaling. We found that using a separate large verifier (closed-source SOTA or a specialized verifier) often leads to LM-hack-LM issues, whereas using the policy model’s own ppl yields smoother learning curves.
- With small batches and limited compute, training is unstable. NOVER introduced a policy-proxy sync: periodically copy policy parameters to a proxy model and compute ppl from the proxy during training. This effectively increases the batch size (similar in spirit to gradient accumulation) and stabilizes reward estimates.

<figure>
<img src="https://i.mji.rip/2025/09/13/55663a25c9fb5ef4fd4d482dd0508251.png" width="400"/>
<figcaption>RLPR shows that ppl can accurately measure the reasoning advantage.</figcaption>
</figure>

Several contemporaneous works adopt related ideas but differ in how they stabilize ppl numerics. 

- VeriFree {% ref reinforcing_general_reasoning %} uses reasoning ppl directly, but restricts to short answers (≤7 tokens) where ppl is less unstable, and shows ppl can approach or exceed verifier-based baselines on short QA. 
- RLPR {% ref rlpr %} uses relative token probabilities (the per-token mean probability, clipped, then advantage computed) rather than ppl and provides detailed ablations showing direct ppl can lose 20 points if used naively. 
- DRO {% ref direct_reasoning_optimization %} targets long answers and uses relative reasoning ppl with per-token weighting for high-variance ground-truth tokens and local weight decay. 
- DeepWriter {% ref reverse_engineered_reasoning %} focuses on long-form writing but uses reasoning ppl purely as a scoring metric to filter and iteratively rewrite drafts (not an RL loop), avoiding numeric instability by staying in a supervised selection regime.

# Observations

<figure>
<img src="https://i.mji.rip/2025/09/13/b6a2baa9c386467e6e30989a3390adf8.png" width="600"/>
<figcaption>Collapse modes in training.</figcaption>
</figure>

We experienced many collapse modes early in training: completion lengths exploding, ill rollouts where the model produces garbled text, and simultaneous blowups of format rewards. We applied the tricks above to stabilize training (see the paper’s ablation for details on the “curse of proxy”).

<figure>
<img src="https://i.mji.rip/2025/09/13/bbbd4be0ecaebc5827a9b09efa82bbcc.png" width="600"/>
<figcaption>The curse of proxy.</figcaption>
</figure>


A small but useful trick is reward dependency: when multiple reward terms are simply summed the model can be uncertain which objective produced a given penalty or bonus. Practically, we found it effective to gate task rewards on a strict format reward: unless the format reward is satisfied, set all other rewards to zero. When the format reward gained, the model is usually “sane”, no hallucination or gibberish. This dependency can also pull the model back from training collapse.

We also found that excessive strictness on format rewards may hinder exploration {% ref simplerl_zoo %}. For example, one interesting reward hacking on format we observed was nested `<think>` tags in CoT: models can nest a sub-reasoning reflection inside an outer `<think>` block to game the signal, e.g.

```
<think>
  inner thoughts
  <think>
    reflection on the earlier thoughts
  </think>
  continue reasoning
</think>
<answer>
...
</answer>
```

Stronger base models exploit dense semantic signals better. For example, we converted multiple-choice questions into free-form answers where the model must output both the option letter and the full option text; comparing 7B vs 3B, the 7B model better leverages ppl to rank rollouts:

- rank 1: option letter and option text both correct
- rank 2: letter wrong, option text correct
- rank 3: letter correct, option text similar to another option
- rank 4: letter correct, option text completely wrong
- …
- lowest: everything wrong

Looking at rollouts beyond the answer, ppl can indirectly reflect differences in the reasoning details. In an astronomy example that required an explanation plus numeric computation, we asked GPT to analyze each rollout (reasoning plus result) sorted by reasoning ppl; the model’s qualitative analyses correlated with ppl rankings.

<figure>
<img src="https://i.mji.rip/2025/09/13/3c6a8ca4f84f7aa6d787a4121aed2482.png" width="600"/>
<figcaption>The correlation between ppl rankings and GPT's qualitative analyses.</figcaption>
</figure>

NOVER also partially works on non-Qwen models, though weaker bases (e.g., some Mistral checkpoints) show erratic behavior. Zero-shot CoT can be seen as an untrained base exploration strategy; if that baseline is close to or exceeds the base model, RL typically provides gains.

<figure>
<img src="https://i.mji.rip/2025/09/13/e346a72fe2821b7eb95045dde4fbd235.png" width="600"/>
<figcaption>NOVER partially works on non-Qwen models.</figcaption>
</figure>

We also observed (without exhaustive experiments) that many general-reasoning datasets are annotated by closed-source large models and thus are not perfectly objective or correct (loose definitions, symbol misuse). Perplexity can still provide a useful guiding signal: in some cases models learned complex reasoning patterns from the ppl signal that can produce arguably more correct answers than the original ground truth.

# Is changing only the reward enough?

<figure>
<img src="https://i.mji.rip/2025/09/13/8cfc464fbb6cc0c7bc08f82c929c5475.png" width="1000"/>
<figcaption>Some works on reproducing GRPO.</figcaption>
</figure>

No, but reward design is the most obvious gap when extending rule-based verification to general reasoning. What's more, from the bitter-lesson viewpoint many algorithmic tricks are spurious: data and compute dominate. By March many people were reproducing GRPO and noting its fragility; our NOVER training surfaced similar issues. Many algorithmic “tricks” proposed in these papers have marginal effects compared with data and scale.

So advancing general reasoning faces larger challenges in data and base models; algorithmic work will be required later to make training more efficient and stable.

- Data: existing general-reasoning datasets vary widely in quality; cleaning consumes substantial effort, and much data is LLM-annotated (distilled from GPT or similar) rather than human-curated. The data are static and finite. RL itself is sample-efficient in some senses; the cost-effective path to scaling is not simply more examples but higher-quality environments and feedback.
- Base model: the base model governs exploration in RL. Practically, it should already possess zero-shot instruction following and CoT capability; richer knowledge helps. Debates over whether RL can raise the ultimate capability ceiling are not the key point: post-training often elicits latent abilities rather than creates them. Some works already explore combining memory and elicitation, and I believe mid-training vs post-training may form new positive feedback loops.

# One more thing: Climb the Solver–Verifier Asymmetry

<figure>
<img src="https://i.mji.rip/2025/09/13/d53696cd2ce193f6011f24093c6e0f26.png" width="600"/>
<figcaption>The Solver-Verifier Asymmetry.</figcaption>
</figure>


A central concept in RLVR is the solver-verifier asymmetry: for some tasks verification is easier than solving, while for others verification is harder. Much of RLVR excels when verification is simpler than solving. The opposite side, where verification is harder, includes:

- General Reasoning with hard-to-verify free-form answers
- Situations requiring long time horizons to obtain a return (e.g., a business plan whose real feedback arrives after weeks, months, or years). Those cases resemble deep conversion problems in recommender systems: we need accurate attribution and systems that handle extremely sparse feedback.
- Scenarios that may require large human labeling efforts or hard-to-acquire real users to verify the solution, which motivates the development of effective user simulators.

The verifier-free design of NOVER introduces a new possibility (though not yet tested): 

**whether it is feasible to synchronize the intelligence of the policy model to the verifier model, thereby enabling co-evolution of solver and verifier along the Solver-Verifier Asymmetry diagonal**. 

A stronger policy model would lead to a stronger verifier model, which in turn could train an even stronger policy model. The key lies in the transmission of intelligence. NOVER’s design of using perplexity as the reward naturally **unifies the form of intelligence in both solver and verifier: both aim to increase the probability of generating the ground truth on good reasoning trajectories.** In this way, co-evolution can be achieved through standard RL without the need to design additional adversarial or self-play tasks. Here, the direction of intelligence transfer is from solving to verifying. A symmetric related work is LLaVA-Critic-R1, which found that a strong preference model can yield a strong policy model, though it required constructing an additional task. {% ref llava_critic_r1 %}.

If we want to achieve such fully automatic co-climbing, we have RL training which performs horizontal climbing (fix verifier y, improve solver x), we have Intelligence Sync which would perform vertical climbing (fix solver x, improve verifier y). However, we also need a third variable: tasks and data. Each point in the solver–verifier grid corresponds to specific tasks and datasets. As argued in my earlier post on [Scaling the Environment](https://thinkwee.top/2025/07/17/env-matrix/#more), beyond solver and verifier there is also the question generator. Most current reasoning-evolution work focuses on self-improvement via model consistency or entropy patterns; some approaches implement co-evolution of two modules, while a tri-evolution of three modules has not been explored:

<figure>
<img src="https://i.mji.rip/2025/09/13/e73159cd71bc40614799577a41c770f5.png" width="600"/>
<figcaption>The Trinity of Solver-Verifier-Generator.</figcaption>
</figure>


- R-Zero and Self-Questioning Language Models consider adversarial generation between a generator and a solver {% ref r_zero %}{% ref self_questioning %}. 
- URPO reframes verification as a solving task and unifies data training. COOPER trains a verifier from positive/negative samples constructed from current policy rollouts. Both lines implement solver–verifier co-evolution {% ref urpo %}{% ref cooper %}.

Another route to continual solver improvement is self-play: with a suitable environment, two solvers can game and thereby improve each other without worrying about asymmetry. For general reasoning such environments are hard to design because the “rules” are nebulous. Recent works have proved that models can learn rules {% ref llms_can_learn_rules %} and [combine atom skills to learn new skills](https://husky-morocco-f72.notion.site/From-f-x-and-g-x-to-f-g-x-LLMs-Learn-New-Skills-in-RL-by-Composing-Old-Ones-2499aba4486f802c8108e76a12af3020) through synthetic data and task, but existing real general-reasoning datasets are limited enumerations rather than comprehensive rule sets. This is still essentially static datset/benchmark-driven RL. In the AI “second half,” we should seek real-world environments and problems rather than static datasets.

Between static data and the real world lies a middleware: simulators. Simulators trade fidelity for feedback speed—like reward models or verifier models—and for general reasoning a useful simulator might look like a patient simulator in medical domains (see Baichuan-M2’s case), since real patients raise ethical and regulatory issues and validation can be slow {% ref baichuan_m2 %}.

A different idea is to forgo task-specific environments and instead play games: self-play on games could improve math and general reasoning if reasoning patterns transfer across games and tasks {% ref play_to_generalize %}{% ref spiral %}. If feasible, we could use game environments and self-play to continually evolve general-reasoning models.

# Citation
If you found the topics in this blog post interesting and would like to cite it, you may use the following BibTeX entry:
```bibtex
@article{general_reasoning_202509,
  author = {Wei Liu},
  title = {Towards General Reasoning},
  year = {2025},
  month = {9},
  url = {https://thinkwee.top/2025/09/13/general-reasoning/#more},
  note = {Blog post}
}
```

{% endlang_content %}

{% lang_content zh %}

<img src="https://i.mji.rip/2025/09/13/fc1d7e4647d71b982f9ad9a0cbbbeaa0.png" width="600"/>

以下为机器翻译，一个中文原生的版本请参考[公众号文章](https://mp.weixin.qq.com/s/ocpI3j3rwlt_9Zo1wYAF6Q)

最近，我们的论文  *NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning* 被 EMNLP 录用。NOVER 将 LLM 在推理轨迹下对真实数据的困惑度作为奖励，这一创新将 RLVR 范式从数学和代码领域拓展至更广泛的文本处理，使得模型能够在任意文本到文本任务中学习通用推理，且无需依赖额外的模型或验证器。

<img src="https://i.mji.rip/2025/09/13/d82968f183f79eeca1606f08b6fcf9f6.png" width="600"/>

当我们在二月启动 NOVER 项目时，大多数 RLVR 研究主要集中在数学和代码推理领域；而针对通用或难以验证领域的 RLVR 研究则相对匮乏。近六个月来，涌现了许多有趣的相关论文。由于资源有限，我们实验中的许多想法未能得到充分验证，因此未能纳入论文。本文旨在整理这些未充分验证的想法，并综述近期相关研究，以评估我们在通用推理方面取得的进展以及未来仍需努力的方向。

# 是什么

<figure>
<img src="https://i.mji.rip/2025/09/13/b019a44cb56556aa079fdfd7c182c4d2.png" width="1000"/>
<figcaption>NOVER 将 RLVR 拓展至通用推理领域，包括科学解释与证明、社会推理、创意写作及翻译</figcaption>
</figure>

NOVER 的目标问题在于，对于那些答案为非结构化自然文本的任务，如何应用 RLVR 来获取推理能力？

此类情况并不少见。数学问题通常有一个唯一的规范答案，可以放在 `\boxed{}` 中；而物理或化学现象的解释可以合法地采用多种形式（想想 Quora 等类似网站上的长篇、主观性答案）。对于实体级 QA，我们可以使用精确匹配，但对于长文本生成（如文档级摘要、翻译、创意写作），没有可靠的规则（ROUGE、BLEU 等指标已被证明不可靠）。这种现象在垂直领域（医学、法律、社会科学、心理学、文学、教育）同样存在，许多真实示例的答案为自由文本。

有几个概念容易混淆；NOVER 仅关注以下几点： 

- 真实答案存在（这不是无监督学习）；我们不希望奖励模型直接给出判断，而是需要一个验证器来比较真实答案和预测； 
- 真实答案可能是主观的或客观的，但数据集至少包含一个参考。 
- 即使真实答案是客观的，基于规则的验证器也经常失败：客观的真实答案可以以多种文本形式表达，否则我们只需使用精确匹配作为奖励。此外，即使是数学问题，其答案看似容易验证（多选标签、数字、公式、短文本、布尔值），但模型响应差异很大，基于规则的验证器也不可靠 {% ref compass_verifier %}.

<figure>
<img src="https://i.mji.rip/2025/09/13/7624446cf9788e075ec0a68461713615.png" width="600"/>
<figcaption>Compass Verifier在易验证任务中发现的错误模式 {% ref compass_verifier %}</figcaption>
</figure>

# 为什么

为什么追求通用推理？许多非数学/代码任务（创意、人文、长文本场景）似乎不适合“推理”模型。但考虑：

- 目前仍不清楚 RLVR 是否是训练推理模型的终极正确范式。
- 目前仍不清楚 CoT 是否真正对应于人类推理风格。
- 目前仍不清楚通过 RLVR 学习的 CoT 是否真正代表模型的内部推理。

所以当下我们可以将 RLVR视为训练原生 CoT 输出的方法，而CoT 只是“在最终答案之前说一些话”。这个“一些话”不一定是推理（一些工作发现重复问题也能提高准确性）；它是模型生成的在最终答案之前的tokens，帮助 LLM 更好地利用学习到的模式并增加下一个正确答案令牌的预测概率。 

<figure>
<img src="https://i.mji.rip/2025/09/13/e066a152bb031bf47c99411fb95f17f6.png" width="400"/>
<figcaption>DataAlchemy 证明了 CoT 的收益源于训练分布附近模式的重复和插值 {% ref is_cot_mirage %}</figcaption>
</figure>


那么从实际角度来看，产生更多文本以提高答案质量并无害处（用户经常在聊天系统中尝试“深度思考”模式以获得更好的答案）。 

另一个研究通用推理的原因是，**对于 LLM，任务难度与验证难度相关**。我们致力于推动模型可以解决的问题的边界：一些对人类来说很困难的任务（例如，奥林匹克数学）只要有足够的奖励，模型很容易学习。相反，任务的奖励难以形式化，对模型来说更难学习。 

# 本质是什么

我们需要在自由文本上进行 RLVR 的验证信号，这实际上是一个奖励函数，用于衡量真实答案和模型预测之间的语义一致性。这正是我们在自然语言生成中追求的。最基本的目标是交叉熵（ppl）。**从这一角度来看，NOVER 本质上将 SFT 损失转移到 RLVR 设置中，而最近的工作表明 SFT 和 RL 之间的差异通常并不大。**

尽管 NOVER 使用了 ppl，但困惑度可能不是最优的。我们可以沿着从细到粗的轴线排列验证信号：信号越粗糙，信息损失越多，奖励越稀疏。在这个轴线上，出现了三种主要方法：

- 困惑度为基础的信号。
- Rubrics/Checklists。
- 训练验证器模型，产生二进制（0/1）奖励。

<figure>
<img src="https://i.mji.rip/2025/09/13/59636a40c655c608a6733bb59d27039c.png" width="800"/>
<figcaption>验证信号坐标轴</figcaption>
</figure>

与二元奖励相比, ppl 提供了更密集的信号，自然扩展到自由文本，并避免了奖励饱和；但它失去了绝对正确性的信号，即模型从未观察到严格的正确/不正确标签，我们无法使用 pass@k 风格的指标来评估样本难度。评分/检查清单介于这些极端之间：它们比二元奖励更细粒度，但仍比 ppl 更稀疏。高质量的评分通常需要样本级、人工专家标注。最近的一些工作探索了Rubrics/Checklists解决方案{% ref ace_rl checklists_are_better ticking_all_the_boxes rubric_anchors rubrics_as_rewards %}. 特别是 Baichuan-M2 开发了一个相当详细的验证器系统，作为模型驱动的环境，具有患者模拟器（数据生成器）和评分生成器（rollout 评估器） {% ref baichuan_m2 %}. 

<figure>
<img src="https://i.mji.rip/2025/09/13/6a376729962f2a3b2ae838d70e7aa97f.png" width="600"/>
<figcaption>Baichuan-M2 的验证器系统{% ref baichuan_m2 %}</figcaption>
</figure>


Rubrics/Checklists 也促进了有控制的合成数据生成，以减少奖励模型的偏差{% ref robust_reward_modeling  %}, 这样奖励模型专注于真正的因果因素，并抵抗来自格式、长度或语气等hack。OpenAI 的 Deliberative Alignment 可以被视为一种 outcome-RL 方法，它使用安全导向的评分/检查清单 {% ref deliberative_alignment %}.

# 怎么做

<figure>
<img src="https://i.mji.rip/2025/09/13/1ae265ffdbc0c5855689ef5c6b64c909.png" width="400"/>
<figcaption>NOVER 的奖励是从策略模型在推理轨迹下对真实答案的条件困惑度</figcaption>
</figure>

NOVER 应用了一个粗糙但直接的方法：对于一个 rollout，计算策略模型在推理轨迹下对真实答案的条件困惑度作为奖励。

<figure>
<img src="https://i.mji.rip/2025/09/13/2746aac29b4bcba4045197579be651f6.png" width="600"/>
<figcaption>推理优势（RA）的想法。</figcaption>
</figure>

推理-ppl 基于的改进想法之前已经出现过。一篇短篇 NeurIPS 2024 LanGame 研讨会论文称这个概念为推理优势（RA），本质上是指推理困惑度与无推理基线的相对变化。该论文使用 RA 进行数据选择，即保持 CoT 示例，使其具有较高的 RA，以便用于 SFT，因此可以被视为一种离线 RL 风格的方法 {% ref on_reward_functions %}.

巧合的是我先在NOVER中尝试了相对reasoning perplexity的想法，然后才发现这篇有关RA的workshop论文：这说明相对提升的想法非常符合直觉。

<figure>
<img src="https://i.mji.rip/2025/09/13/165b17d45b447fe0d39c0960d5557f99.png" width="600"/>
<figcaption>longPPL 的想法。</figcaption>
</figure>

另一个与 ppl 相关的改进是 longPPL，它测量上下文依赖的子集令牌的困惑度：longPPL 从带有长上下文的困惑度中减去没有长上下文的困惑度，从而专注于那些真正依赖于长距离上下文的令牌 {% ref what_is_wrong_with_perplexity %}. RA 共享相同的理念：我们希望奖励来自那些在真实答案中真正需要 CoT 推理的令牌。 

更有趣的是，在 GRPO 中，一个简单的组归一化使相对 ppl 改进和绝对 ppl 在优势计算上有效等价，因此绝对推理 ppl 本身是一个 solid 奖励信号。

但直接应用 ppl 有以下问题：

- 首先，ppl 是数值不稳定的：优势估计在批次之间变化并表现出长度偏差。NOVER 将 ppl 转换为组量化，以产生更稳定的奖励。QRPO 应用量化变换更严格：它将奖励映射到数据集上基策略奖励分布的量化，使分区函数可处理，即使在离线 RL 中也能实现数值稳定的逐点奖励 {% ref quantile_reward_policy_optimization %}.
- 应该使用哪个模型来计算 ppl？原则上，一个更强大的外部模型可以是一个更准确的验证器，但大模型和小模型之间的差距会导致问题，这与使用 DPO 从 GPT 蒸馏标签训练小模型时的糟糕蒸馏结果类似。NOVER 使用策略模型本身来计算 ppl，这节省了额外模型并降低了缩放难度。我们发现使用单独的大型验证器（闭源 SOTA 或专用验证器）通常会导致 LM-hack-LM 问题，而使用策略模型的 ppl 产生更平滑的学习曲线。
- 在小批次和有限计算的情况下，训练不稳定。NOVER 引入了一个策略代理同步：定期将策略参数复制到代理模型，并在训练期间从代理计算 ppl。这有效地增加了批次大小（类似于梯度累积）并稳定了奖励估计。

<figure>
<img src="https://i.mji.rip/2025/09/13/55663a25c9fb5ef4fd4d482dd0508251.png" width="400"/>
<figcaption>RLPR 证明了 ppl 可以准确测量推理优势。</figcaption>
</figure>

许多同时期的作品采用了相关想法，但差异在于如何稳定 ppl 数值。

- VeriFree {% ref reinforcing_general_reasoning %} 直接使用推理 ppl，但限制为短答案（≤7 个令牌），其中 ppl 更稳定，并展示了 ppl 可以在短 QA 上接近或超过基于验证器的基线。
- RLPR {% ref rlpr %} 使用相对令牌概率（每个令牌的平均概率，裁剪，然后计算优势）而不是 ppl，并提供了详细的消融实验，表明直接使用 ppl 如果使用不当会损失 20 分。
- DRO {% ref direct_reasoning_optimization %} 针对长答案，使用相对推理 ppl 进行每个令牌加权，用于高方差真实答案令牌和局部权重衰减。
- DeepWriter {% ref reverse_engineered_reasoning %} 专注于长文本写作，但纯粹使用推理 ppl 作为评分指标来过滤和迭代重写草稿（不是 RL 训练），通过保持在监督选择制度中避免数值不稳定性。

# 观察

<figure>
<img src="https://i.mji.rip/2025/09/13/b6a2baa9c386467e6e30989a3390adf8.png" width="600"/>
<figcaption>训练中的崩溃模式。</figcaption>
</figure>

我们早期训练中遇到了许多崩溃模式：completion length爆炸，模型产生混乱文本的糟糕 rollout，以及格式奖励的同时爆炸。我们应用了上述技巧来稳定训练（见论文的消融实验，详细介绍“代理的诅咒”）。

<figure>
<img src="https://i.mji.rip/2025/09/13/bbbd4be0ecaebc5827a9b09efa82bbcc.png" width="600"/>
<figcaption>The curse of proxy.</figcaption>
</figure>


一个小的但有用的小技巧是奖励依赖：当多个奖励项简单相加时，模型可能不确定哪个目标产生了给定的惩罚或奖励。实际上，我们发现将任务奖励限制在严格的格式奖励上有效：除非格式奖励满足，否则将所有其他奖励设置为零。当格式奖励获得时，模型通常是“合理的”，没有幻觉或乱码。这种依赖也可以将模型从训练崩溃中拉回来。

我们发现，对格式奖励的过度严格可能会阻碍探索 {% ref simplerl_zoo %}. 例如，我们观察到的一种有趣的格式奖励 hack 是 CoT 中的嵌套 `<think>` 标签：模型可以在外层 `<think>` 块内嵌套一个子推理反射，以游戏信号，例如

```
<think>
  inner thoughts
  <think>
    reflection on the earlier thoughts
  </think>
  continue reasoning
</think>
<answer>
...
</answer>
```

更强大的基础模型能更有效地利用密集的语义信号。例如，我们将选择题转换为开放式答案，要求模型同时输出选项字母和完整的选项文本；在比较 7B 和 3B 模型时，7B 模型在利用 ppl 对输出结果进行排序方面表现更优：

- rank 1: 选项字母和选项内容均正确
- rank 2: 字母填错，但选项文本正确
- rank 3: 字母正确，选项文本与另一个选项相似
- rank 4: 字母正确但选项文本完全错误
- …
- lowest: 完全错误

通过分析答案之外的推理输出，人们可以间接了解推理过程的差异。在一个需要解释和数值计算的天文问题中，我们让 GPT 分析每个推理过程（包括推理和结果），并按照推理质量进行排序；模型的定性分析结果与质量排名相符。

<figure>
<img src="https://i.mji.rip/2025/09/13/3c6a8ca4f84f7aa6d787a4121aed2482.png" width="600"/>
<figcaption>PPL 排名和 GPT 定性分析之间的关系。</figcaption>
</figure>

NOVER 也部分适用于非 Qwen 模型，但一些较弱的基座（例如某些 Mistral 检查点）会表现出异常行为。零样本 CoT 可以看作是一种未训练的基座探索策略；如果该基线接近或超过基础模型，强化学习（RL）通常能带来收益。

<figure>
<img src="https://i.mji.rip/2025/09/13/e346a72fe2821b7eb95045dde4fbd235.png" width="600"/>
<figcaption>NOVER 在一定程度上可以支持非 Qwen 模型。</figcaption>
</figure>

我们还注意到（并未进行详尽实验），许多通用推理数据集是由封闭式大型模型标注的，因此其客观性和准确性并不完美（定义松散，符号误用）。困惑度依然能提供有价值的指导信号：在某些情况下，模型通过困惑度信号学习到了复杂的推理模式，这些模式产生的答案可能比原始的真实标签更为合理。

# 修改奖励就足够了吗

<figure>
<img src="https://i.mji.rip/2025/09/13/8cfc464fbb6cc0c7bc08f82c929c5475.png" width="1000"/>
<figcaption>Some works on reproducing GRPO.</figcaption>
</figure>

不，但在将基于规则的验证扩展到通用推理时，奖励设计是最明显的不足。此外，从经验教训来看，许多算法技巧都是徒劳的：数据和计算才是关键。到三月，许多人都在复制 GRPO 并指出其脆弱性；我们的 NOVER 训练也暴露了类似的问题。与数据和规模相比，这些论文中提出的许多算法“技巧”的效果并不显著。

推进通用推理在数据基础模型方面面临更大挑战，后期需要通过算法工作来提升训练的效率和稳定性。

- 现有的通用推理数据集质量参差不齐；清理数据需要耗费大量精力，而且许多数据是由 LLM 标注的（源自 GPT 或类似模型），而非人工精心编辑。这些数据是静态且数量有限的。强化学习在样本效率方面具有优势；实现规模化扩展的具成本效益的路径并非简单地增加更多示例，而是要提升环境和反馈的质量。
- 基础模型负责强化学习中的探索。实际上，它应已具备零样本指令跟随和思维链（CoT）能力，更丰富的知识会更有利。关于强化学习能否达到最终能力上限的讨论并非重点：训练后往往能激发潜在能力而非创造新能力。部分研究已探索结合记忆与启发式方法，我认为中期训练与训练后可能形成新的正反馈循环。

# One more thing: 攀爬Solver–Verifier不对称性

<figure>
<img src="https://i.mji.rip/2025/09/13/d53696cd2ce193f6011f24093c6e0f26.png" width="600"/>
<figcaption>The Solver-Verifier Asymmetry.</figcaption>
</figure>

RLVR 的一个核心概念是求解器-验证器的不对称性：对于某些任务，验证比求解要容易，而对于另一些任务，验证则更困难。当验证相对求解较为简单时，RLVR 的表现尤为出色。而验证更困难的情况则包括：

- 面对难以验证的自由回答的一般性推理
- 需要长时间才能获得回报的情境（例如，一个商业计划，其真实反馈可能在数周、数月甚至数年后才出现）。这些情况类似于推荐系统中的深度转化问题：我们需要精确的归因方法以及能够处理极度稀疏反馈的系统。
- 可能需要大量人工标注或难以获取的真实用户来验证解答的场景，这推动了高效用户模拟器的开发。

NOVER 的 无验证器（verifier-free） 设计带来了一个新的可能性（尽管尚未实验）：

**是否可以将策略模型（policy model）的智能同步到验证器模型（verifier model），从而使求解器与验证器能够沿着 Solver-Verifier Asymmetry 的对角线共同进化？**

更强的策略模型会带来更强的验证器模型，而更强的验证器模型又能训练出更强的策略模型。关键在于智能的传递。NOVER 基于困惑度（perplexity）作为奖励的设计，自然地统一了求解器和验证器的智能形式：二者都旨在提高在良好推理轨迹上生成真值（ground truth）的概率。 因此，可以通过标准的 RL 来实现共同进化，而无需额外设计对抗或自博弈任务。在这里，智能的传递方向是从求解到验证。一个对称的相关工作是 LLaVA-Critic-R1，它发现强大的偏好模型可以带来强大的策略模型，但它需要构造一个额外的任务。{% ref llava_critic_r1 %}

如果我们希望实现这种完全自动化的共同攀爬，那么现有的 RL 训练相当于执行 横向攀爬（固定 verifier y，提升 solver x），而 智能同步（Intelligence Sync） 则会执行 纵向攀爬（固定 solver x，提升 verifier y）。然而，我们还需要第三个变量：任务与数据。在求解器–验证器网格中的每一个点，都对应着特定的任务和数据集。正如我在早先关于 Scaling the Environment 的文章中所论述的，除了求解器与验证器之外，还有出题器（question generator）。目前大多数关于推理进化的研究都集中在通过模型一致性或熵模式实现自我改进；部分方法实现了两个模块的共同进化，但三个模块的三重进化（tri-evolution）尚未被探索：

<figure>
<img src="https://i.mji.rip/2025/09/13/e73159cd71bc40614799577a41c770f5.png" width="600"/>
<figcaption>The Trinity of Solver-Verifier-Generator.</figcaption>
</figure>

- R-Zero 和 Self-Questioning Language Models 考虑了生成器与求解器之间的对抗式生成 {% ref r_zero %}{% ref self_questioning %}。
- URPO 将验证重新表述为一个求解任务并统一了数据训练；COOPER 则从当前策略的 rollout 构造正/负样本来训练验证器。这两条路线都实现了求解器–验证器的共同进化 {% ref urpo %}{% ref cooper %}。

另一条持续改进求解器的路径是 自博弈（self-play）：在合适的环境下，两个求解器可以通过对弈来相互提升，而不必担心非对称性。对于一般性推理，这类环境很难设计，因为“规则”本身是模糊的。近期有研究证明模型能够学习规则 {% ref llms_can_learn_rules %}，并且可以通过合成数据和任务 组合原子技能以学习新技能，但现有的真实通用推理数据集仍然只是有限的枚举，而非全面的规则集。这依旧本质上是静态数据集/基准驱动的 RL。在 AI 的“下半场”，我们应当寻求真实世界的环境与问题，而非停留在静态数据集。

在静态数据与真实世界之间存在一种中间件：模拟器（simulators）。模拟器以牺牲真实性换取反馈速度——类似奖励模型或验证器模型——而在通用推理场景中，一个有用的模拟器可能会类似医学领域的病人模拟器（参考 Baichuan-M2 的案例），因为真实病人涉及伦理与监管问题，验证也往往较慢 {% ref baichuan_m2 %}。

另一种思路是放弃特定任务环境，而转向 博弈环境：如果推理模式能够跨游戏与任务迁移，那么在游戏中的自博弈可能提升数学与通用推理能力 {% ref play_to_generalize %}{% ref spiral %}。若可行，我们就能够利用游戏环境和自博弈来持续进化通用推理模型。

# 引用
如果你觉得这篇博文的话题很有趣，需要引用时，可以使用如下bibtex:
```bibtex
@article{general_reasoning_202509,
  author = {Wei Liu},
  title = {Towards General Reasoning},
  year = {2025},
  month = {9},
  url = {https://thinkwee.top/2025/09/13/general-reasoning/#more},
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
[compass_verifier] CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward.
[is_cot_mirage] Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens.
[ace_rl] ACE-RL: Adaptive Constraint-Enhanced Reward for Long-form Generation Reinforcement Learning.
[checklists_are_better] Checklists Are Better Than Reward Models For Aligning Language Models.
[ticking_all_the_boxes] TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation.
[rubric_anchors] Reinforcement Learning with Rubric Anchors.
[rubrics_as_rewards] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains.
[baichuan_m2] Baichuan-M2: Scaling Medical Capability with Large Verifier System.
[robust_reward_modeling] Robust Reward Modeling via Causal Rubrics.
[deliberative_alignment] Deliberative Alignment: Reasoning Enables Safer Language Models.
[on_reward_functions] On Reward Functions For Self-Improving Chain-of-Thought Reasoning Without Supervised Datasets.
[what_is_wrong_with_perplexity] What is Wrong with Perplexity for Long-context Language Modeling?
[quantile_reward_policy_optimization] Quantile Reward Policy Optimization: Alignment with Pointwise Regression and Exact Partition Functions.
[reinforcing_general_reasoning] Reinforcing General Reasoning without Verifiers.
[rlpr] RLPR: Extrapolating RLVR to General Domains without Verifier.
[direct_reasoning_optimization] Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks.
[reverse_engineered_reasoning] Reverse-Engineered Reasoning for Open-Ended Generation.
[simplerl_zoo] SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild.
[llava_critic_r1] LLaVA-Critic-R1: Your Critic Model is Secretly a Strong Policy Model.
[r_zero] R-Zero: Self-Evolving Reasoning LLM from Zero Data.
[self_questioning] Self-Questioning Language Models.
[urpo] URPO: A Unified Reward & Policy Optimization Framework for Large Language Models.
[cooper] COOPER: CO-OPTIMIZING POLICY AND REWARD MODELS IN REINFORCEMENT LEARNING FOR LARGE LANGUAGE MODELS.
[llms_can_learn_rules] Large Language Models can Learn Rules.
[play_to_generalize] Play to Generalize: Learning to Reason Through Game Play.
[spiral] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning.
{% endreferences %}