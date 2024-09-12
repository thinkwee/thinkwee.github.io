---
title: What's the next step to Large Models?
date: 2024-04-23 15:50:02
categories: NLP
tags:
  - inference
  - math
  - seq2seq
mathjax: true
---

Some very-personal questions, assumptions and predictions on the future after the large model era. I hope to keep it a habit for writing such future-ask post for every half year or one year to keep me thinking about the "next token" in the AI era. Still in draft.

<!--more-->


# Is compression all we need?
- The first question concerns compression. Large models compress the corpus of the world into a model and can be "extracted" by everyone through natural language interaction. This process undoubtedly alleviates information or knowledge asymmetry. For instance, a dentist can code a program by querying a LLM, while a programmer can boost their work with LLM assistance. Extracting existing knowledge encoded in LLMs is always beneficial, but we want more from LLM and may ask:
  - Is it possible to discover new things with existing compressed information/knowledge? For example, could a physicist uncover a new law from an LLM? Or could an LLM predict this post? The answer is both yes and no. 
    - For yes, mathematicians have proven the affirmative, as many pure-theory research discoveries stem solely from scientists' brains and past knowledge. Compression-based large models excel in leveraging past knowledge and may discover new discoveries if they role-play a scientist's cognitive processes effectively. 
    - For no, since some new discoveries often arise from empirical observations (they "discover" sth because they see it), such as in biology, rather than solely from reasoning on past existing information.
  - Another question is: do we need to discover new things? Perhaps 99.999% of the world's operations in the next second adhere to past patterns. A tool that efficiently extracts and utilizes these patterns can have profoundly positive impacts. While true, our pursuit of AGI propels us to seek more.
  - The crux of this fuzzy discussion lies in the subtitle: "Is compression all we need?" If I can compress the entire world, with its countless and all kinds of data, into a model, can it predict the future? The answer is affirmative if the model can simulate the entire world. By fast-forwarding the world simulation process, one could glimpse into the future. However, does compression and conditional extraction equals to simulation?
  - Musk posited that our primary concern should revolve around the transformation between energy and intelligence. But is compression the optimal transformation method? It may serve as a proficient transformation between energy and compressed knowledge.
  - Accompanying this "compression doubt" is the question of "Is predicting the next token all we need?". This query probes whether all knowledge is procedural and causal.

# World Model, in a data-driven fasion?
- For world model, it seems like splitting the intelligence into several sub-parts connecting each other based on the prior of human recognition. But most data we have is unsupervised or end2end.
- If the data can be splited and organized for all these sub-systems, can we realize world model in the format of multi-agent or multi-lm systems?

# Agents
- Many studies on large models may be overshadowed by OpenAI's Bitter Lesson. Will agents face a similar fate? In other words, is the content studied by agents irreplaceable even after scaling up large models? This might depend on whether the most naive response of LLM can transition from System 1 to System 2.
- If a agents holds all the action and information of one human, can we say it is a human?

# Alignment/Feedback
- It's all about Data Flywheel. The target is to get better signals after aligning model in each update.
- Alignment proves that we want better positive samples instead of constructing negative samples.
- Alignment can be good and bad, depends on the target the model aligns to.
- How can we connect various feedback, human/non-human, text/other modalities, human society/physical world.
- By connecting all these feedback, we can align with more powerful target. What's more, the law to connecting these feedback may reveals the rule of world.
- Energy hides in the tradeoff of using reward model to replace human label. We can tradeoff a little precision for scalable training/rewarding/labeling. Can we discover more energy in this process?
- Another tradeoff is the alignment tax. What is the energy hiding in it?

# Beyond Language
- Language is complicated, reasoned, abstracted than other modality, since it is actually "unnatural" and created by human.
- But researchers found a amazing objective for language, which reflects the whole history of computational linguistic, the language model, i.e, predicting next token.
- Other modality like image/video/sound, is natural as they convey the original information of the physical world. Could these information have a similar or more naive objective?
- What is the meaning of more modality to the reasoning ability of large models?