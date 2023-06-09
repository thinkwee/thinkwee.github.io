---
title: Debates between GPTs
date: 2023-06-05 23:08:23
categories: 自然语言处理
tags:
  - GPT
  - NLP
  -	LLM
mathjax: true
html: true
---
***
-	基于[ChatGPT-Shortcut](https://github.com/rockbenben/ChatGPT-Shortcut)改了一个网页，展示了一些GPT自己与自己发生的有趣辩论。
-	体验地址在[这](https://thinkwee.top/debate/)
-   A webpage based on [ChatGPT-Shortcut](https://github.com/rockbenben/ChatGPT-Shortcut) that shows some interesting debates that took place between GPTs.
-   The experience website is [here](https://thinkwee.top/debate/)

<!--more-->

# Intro
- 这是一个基于[ChatGPT-Shortcut](https://github.com/rockbenben/ChatGPT-Shortcut)更改的项目，展示一些GPT自己和自己辩论的记录。这是一个纯前端展示页面，不包含任何的模型、数据、训练过程，也不是一个平台，没有ChatGPT-Shortcut的登录和平台共享功能。
- 这只是一个爱好、偏收集的项目，没有研究目的和商业目的。此类项目也已经有很多不错的尝试，比如b站上的[AI-talk](https://space.bilibili.com/405083326)或者油管上的[Watch GPT-4 Debate with Itself! (About whether it is an AGI)](https://www.youtube.com/watch?v=OdixRqJsA_4)
- 网址在[这](https://thinkwee.top/debate/)
- This is a project based on [ChatGPT-Shortcut](https://github.com/rockbenben/ChatGPT-Shortcut) to showcase some of GPT's own records of debates with themselves. This is a pure front-end display page, containing no models, data, training process, nor is it a platform with the login and platform sharing features of ChatGPT-Shortcut.
- This is just a hobby, collective-favour project, with no research purpose or commercial purpose. There have been many good attempts at such projects, such as [AI-talk](https://space.bilibili.com/405083326) on bilibili or [Watch GPT-4 Debate with Itself! (About whether it is an AGI)](https://www.youtube.com/watch?v=OdixRqJsA_4) on YouTube
- The website is [here](https://thinkwee.top/debate/)

# Prompt
- 以单句辩论为例，给予GPT的background prompt类似于："你是一个具有顶尖水平的专业辩手。现在你参加了一个特殊的辩论，每次发言不能超过一句话。你将会得到一个辩题和你方观点，你需要引经据典，整理语言，逻辑严谨的为这个辩题辩护。你需要首先阐述观点，之后你会得到多轮对方的阐释，你需要不断驳斥他直到说服对方。记住，每次发言不能超过一句话。所有回答以中文呈现。"
- 之后给出论点：“辩题为：{}.你方观点是支持/反对。”
- 之后在两个GPT bots之间传递观点：“对方发言：“ {}”，请反驳他，依然发言不超过一句。不要重复你方观点。不要重复之前的发言。尽可能找出对方观点漏洞。尽可能提出新证据攻击对方。”
- Take the example of a one-sentence debate, where the background prompt given to the GPT is something like: "You are a professional debater at the top of your game. Now you are taking part in a special debate where you can speak no more than one sentence at a time. You will be given a topic and your side of the argument, and you will be required to defend it logically, using quotations from the classics and organising your language. You will be given several rounds of elucidation from your opponent, and you will have to refute him until you are convinced. Remember, no more than one sentence per statement. All responses will be presented in Chinese."
- The argument is then given: "The debate is entitled: {}. Your side's argument is for/against."
- Then pass the argument between the two GPT bots: "The other side speaks:" {}", please rebut him, still speaking in no more than one sentence. Do not repeat your side of the argument. Do not repeat what you have said before. Find as many holes in the other person's argument as possible. Present new evidence to attack the other person whenever possible."

# Discovery
- 该项目想通过辩论这一极具挑战和思辨的语言应用来探索一下GPT的语言能力、逻辑能力，以及探索人类的思想究竟是否可以被概率所拟合
- 可以设计许多有意思的场景，观察GPT如何给出他的最优解，例如：
  - 限制每次只能发言一句进行辩论
  - 设计一个反事实的辩题
  - 引入第三个gpt作为裁判
  - 三方乃至n方辩论
  - 只提供背景，gpt自己设计辩题
  - 何时一个GPT bot才会被另一个GPT bot说服
  - and more
- The project aims to explore the linguistic and logical capabilities of the GPT through the challenging and discursive use of language in debate, and to explore whether human thought can be fitted to probabilities.
- A number of interesting scenarios can be devised to see how the GPT gives his optimal solution, for example
  - Limit debate to one sentence at a time
  - devising a counterfactual debate question
  - Introducing a third GPT as a referee
  - Three-way or even n-way debates
  - Provide only the background, the gpt designs his own debate
  - When will one GPT bot be convinced by another GPT bot
  - and more