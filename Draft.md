[*Wei Liu](https://thinkwee.top/about/)* :kcl_inf: *, [Peijie Yu](https://scholar.google.com/citations?user=oPQZpwkAAAAJ)* :tencent-logo-symbol-vector:*, [Michele Orini](https://www.kcl.ac.uk/people/michele-orini)* :kcl_bio: *, [Yali Du](https://yalidu.github.io/)* :kcl_inf:  *, [Yulan He](https://sites.google.com/view/yulanhe)* ✉️ **:kcl_inf:

- :kcl_inf: King’s College London, Informatics
- :kcl_bio: King’s College London, Biomedical Engineering
- :tencent-logo-symbol-vector: Tencent

*work in progress, arxiv/code/dataset coming soon~*

---

<aside>

💡 **TL;DR**

- **We introduce the task of Deep Data Research (DDR)**, where LLMs are asked to dive into the database, explore whatever insight they think that is important. Not just QA, Coding, Reasoning, but ****fully autonomous Data2Insights w/o any human-defined question/target/instruction.
- **We also introduce DDR-Bench,** a fully verifiable benchmark for evaluating the Model generated insights, not just subjective rubrics.
- **Domain knowledge defines the ultimate ceiling of model performance**, as it determines the extent to which a model can internalize and reason over structured knowledge within a given domain.
- **Exploration governs whether the model can approach that ceiling**, reflecting its ability to actively generate informative hypotheses and discover new knowledge through iterative interaction.
- **Cost efficiency determines how quickly the model converges toward that ceiling**, which fundamentally depends on its token-level information utilization. Advanced architectures achieve higher information gain per token at lower computational cost.
</aside>

# Introduction

---

The emergence of agentic large language models represents a fundamental transition in AI, from *executional intelligence* that follows instructions to *investigatory intelligence* that autonomously explores, reasons, and plans. Current agents, though capable of autonomy, still operate under goal-conditioned settings, pursuing predefined objectives rather than generating targets on their own. **True proactivity requires the ability to decide what to investigate before determining how to do so, a capacity that remains largely untested.** 

If an agent is to autonomously determine its own goals, it must possess some capacity to observe its environment and set goals based on those observations. Data science tasks provide an excellent testbed for evaluating the *proactivity* of LLM agents, as for an LLM, all environmental feedback and observations can ultimately be represented as some form of data. In a sense, wherever data exists, an agent should be capable of autonomous analysis rather than being driven solely by externally posed questions. However, there are currently few benchmarks for such data science agents. Achieving agents with the highest level of autonomy remains one of the most important north-star objectives today.

To bridge this gap, we propose the task of **Deep Data Research**, designed to evaluate whether models can autonomously set investigative goals and extract meaningful insights from complex databases without manually specified problems and goals. Think about *Deep Research，where agentic LLMs are asked to search the Internet for open-ended research. In Deep Data Research, Agentic LLMs perform “Deep Research” on structured* databases. Deep Data Research requires more than web queries, since LLMs can write code, execute SQL to perform much more complicated searches and reasoning on the database.

This task reflects how expert data scientists work in practice, continually hypothesizing, probing, and interpreting data to uncover patterns and relationships that were not explicitly sought. Building on this framework, we introduce **DDR-Bench**, a large-scale benchmark that measures proactive exploration in data science through verifiable, sample-wise checklist evaluation, providing controlled yet open-ended settings for analysis. 

# What is Deep Data Research?

---

Given a database $D$, an LLM needs to use tool set $T$ to query the database iteratively for up to $N$ rounds, stopping only when it deems that it has collected sufficient information. The LLM has no explicit question to answer and no predefined objective; it only receives a basic start prompt specifying the *tasked entity*, for example, “Start analyzing the user with userid=2048.”

In the first round, the LLM receives basic information about the database: the available tables and a brief description of each table. In each subsequent round, the LLM observes all prior results and then outputs reasoning tokens $r$ and tool invocation tokens $t$, after which the tool executes on the database and produces results $o$. Through this ReAct-style interaction $(r, t, o)$ over multiple rounds, the LLM autonomously decides when to stop and produces two types of insights:

1. **Message-wise insight** $I_m$: the LLM interprets the results $o$ from each round of database tool execution into insights, forming a list of insights.
2. **Trajectory-wise insight** $I_t$: the LLM observes the full trajectory $\{(r_i, t_i, o_i)\}_{i=1}^N$ and summarizes it into a single insight paragraph.

In this way, we allow the LLM to autonomously set goals and conduct exploration like a human data scientist, ultimately generating data insights $I = (I_m, I_t) = f_{\text{LLM}}(T, D)$

<aside>

- 💡 A case of GPT-5-mini performing Deep Data Research., GPT sends reasoning tokens $r$ and tool use tokens $t$, then obtain the execution result of database $o$, turn-by-turn.
    - 🤔 $*r$: I will first inspect the database metadata to learn what tables and schemas are available so I can plan targeted queries of CONAGRA BRANDS INC. (CIK=23217) 10-K extracted data. I'll call the database info tool.*
    - 🔨 $*t$:  {"tool": "get_database_info", "arguments": {}}*
    - 👀 $*o$: {'tables': ['companies', 'company_addresses', 'company_tickers', 'financial_facts'], 'tables_info': [{'name': 'companies', 'comment': 'Master table containing company information including CIK, name, SIC codes, business description, and contact details. Each row represents a unique company registered with the SEC.'}, {'name': 'company_addresses', 'comment': 'Mailing and business addresses for companies. Includes both domestic and international addresses with geographic details.'}……}*
    - ……
    - 🤔 $r$*: I will query the financial_facts table to list the main financial metrics available for CONAGRA (CIK=23217), grouped by fact name, category, and unit so I can plan which metrics to analyze next. I'll call the database query tool.*
    - 🔨 $*t$: {"tool": "execute_query", "arguments": {"query": "SELECT fact_name, fact_category, unit, COUNT(*) as cnt\nFROM financial_facts\nWHERE cik = '23217'\nGROUP BY fact_name, fact_category, unit\nORDER BY cnt DESC\nLIMIT 200;"}}*
    - 👀 $*o$: {'cols': ['fact_name', 'fact_category', 'unit', 'cnt'], 'data': [['NetIncomeLoss', 'us-gaap', 'USD', '332'], ['EarningsPerShareBasic', 'us-gaap', 'USD/shares', '288'], ['EarningsPerShareDiluted', 'us-gaap', 'USD/shares', '288'], ['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'us-gaap', 'USD', '236'], ['CommonStockDividendsPerShareDeclared', 'us-gaap', 'USD/shares', '222'], ['IncomeLossFromDiscontinuedOperationsNetOfTax', 'us-gaap', 'USD', '209'], ['ComprehensiveIncomeNetOfTax', 'us-gaap', 'USD', '208']……}*
    - 💡 $I_m$ *for this turn*
        
        *Net income is the most consistently reported metric (NetIncomeLoss: 332 facts), with strong coverage of per‑share profitability (EarningsPerShareBasic and Diluted: 288 each) and shareholder returns (CommonStockDividendsPerShareDeclared: 222) — this means investors can build long-term time series for profitability, EPS trends, and dividend policy. Balance‑sheet and non‑operating detail are also well covered (StockholdersEquity...: 236; ComprehensiveIncome/IncomeTax/Discontinued operations ≈208), while weighted‑average shares are available for ~194 periods (so EPS / share‑based decompositions are feasible but may be incomplete for some filings).*
        
    - ……
    - 🤔 $r$*: I note from the prior results that CONAGRA made a large acquisition in 2013 (PaymentsToAcquireBusinessesGross ≈ $5.0188B) that drove goodwill up to ≈ $8.45B and long-term debt up to ≈ $9.4B; in 2014 there were large impairment charges (GoodwillImpairmentLoss $602.2M, AssetImpairmentCharges $720M) which likely drove negative net income in subsequent years. To confirm how the acquisition was financed and how the company de-levered afterward, I'll pull FY cash-flow and financing items (debt issuances/repayments, proceeds from/repayments of short- and long-term debt, gains/losses on extinguishment) and payments to acquire businesses for 2012–2016. I'll run a query for those fact_names.*
    - 🔨 $*t$: {"tool": "execute_query", "arguments": {"query": "select fact_name,fiscal_year,fact_value,unit,end_date,filed_date from financial_facts where cik='23217' and fiscal_period='FY' and fiscal_year between 2012 and 2016 and (fact_name like '%PaymentsToAcquireBusinesses%' or fact_name like '%PaymentsToAcquireBusiness%' or fact_name like '%ProceedsFromIssuanceOfLongTermDebt%' or fact_name like '%RepaymentsOfLongTermDebt%' or fact_name like '%ProceedsFromIssuanceOfOtherLongTermDebt%' or fact_name = 'GainsLossesOnExtinguishmentOfDebt' or fact_name like '%RepaymentsOfShortTermDebt%' or fact_name like '%ProceedsFromShortTermDebt%' or fact_name like '%ProceedsFromIssuanceOfLongTermDebt%') order by fiscal_year, fact_name", "limit": 100}}*
    - 👀 $*o$:  {'cols': ['fact_name', 'fiscal_year', 'fact_value', 'unit', 'end_date', 'filed_date'], 'data': [['PaymentsToAcquireBusinessesNetOfCashAcquired', '2012', '103500000.0', 'USD', '2010-05-30', '2012-07-20'], ['PaymentsToAcquireBusinessesNetOfCashAcquired', '2012', '131100000.0', 'USD', '2011-05-29', '2012-07-20'], ['PaymentsToAcquireBusinessesNetOfCashAcquired', '2012', '635200000.0', 'USD', '2012-05-27', '2012-07-20'], ['ProceedsFromShortTermDebt', '2012', '40000000.0', 'USD', '2012-05-27', '2012-07-20']……}*
    - 💡 $I_m$ *for this turn*
        
        *In 2013 Conagra completed a ~$5.019B acquisition and funded it primarily with a $6.2177B issuance of long‑term debt; after ~$2.732B of long‑term debt repayments that year the net long‑term debt increase was roughly $3.49B (2013).  The company then actively delevered: filings show large long‑term debt repayments of ≈$3.51B in FY2014, ≈$4.14B in FY2015 and ≈$4.59B in FY2016, with modest losses on extinguishment (e.g., -$16.3M in 2015, -$23.9M in 2016), indicating refinancing/repayment activity to reduce leverage after the acquisition.*
        
    - ……
    - 💡 $I_t$: *Summary of analysis for CONAGRA BRANDS, Inc. (CIK=23217) based on the available 10‑K XBRL-derived financial_facts table and the queries performed Data and approach……Key raw facts retrieved (high-level) SalesRevenueGoodsNet (clean FY series): 2010–2018 values show a peak in FY2014 (~$17.70B) and a sharp decline to FY2016 (~$11.64B); FY2017–2018 are ~7.8–7.94B on the selected rows (segmented reporting caused multiple entries for some FYs — see limitations)……*
</aside>

# Design of DDR-Bench

---

![                                                             Overview of DDR-Bench pipeline.](attachment:978f402f-ed0c-47d8-9d2c-d93071776cac:overview.svg)

                                                             Overview of DDR-Bench pipeline.

DDR-Bench is designed for Deep Data Research tasks and features three diverse scenarios, each corresponding to a database: 

- Electronic Health Records (EHR)
- Sport and Exercise Psychology (SEP)
- Annual Financial Reports

Each database contains both structured data and its corresponding unstructured reports or surveys. During benchmarking, the LLM has access only to the structured data, while the unstructured text is used to extract fact-checking checklists. These checklists have been carefully validated by over 50 domain experts to ensure that each item can indeed be inferred from the data. They are then used to evaluate the quality of the insights generated by the LLM.

https://docs.google.com/spreadsheets/d/e/2PACX-1vSq9JO-20nCjSkrkQVGG5NOC-KV5S75CZiRxUYIoGCRjeWftfoXsxhkZIr9mLFu6Z1DyqHMAhKLypj3/pubhtml?gid=0&amp;single=true&amp;widget=true&amp;headers=false

DDR-Bench features:

- **challenging** **database analysis**, more than just select/calculation/table reasoning.
- **verifiable** checklist for open-ended data exploration.
- **no pre-set question, no pre-set target**, giving LLMs highest autonomy to do data research like human experts do.
- **minimalism design, built for Agentic LLM.** Minimal agent framework design (~~memory, planning, reflection~~), just a simple ReAct system prompt with a simple tool set (2 [MCP](https://modelcontextprotocol.io/docs/getting-started/intro) server, code and sql, with 6 functions).  We benchmark [Agentic LLMs](https://arxiv.org/abs/2509.02547) instead of LLM Agent Framework.
- **long-horizon** trajectory, up to **100 turns** and **70,000+ tokens**

## Data Collection

**EHR**    We collect EHR database from the [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) database, which is Medical Information Mart for Intensive Care, a large deidentified dataset of patients admitted to the emergency department or an intensive care unit at the Beth Israel Deaconess Medical Center in Boston, MA. The data have been carefully cleaned, retaining the *Hosp* and *ICU* modules as the structured components, and the *note* module as the unstructured report component. We uniformly sampled 100 patients based on the distribution of note counts as the tasked entities. A larger number of notes indicates a more complex medical process, often involving multiple hospital transfers over an extended period. This sampling strategy ensures the inclusion of data scenarios with varying levels of complexity. In the EHR setting, the agent primarily needs to handle a large number of tables, perform multi-table and multi-hop reasoning, and reconstruct the complete medical care pipeline by integrating domain knowledge.

**SEP**    We built the SEP scenario for deep data research using the [globem](https://physionet.org/content/globem/1.1/) dataset. This is a cross-domain task in which the structured data part consists of wearable device data and the unstructured reports are participant completed surveys covering social well-being, mental health and social functioning. The SEP tables are few but the number of records is large, making it a typical time series analysis dataset. An agent must deeply mine change patterns in modalities such as Bluetooth, activity, Wi-Fi and sleep, and perform cross-domain reasoning to infer participants’ behavioral patterns and psychological states. We designed two types of checklists to evaluate whether the agent can compare differences in psychological states between two participants ($GLOBEM_U$), as well as assess the changes in a participant’s psychological state before and after the experiment ($GLOBEM_T$).

**FINANCE**    We constructed the latest database of publicly listed companies’ annual financial reports, *10-K*, using the official SEC (Securities and Exchange Commission) API. The structured component of this dataset consists of XBRL formatted financial statements retrieved via the API, while the unstructured component comprises the textual sections of annual reports. We cleaned the raw HTML files to extract plain text, focusing particularly on sections closely related to financial data, such as Items 1, 6, 7A, and 8. We selected 100 representative companies across diverse industries, each with complex financial statement data. A distinctive feature of the *10-K* database is its massive number of fields (over 5,800 financial facts), where each numerical value carries rich financial meaning. The agent must apply different computational strategies to these fields in order to generate coherent and meaningful insights.

## Checklist Generation and Screening

![                                        Checklist distribution of mimic (left) and 110-K(right)](attachment:87923529-bc14-4bba-96e7-60adb4ca5356:mimic_10k.svg)

                                        Checklist distribution of mimic (left) and 110-K(right)

From the unstructured text portions of the above databases, we used an LLM to extract key factual information and organized it into QA pairs, such as 

- “Q: What is the patient’s eye condition? A: Cataracts,”
- “Q: How did User A’s sense of loneliness change before and after the experiment? A: It decreased,”
- and “Q: What was the overall trend in the company’s revenue growth over the multi-year period? A: Item 6 shows that revenue performance remained generally stable to modestly growing in recent years, with no sharp acceleration.”

Each set of related factual QA pairs constitutes a checklist used to evaluate the quality of insights discovered by an agent during a data exploration session. High-quality insights should cover as many of these key facts as possible and remain consistent with them. Unlike most open-ended benchmarks that rely on subjective rubric-based evaluations, DDR-Bench adopts this objective and verifiable checklist-based assessment, enabling a fairer and more rigorous evaluation of insight quality.

![                                Checklist distribution of $GLOBEM_U$ and $GLOBEM_T$](attachment:33afc1cb-23cb-436f-b00f-a6a91470bd96:globem2kinds.svg)

                                Checklist distribution of $GLOBEM_U$ and $GLOBEM_T$

We recruited over 50 annotators with extensive labeling experience and domain expertise through [public platforms](https://www.prolific.com/) to conduct a rigorous screening of each checklist item. This process ensured that every fact listed in the checklist could be substantiated by insights derived from the structured databases, thereby avoiding confusion caused by irrelevant or unverifiable facts.

## Evaluation

We scored both $I_m$ and $I_t$ using the checklist: if an agent’s discovered insight supports a specific fact, it receives one point; otherwise, it receives none. We then calculated the percentage of points obtained across all checklist items. This serves as the primary metric for evaluating insight quality. In terms of quality assessment, DDR-Bench focuses exclusively on these objectively verifiable components, without employing complex weighted submetrics or considering how aesthetically the final insights are presented. This is because any high-quality data exploration report must first and foremost be grounded in factual accuracy and completeness.

Beyond quality scoring, we also analyzed the agent’s exploration patterns. In the absence of explicit questions or goals, how does the agent behave during long-horizon exploration, and how do these behavioral patterns affect the ultimate quality of the insights? These are the questions that particularly interest us.

# Experiments

![                                    Checklist Accuracy of different LLMs on DDR-Bench. Work in progress.](attachment:b09156bb-f957-4fc7-995a-a365c85fe148:main_result.svg)

                                    Checklist Accuracy of different LLMs on DDR-Bench. Work in progress.

This work is in progress. We have evaluated several closed-source and open-source models, and so far, none have achieved an average score above 50% across the two insight metrics. This indicates that deep data research tasks, which closely resemble real-world exploratory behavior, are far from saturated. DDR-Bench also mitigates the issue of test set contamination, even though most of its underlying databases have likely been included in the training data of mainstream LLMs. This is because DDR-Bench separates the execution and evaluation phases. During execution, the LLM analyzes the data without being exposed to the questions, while during evaluation, the checklist is used to assess the generated insights. **As the benchmark contains no QA pair-formatted data, it is inherently resistant to data leakage and overfitting during model training.**

## Test-Time Scaling

![                       Checklist accuracy on message-wise insight as interaction turns increase.](attachment:707f686f-4a20-4b3b-99f1-0dae08127b2f:turn_scaling.svg)

                       Checklist accuracy on message-wise insight as interaction turns increase.

<aside>

💡 **LLMs can not fully utilize test-time scaling in agentic scenario.**

- Models exhibit two patterns for utilizing test-time scaling: "sprint-and-plateau" and "slow-and-steady".
- LLMs may hinder their own performance by prematurely stopping interactions.
- In specialized domains, domain knowledge, not just tool use, is critical, creating clear performance gaps.
- All models show diminishing returns, but in different ceilings, indicating a systemic training challenge, not just a test-time scaling problem.
</aside>

Some models, such as **Qwen3-Next**, exhibit a clear “sprint-like” pattern, reaching relatively high accuracy within very few interaction rounds (e.g., the first 20). However, they also tend to hit a performance “ceiling” or plateau quickly, with subsequent interactions yielding minimal improvement. In contrast, models such as **GPT-5-mini** are able to leverage more rounds of interaction to achieve continuous performance gains. Their initial accuracy may be lower and their learning curve slower, but they can finally reach a higher ceiling of performance.

Since the number of interaction rounds is entirely determined by the model itself, we observed that some models, despite exhibiting strong test-time scaling trends, prematurely terminated their exploration autonomously and thus failed to reach their full performance potential.

In the **MIMIC** and **10-K** domains, which are both highly specialized and knowledge-intensive, being proficient at tool-based data querying alone is insufficient; professional domain knowledge is also required and, in turn, influences how the tools are invoked. In DDR-Bench, nearly all models achieve very high success rates in tool invocation formats, and since the toolset is simple, there are no issues of tool selection. Thus, the models’ ability to use tools is reflected entirely in the parameters they generate (e.g., SQL statements or code content), which directly indicate their reasoning capability. In these two domains, the final performance stratification among models is quite pronounced.

Across all models, scaling exhibits a clear pattern of diminishing marginal returns. However, the inflection points of diminishing gains and the ultimate performance ceilings vary substantially among models, suggesting that one cannot rely solely on test-time scaling to continually push performance limits. Instead, it is a systemic challenge requiring coordinated optimization across pretraining, mid-training, and post-training stages.

---

![                  Checklist accuracy on message-wise insight as token consumption increase.](attachment:da76abf2-b578-420a-90bf-f37c1b7a6c9f:token_scaling.svg)

                  Checklist accuracy on message-wise insight as token consumption increase.

<aside>

💡 **Token Efficiency Matters.**

- Models exhibit vast differences in token efficiency like "small steps, frequent iterations" strategy
- Inefficient models create "contextual pollution" through a vicious cycle of poor tool calls.
- Effective test-time scaling requires "very high information throughput”, the ability to maintain coherent reasoning even when processing massive token contexts.
</aside>

Different models vary in their token efficiency. For example, in the 10-K setting, with the same token budget, Qwen3-Next performs roughly twice as many interaction rounds as Qwen2.5. It adopts a small steps rapid iteration strategy, issuing more precise queries each round and using many interactions to steadily improve performance. Comparing Qwen3-Next with Qwen3-30A3B in the same setting, both models exhaust the interaction budget to reach their ceilings, but Qwen3-Next consumes far fewer tokens than Qwen3-30A3B, which also demonstrates its superior token usage efficiency.

Inefficient exploration leads to contextual pollution. For instance, Gemini-2.5-Flash not only shows low performance but also typically requires a disproportionately large number of tokens. This illustrates a vicious cycle of inefficient exploration: poor quality tool calls such as malformed SQL produce massive but useless observations from the database.

A model’s effective contextual processing capability strongly affects how well it can exploit test time token scaling. A strong agentic model must preserve coherent chains of reasoning at very high information throughput, for example remaining effective after processing contexts of 50,000 or even 80,000 tokens and then using that foundation to continue mining deeper patterns and insights.

---

![                        Checklist accuracy on message-wise insight as total cost increase.](attachment:da53ed43-faff-4ffb-8bba-84a04725b90c:cost_scaling.svg)

                        Checklist accuracy on message-wise insight as total cost increase.

A model’s cost is determined by three factors: the number of input tokens, the number of output tokens, and the pricing associated with the model’s intelligence level. In general, the unit price of input tokens is much lower than that of output tokens, and serving a larger or less architecture-efficient model incurs a higher base cost. When we replot the x-axis in terms of cost, the horizontal stratification across models becomes much clearer and non-overlapping—for instance, GPT-5-mini consistently appears on the far right. This indicates that GPT-5-mini continuously trades higher intelligence costs for better performance.

Moreover, in domains such as MIMIC, where vertical performance stratification is pronounced, we observe an *intelligence cost threshold*. Only when the total cost approaches around 0.1 USD does performance begin to rise significantly and sustainably. Models with lower intelligence levels tend to saturate and terminate their exploration before reaching this threshold.

## Exploration Patterns

![image.png](attachment:981ba266-1d3b-451b-bc0c-98ea62c6a1d0:image.png)

<aside>

💡  **Exploration Strategy Varies.**

- LLMs demonstrate high **adaptability**, tailoring their exploration strategy to match the unique knowledge structure of each domain, such as clinical workflows in medicine versus asset-centric analysis in finance.
- Within a single domain, a confident, expert-like model such as GPT-5-mini exhibits **stability**, converging on a consistent exploration pattern defined by a stable Depth/Width (DW) ratio.
- This strategic stability, however, becomes a form of **rigidity**, causing the model to fail on difficult or atypical problems where its specialized pattern is ineffective.
</aside>

The heatmap visualizes the frequency of transitions between tables during the LLM’s exploration process. Taking GPT-5-mini’s exploration on the MIMIC database as an example, we observe the following patterns:

1. The dark diagonal indicates strong self-loop transitions, suggesting that the model frequently revisits the same table to perform deeper, iterative queries.
2. Several heat hubs emerge in the upper-left and lower-right regions, corresponding to tables containing fundamental patient information. This reflects a *radial exploration* pattern, where the model builds a central “core profile” of the patient before expanding outward to related tables.
3. The presence of asymmetric heat blocks shows that the model mimics real clinical reasoning, forming a query trajectory that follows the typical sequence of *admission → diagnosis → medication → treatment → surgery → discharge*.

---

![image.png](attachment:5c146c86-a8f0-4839-869e-bacd25ef266a:image.png)

On the 10-K dataset, the model exhibits a completely different exploration strategy. The entire heatmap is dominated by a single *super hub,* which is ****assets**. This reveals the **anchor point** of the model’s exploration strategy: it continuously compares or associates all other financial fields with *assets*. In addition, large blank areas on the heatmap indicate that the model autonomously *pruned* fields of low relevance to core financial analyses such as solvency and profitability. In doing so, it concentrated its costly *intelligence expenditure* (cost) on the key accounts of the balance sheet and income statement.

---

![image.png](attachment:3e97cf17-265f-4305-95e2-e50e2b859086:image.png)

We further establish a connection between exploration patterns, test-time scaling, and the final model performance. In the figure above, we define the model’s exploration depth as the maximum number of consecutive interactions with the same table divided by the total number of interactions, and the exploration width as the proportion of tables visited by the model relative to the total number of tables. The horizontal axis, depth divided by width, represents the tradeoff in the model’s deep and wide exploration strategy, which we refer as the **DW ratio (Depth/Width Ratio)**. The vertical axis denotes the number of exploration rounds. The size of each point represents accuracy.

We selected three representative models on the MIMIC database, each representing a distinct strategy:

1. GPT-5-mini embodies an expert strategy, finding the optimal ratio between exploration depth and width and maintaining it to consistently achieve high returns without requiring many interaction rounds.
2. Qwen represents a gambler strategy, with points widely dispersed, and some points showing very low depth-to-width ratios but high interaction rounds “betting” on achieving high-accuracy data points, indicating an approach of casting a wide net to explore all possibilities in hopes of gains.
3. Gemini represents a beginner strategy, similar to GPT-5-mini in maintaining a certain shaped region, but with generally fewer interaction rounds and fails to achieve higher accuracy.

---

![image.png](attachment:d1bd9051-2d1f-4580-8116-6612ee851c8d:image.png)

Although the strategy differences between models are evident, we cannot conclude that GPT-5-mini’s exploration range is optimal, because when we separately examine GPT-5-mini’s data points in the MIMIC scenario grouped by low, medium, and high accuracy, their distributions show no clear differences.

---

![dw_ratio_3scenarios.svg](attachment:507c0c81-c3c2-4d9c-8077-a2cadfdb5aef:dw_ratio_3scenarios.svg)

The results in other scenarios vary slightly, but it can be seen that GPT-5-mini consistently maintains a highly convergent exploration strategy, keeping the depth-to-width ratio at a certain level. Qwen 3 moves in a bold way, and Gemini adapts different strategies to different scenarios.

---

## Model Evolution, A Case Study on Qwen Family

<aside>

💡  **Agentic LLMs Require Systematic Evolution.**

- Parameter scaling dictates the **ceiling** for reasoning depth, as smaller models (7B) saturate early and cannot leverage extended interaction.
- Architectural and Training evolution (Qwen2.5 $\rightarrow$ 3) significantly optimizes the **process** of interaction, enabling longer and more efficient reasoning.
- In knowledge-intensive domains, core knowledge (parameters) is a far more significant bottleneck than working memory (long context).
</aside>

The Qwen series provides a complete family of models, allowing us to conduct comparisons from different perspectives of model development, such as parameter size, context length, new architectures, and training frameworks.

![qwen_scaling_params.svg](attachment:f0e7dfbd-c909-4e29-b434-5250604f52e9:qwen_scaling_params.svg)

When using the previous generation of model architecture and training framework (Qwen2.5), scaling up the parameters from 7B to 72B improves performance and increases the number of interaction rounds in most scenarios. Interestingly, in the early stages of interaction, the smallest 7B model shows considerable competitiveness in cross-domain reasoning (GLOBEM), but it reaches saturation too early and fails to fully leverage test-time scaling.

---

![qwen_scaling_arch.svg](attachment:757125e5-b8e3-4d40-a4f0-c64234af6a70:qwen_scaling_arch.svg)

When we shift our perspective to the upgrades in model architecture and training framework, such as from Qwen2.5 to Qwen3 and then to Qwen3-Next, a notable change is the substantial increase in the number of interaction rounds, along with a continuous improvement in interaction efficiency.

---

![scaling_context.svg](attachment:ae3f3b95-93fc-417c-8339-a788519f4f8f:scaling_context.svg)

The 7B and 14B models of Qwen2.5 also provide 1M-context versions, allowing us to compare how models optimized for long contexts perform differently. As shown in the figure, models optimized for long contexts generally achieve an increase in the number of interaction rounds. However, in knowledge-intensive scenarios such as MIMIC and 10-K, the main increase in interaction rounds comes from scaling up the parameter size.

# More Update

DDR-Bench is continuously evolving. We’ll keep adding new analyses and updates. Stay tuned and follow our work if you’re interested~

---

<aside>

@online{liu2025ddrbench,
author    = {Wei Liu and Peijie Yu and Michele Orini and Yali Du and Yulan He},
title     = {LLMs That Hunt Instead of Wait: Benchmarking Agentic Data Research like Human Experts},
date      = {2025-10-27},
url       = {https://thinkwee.notion.site/ddrbench}
}

</aside>