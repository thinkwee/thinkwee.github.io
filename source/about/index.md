title: About Me
comments: false
html: true
---
***

<p align="center">
  <img src="https://thinkwee.top/img/avatar.jpg" alt="Your Image Description" width="200" height="200">
</p>
<center>2023.5 at XiaMen</center>

# About Me
-   Hello, I'm Wei Liu (刘维). Here are my [Email](mailto:thinkwee2767@gmail.com), [Github](https://github.com/thinkwee) and [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&user=QvW2leIAAAAJ).
    -   2014-2018: Bachelor of Communication Engineering in BUPT
    -   2018-2021: Master of Computer Engineering in CIST Lab@BUPT
    -   2021-2023: NLP Researcher, [Tencent](https://www.tencent.com/en-us/about.html)
    -   **2023.8**: I will be working as a RA at [THUNLP](https://nlp.csai.tsinghua.edu.cn/), in collaboration with Prof. [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/)
    -   **🙋🏻‍♂ Now actively seeking Ph.D. opportunities (2024 Fall/2023 Spring) on NLP! **

# Research Interests
-   I focus on exploring the intelligence in deep natural language processing models when **compressing and summarizing languages**. I believe that in the process of language compression lies the birth of knowledge and intelligence. 
-   I also started to explore the next stage of intelligence in natural languages by studying **LLM multi-agent behaviors in a complex environment or society**. It is still an early exploration and I am interested in several potential research directions, including:
    -   let LLMs build and use complicated tools with multi-agent collaboration
    -   self-organization and automatic job differentiation in multi-agent collaboration
    -   alignment in multi-agent activities
    -   improve the communication efficiency among agents
    -   make multi-agent activities improve LLMs

# Research Details
-   More Comprehensive and Factual Summarization: 
    -   Introduce Determinantal Point Processes to solve the attention degeneration in Summarization<sup>[1]</sup>.
    -   Discover the subjective bias in public summarization datasets which leads to text degeneration<sup>[2]</sup>. 
    -   Design a fine-tuning schema based on mutual information that minimizes hallucination in summarization<sup>[3]</sup>. 
    -   Improve the sentiment consistency in abstractive summarization through a memory-based approach<sup>[4]</sup>.
    -   Scientific Paper Summarization, Multi-lingual Lay Summarization, Long Document Summarization<sup>[5,6,7]</sup>.
-   More Accurate and Controllable Keyphrase Prediction: 
    -   Consider Keyphrase Ranking as an MRC task that better leverages PLM to improve performance. (Patent-only)
    -   Develop a unified present/absent Keyphrase Prediction method<sup>[8]</sup>. 
    -   Explore Controllable Keyphrase Generation as an early attempt at prompt engineering<sup>[9]</sup>. 
-   We will release a repo for building complicated tools within a LLM multi-agent virtual environment.

# Industrial Experience
-   At Tencent, I aim to improve the performance of News Feed Recommendations and Advertising.
    -   Improve the NLU ability for News Feed Recommendation by more accurate and controllable keyphrase prediction.
    -   Introducing non-commercial behaviors into advertising modeling through graph modeling.
    -   Explore stable and end-to-end feature quantization methods for Advertising models.
    -   Diverse user interest modeling in a diffusion-based way<sup>[10]</sup>.
    -   Achieve better tradeoffs between single-tower and two-tower models during the recall/pre-rank stage in Advertising<sup>[11]</sup>.

# Publications
-	[**My Google Scholar**](https://scholar.google.com/citations?view_op=list_works&hl=en&user=QvW2leIAAAAJ)
<style>
table {
border-collapse: collapse;
width: 100%;
margin-bottom: 20px;
}
th, td {
text-align: left;
padding: 8px;
border: 1px solid #ddd;
}
th {
background-color: #f2f2f2;
}
img {
display: block;
margin-left: auto;
margin-right: auto;
background-color: #fff;
}
.link {
display: inline-block;
padding: 2px 6px;
border: 1px solid #ccc;
border-radius: 4px;
margin-right: 5px;
}
.even {
background-color: #f2f2f2;
}
</style>

<table>
<tr>
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHHBSs.png" width="150"></td>
<td><strong>[1] In Conclusion Not Repetition: Comprehensive Abstractive Summarization with Diversified Attention Based on Determinantal Point Processes</strong></td>
</tr>
<tr>
<td>CoNLL 2019 Long Paper</td>
</tr>
<tr>
<td>Lei Li<sup></sup>, <strong>Wei Liu<sup></sup></strong>, Marina Litvak, Natalia Vanetik, Zuying Huang</td>
</tr>
<tr>
<td>
<a href="https://github.com/thinkwee/DPP_CNN_Summarization" class="link">code</a>
<a href="https://www.aclweb.org/anthology/K19-1077/" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHLEyn.png" width="150"></td>
<td><strong>[2] Subjective Bias in Abstractive Summarization</strong></td>
</tr>
<tr class="even">
<td>Arxiv Preprint</td>
</tr>
<tr class="even">
<td>Lei Li<sup></sup>, <strong>Wei Liu<sup></sup></strong>, Marina Litvak, Natalia Vanetik, Jiacheng Pei, Yinan Liu, Siya Qi</td>
</tr>
<tr class="even">
<td>
<a href="https://github.com/thinkwee/SubjectiveBiasABS" class="link">code</a>
<a href="https://arxiv.org/pdf/2106.10084.pdf" class="link">paper</a>
</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHjpo6.png" width="150"></td>
<td><strong>[3] CO2Sum: Contrastive Learning for Factual-Consistent Abstractive Summarization</strong></td>
</tr>
<tr>
<td>Arxiv Preprint</td>
</tr>
<tr>
<td><strong>Wei Liu</strong>, Huanqin Wu, Wenjing Mu, Zhen Li, Tao Chen, Dan Nie</td>
</tr>
<tr>
<td>
<a href="https://arxiv.org/pdf/2112.01147.pdf" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHjgTx.png" width="150"></td>
<td><strong>[4] A Multi-View Abstractive Summarization Model Jointly Considering Semantics and Sentiment</strong></td>
</tr>
<tr>
<td>CCIS 2018 Long Paper</td>
</tr>
<tr>
<td>Moye Chen, Lei Li, <strong>Wei Liu</strong></td>
</tr>
<tr>
<td>
<a href="https://www.researchgate.net/publication/332432404_A_Multi-View_Abstractive_Summarization_Model_Jointly_Considering_Semantics_and_Sentiment" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHLQW4.png" width="150"></td>
<td><strong>[5] CIST@CLSciSumm-19: Automatic Scientific Paper Summarization with Citances and Facets</strong></td>
</tr>
<tr>
<td>SIGIR 2019 Shared Task</td>
</tr>
<tr>
<td>Lei Li, Yingqi Zhu, Yang Xie, Zuying Huang, <strong>Wei Liu</strong>, Xingyuan Li, Yinan Liu</td>
</tr>
<tr>
<td>
<a href="http://ceur-ws.org/Vol-2414/paper20.pdf" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHLoXn.png" width="150"></td>
<td><strong>[6] Multi-lingual Wikipedia Summarization and Title Generation On Low Resource Corpus</strong></td>
</tr>
<tr class="even">
<td>RANLP 2019 Shared Task</td>
</tr>
<tr class="even">
<td><strong>Wei Liu</strong>, Lei Li, Zuying Huang, Yinan Liu</td>
</tr>
<tr class="even">
<td>
<a href="https://www.aclweb.org/anthology/W19-8904.pdf" class="link">paper</a>
</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHOhE6.png" width="150"></td>
<td><strong>[7] CIST@CL-SciSumm 2020, LongSumm 2020: Automatic Scientific Document Summarization</strong></td>
</tr>
<tr>
<td>EMNLP 2020 Shared Task</td>
</tr>
<tr>
<td>Lei Li, Yang Xie, <strong>Wei Liu</strong>, Yinan Liu, Yafei Jiang, Siya Qi, Xingyuan Li</td>
</tr>
<tr>
<td>
<a href="https://www.aclweb.org/anthology/2020.sdp-1.25.pdf" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHOLDI.png" width="150"></td>
<td><strong>[8] UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction</strong></td>
</tr>
<tr class="even">
<td>ACL 2021 Findings Long Paper</td>
</tr>
<tr class="even">
<td>Huanqin Wu<sup></sup>, <strong>Wei Liu<sup></sup></strong>, Lei Li, Dan Nie, Tao Chen, Feng Zhang, Di Wang</td>
</tr>
<tr class="even">
<td>
<a href="https://github.com/thinkwee/UniKeyphrase" class="link">code</a>
<a href="https://arxiv.org/pdf/2106.04847.pdf" class="link">paper</a>
</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align: middle;"><img src="https://s1.ax1x.com/2023/02/16/pSHXu24.png" width="150"></td>
<td><strong>[9] Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning</strong></td>
</tr>
<tr>
<td>AAAI 2022 Long Paper</td>
</tr>
<tr>
<td>Huanqin Wu, Baijiaxin Ma, <strong>Wei Liu</strong>, Tao Chen, Dan Nie</td>
</tr>
<tr>
<td>
<a href="https://ojs.aaai.org/index.php/AAAI/article/download/21402/version/19689/21151" class="link">paper</a>
</td>
</tr>
<tr class="even">
<td rowspan="4" style="vertical-align: middle;"></td>
<td><strong>[10] In Submission To SCIS</strong></td>
</tr>
<tr class="even">
<td>Anonymous</td>
</tr>
<tr class="even">
<td>Anonymous</td>
</tr>
<tr class="even">
<td>Anonymous</td>
</tr>
<tr>
<td rowspan="4" style="vertical-align: middle;"></td>
<td><strong>[11] In Submission to CIKM</strong></td>
</tr>
<tr>
<td>Anonymous</td>
</tr>
<tr>
<td>Anonymous</td>
</tr>
<tr>
<td>Anonymous</td>
</tr>
</table>