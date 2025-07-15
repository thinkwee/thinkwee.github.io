title: About Me
comments: false
html: true
---

<!-- Firebase SDK -->
<script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-database-compat.js"></script>

<!-- Initialize Firebase and PV counter -->
<script>
  // Your web app's Firebase configuration
  const firebaseConfig = {
    databaseURL: "https://blog-a38f5-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "blog-a38f5",
    apiKey: "AIzaSyDgs_qagScUwYJr0O1MUyc3EcWbhyWzoCw",
    authDomain: "blog-a38f5.firebaseapp.com",
    storageBucket: "blog-a38f5.appspot.com",
    messagingSenderId: "1091568780806",
    appId: "1:1091568780806:web:774c32a22ef91802d92587"
  };

  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);

  // Get a reference to the page views - update to match your database structure
  const pvRef = firebase.database().ref('pageViews');

  // Update page views
  pvRef.transaction(currentViews => {
    return (currentViews || 10000) + 1;  // Set default to 10000 to match your initial value
  });

  // Display page views
  pvRef.on('value', (snapshot) => {
    document.getElementById('page-views').textContent = snapshot.val() || 0;
  });
</script>

<style>
    .bc {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 40px; /* 固定按钮宽度为150像素 */
        text-decoration: none;
        background-color: #FFFFFF; /* Apple-style blue color */
        color: black;
        margin-bottom: 5px; /* 调整按钮之间的下外边距 */
        border-radius: 8px; /* Slight border radius for a softer look */
        border: 1px solid #CCCCCC; /* Border color same as background color */
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }

    .bc:hover {
        background-color: #999999; /* Darker blue color on hover */
        color: white;
        border: 1px solid transparent; /* 将边框颜色设置为透明 */
    }
    .bp {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        width: 40px; /* 固定按钮宽度为150像素 */
        text-align: center;
        text-decoration: none;
        margin-bottom: 5px; /* 调整按钮之间的下外边距 */
        background-color: #FFFFFF; /* Apple-style blue color */
        color: black;
        border-radius: 8px; /* Slight border radius for a softer look */
        border: 1px solid #CCCCCC; /* Border color same as background color */
        transition: background-color 0.3s ease; /* Smooth transition on hover */
    }

    .bp:hover {
        background-color: #6699FF; /* Darker blue color on hover */
        color: white;
        border: 1px solid transparent; /* 将边框颜色设置为透明 */
    }

    /* Conference button styles - different colors for different conferences */
    .conf-neurips {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-neurips:hover {
        background-color: #8668a3 !important; /* Red for NeurIPS on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-iclr {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-iclr:hover {
        background-color: #6fca62 !important; /* Teal for ICLR on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-acl {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-acl:hover {
        background-color: #f14950 !important; /* Blue for ACL on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-aaai {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-aaai:hover {
        background-color: #5b90a8 !important; /* Orange for AAAI on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-conll {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-conll:hover {
        background-color: #f14950 !important; /* Purple for CoNLL on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-emnlp {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-emnlp:hover {
        background-color: #f14950 !important; /* Green for EMNLP on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-sigir {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-sigir:hover {
        background-color: #e6a800 !important; /* Dark red for SIGIR on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-ranlp {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-ranlp:hover {
        background-color: #f14950  !important; /* Dark blue for RANLP on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-ccis {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-ccis:hover {
        background-color: #f14950 !important; /* Gray for CCIS on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .conf-arxiv {
        display: inline-block;
        padding: 0px 5px;
        font-size: 14px;
        text-align: center;
        width: 110px;
        text-decoration: none;
        background-color: #FFFFFF !important; /* White background by default */
        color: black !important;
        margin-bottom: 5px;
        margin-right: 5px;
        border-radius: 8px;
        border: 1px solid #CCCCCC;
        transition: background-color 0.3s ease;
        cursor: default;
    }

    .conf-arxiv:hover {
        background-color: #c45569 !important; /* Light gray for arXiv on hover */
        color: white !important;
        border: 1px solid transparent;
    }

    .pv-counter {
        display: inline;
        padding: 0;
        font-size: inherit;
        color: inherit;
        background: none;
        border: none;
        transition: color 0.3s ease;
    }

    .pv-counter:hover {
        color: #6699FF;
        background: none;
        border: none;
    }
</style>

<p align="center">
  <img src="/img/bg_blog.jpg" alt="I'm in London" width="600">
</p>
<center>Since February 2025, I've been based in London pursuing my PhD journey!</center>
<center>Fun Fact: My giraffe icon isn't random - I actually have a long neck! </center>
<center>See for yourself and meet other talented researchers on our <a href="https://kclnlp.github.io/team.html" target="_blank" rel="noopener">team page</a>.</center>
<center>
  <style>.libutton { display: flex; flex-direction: column; justify-content: center; padding: 7px; text-align: center; outline: none; text-decoration: none !important; color: #ffffff !important; width: 200px; height: 20px; border-radius: 16px; background-color: #769ecb; font-family: "SF Pro Text", Helvetica, sans-serif; } </style>
  <a class="libutton" href="http://www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=thinkwee" target="_black">Follow This Guy on Linkedin</a>
</center>
<center>
  or Find This Guy on WeChat
</center>
<p align="center">
  <img data-src="/img/wechat.png" alt="wechat" width="350">
</p>

# About Me
- Hello, I'm Wei Liu (刘维). Welcome to my blog (<span class="pv-counter"><span id="page-views">0</span> views</span>). You can search me on google with keyword "thinkwee", which means "The Thinking Wei".
- Here are my <a href="mailto:thinkwee2767@gmail.com" style="display: inline-flex; align-items: center; text-decoration: none; line-height: 1;">Email</a>, <a href="https://github.com/thinkwee" target="_blank" rel="noopener" style="display: inline-flex; align-items: center; text-decoration: none; line-height: 1;">Github</a>, and <a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=QvW2leIAAAAJ" target="_blank" rel="noopener" style="display: inline-flex; align-items: center; text-decoration: none; line-height: 1;">Google Scholar</a>.
- My previous experience
    -   2014-2021: Bachelor of Communication Engineering in BUPT, and Master of Computer Engineering in CIST Lab@BUPT
    -   2021-2023: Application Research, [Tencent](https://www.tencent.com/en-us/about.html)
    -   2023.8-2025.1: Working at [THUNLP](https://nlp.csai.tsinghua.edu.cn/) with Prof. [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/) and Prof. [Chen Qian](http://qianc62.github.io) with a focus on LLM Multi-Agent System.
    -   2025.2 Proud to be a PhD advised by [Prof. Yulan He](https://sites.google.com/view/yulanhe) and a member of [KCLNLP](https://kclnlp.github.io/)!.
- Recent News
    -   **2025.5 Checkout KCLNLP's amazing works [here](https://x.com/kclnlp/status/1923409800009748788), with 15 papers accepted by ACL 2025 and 3 papers accepted by ICML 2025!**.
    - **Check out [NOVER](https://arxiv.org/abs/2505.16022), a novel verifier-free reinforcement learning framework for training Large Reasoning Model. Train your own R1-Zero-like reasoning model on ANY DATA!**

# Research Interests
-   **Inference Time Scaling and Agentic AI**.
-   Compression Intelligence in NLP.
-   Served as reviewer for 
    - ACL(2021,2022,2024)
    - EMNLP(2020,2023,2024,2025)
    - NeurIPS(2024,2025)
    - ICLR(2024)
    - CVPR(2025)

# Industrial Experience
-   At Tencent, I aim to bridge the gap between technology in NLP and scenario in Recommendation & Advertisement.
    -   Improving the NLU ability for News Feed Recommendation.
    -   Resolving the mismatch between commercial inclinations and content interests for Wechat Ads.
    -   Stability, Warm-Up, Efficiency-Quality Tradeoff, Interpretability & Explainability on Large Recommendation System.
    -   Diverse user interest modeling.

# Publications
-   Personal Agentic AI:
    -   <span class="conf-neurips">NeurIPS 2024</span> <a href="https://arxiv.org/abs/2406.14928" class="bp">paper</a>  <a href="https://github.com/thinkwee/iAgents" class="bc">code</a> Autonomous Agents for Collaborative Task under Information Asymmetry.
-   Inference Time Scaling:
    -   <span class="conf-arxiv">arXiv 2025</span> <a href="https://arxiv.org/abs/2505.16022" class="bp">paper</a>  <a href="https://github.com/thinkwee/NOVER" class="bc">code</a> NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning.
    -   <span class="conf-iclr">ICLR 2025</span> <a href="https://arxiv.org/pdf/2406.07155" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Scaling Large-Language-Model-based Multi-Agent Collaboration.
-   Multi-Agents System with LLMs:
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2307.07924" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Communicative Agents for Software Development.
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2312.17025" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Experiential Co-Learning of Software-Developing Agents.
    -   <span class="conf-acl">ACL 2025</span> <a href="https://arxiv.org/pdf/2406.08979" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Multi-Agent Software Development through Cross-Team Collaboration.
    -   <span class="conf-arxiv">arXiv 2024</span> <a href="https://arxiv.org/pdf/2405.04219" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Iterative Experience Refinement of Software-Developing Agents.
    -   <span class="conf-arxiv">arXiv 2025</span> <a href="https://arxiv.org/pdf/2505.06904" class="bp">paper</a>  <a href="https://github.com/xymou/EcoLANG" class="bc">code</a> EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation.
-   Compression Intelligence in NLP: 
    -   <span class="conf-acl">ACL 2021</span> <a href="https://arxiv.org/pdf/2106.04847.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/UniKeyphrase" class="bc">code</a> UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction.
    -   <span class="conf-aaai">AAAI 2022</span> <a href="https://ojs.aaai.org/index.php/AAAI/article/download/21402/version/19689/21151" class="bp">paper</a> <a href="https://github.com/m1594730237/FastAndConstrainedKeyphrase" class="bc">code</a> Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning.
    -   <span class="conf-conll">CoNLL 2021</span> <a href="https://www.aclweb.org/anthology/K19-1077/" class="bp">paper</a> <a href="https://github.com/thinkwee/DPP_CNN_Summarization" class="bc">code</a> In Conclusion Not Repetition: Comprehensive Abstractive Summarization with Diversified Attention Based on Determinantal Point Processes.
    -   <span class="conf-sigir">SIGIR 2019</span> <a href="http://ceur-ws.org/Vol-2414/paper20.pdf" class="bp">paper</a> CIST@CLSciSumm-19: Automatic Scientific Paper Summarization with Citances and Facets.
    -   <span class="conf-emnlp">EMNLP 2020</span> <a href="https://www.aclweb.org/anthology/2020.sdp-1.25.pdf" class="bp">paper</a> CIST@CL-SciSumm 2020, LongSumm 2020: Automatic Scientific Document Summarization.
    -   <span class="conf-ranlp">RANLP 2019</span> <a href="https://www.aclweb.org/anthology/W19-8904.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/multiling2019_wiki" class="bc">code</a> Multi-lingual Wikipedia Summarization and Title Generation On Low Resource Corpus.
    -   <span class="conf-ccis">CCIS 2018</span> <a href="https://www.researchgate.net/publication/332432404_A_Multi-View_Abstractive_Summarization_Model_Jointly_Considering_Semantics_and_Sentiment" class="bp">paper</a> A Multi-View Abstractive Summarization Model Jointly Considering Semantics and Sentiment.
    -   <span class="conf-arxiv">arXiv 2021</span> <a href="https://arxiv.org/pdf/2112.01147.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/co2sum" class="bc">code</a> CO2Sum: Contrastive Learning for Factual-Consistent Abstractive Summarization.
    -   <span class="conf-arxiv">arXiv 2021</span> <a href="https://arxiv.org/pdf/2106.10084.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/SubjectiveBiasABS" class="bc">code</a> Subjective Bias in Abstractive Summarization.
