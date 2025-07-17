title: About Me
comments: false
html: true
---

<!-- Firebase SDK -->
<script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-app-compat.js"></script>
<script src="https://www.gstatic.com/firebasejs/10.8.0/firebase-database-compat.js"></script>

<!-- Initialize Firebase and PV counter -->
<script>
  // Wait for DOM to be fully loaded
  document.addEventListener('DOMContentLoaded', function() {
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
      const pageViewsElement = document.getElementById('page-views');
      if (pageViewsElement) {
        pageViewsElement.textContent = snapshot.val() || 0;
      }
    });
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
    }

    .pv-counter #page-views {
        transition: color 0.3s ease;
    }

    .pv-counter:hover #page-views {
        color: #6699FF;
    }
</style>

<p align="center">
  <img src="/img/bg_blog.jpg" alt="I'm in London" width="600">
</p>
<center>Since February 2025, I've been based in London pursuing my PhD journey!</center>
<center>Fun Fact: My giraffe icon isn't random - I actually have a long neck! </center>
<center>See for yourself and meet other talented researchers on our <a href="https://kclnlp.github.io/team.html" target="_blank" rel="noopener">KCLNLP team page</a>.</center>

---

<center>
  <style>
    .libutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: #ffffff !important; 
      background: linear-gradient(135deg, #8fa6c4 0%, #3d5470 100%);
      border: none;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      box-shadow: 0 2px 8px rgba(118, 158, 203, 0.15);
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .libutton:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(118, 158, 203, 0.25);
      background: linear-gradient(135deg, #7a95b8 0%, #324660 100%);
    }
    
    .libutton:active {
      transform: translateY(0);
      box-shadow: 0 2px 6px rgba(118, 158, 203, 0.15);
    }
    
    .gsbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: #ffffff !important; 
      background: linear-gradient(135deg, #6090d4 0%, #2d4a8a 100%);
      border: none;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      box-shadow: 0 2px 8px rgba(66, 133, 244, 0.15);
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .gsbutton:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(66, 133, 244, 0.25);
      background: linear-gradient(135deg, #5280c8 0%, #253f7a 100%);
    }
    
    .gsbutton:active {
      transform: translateY(0);
      box-shadow: 0 2px 6px rgba(66, 133, 244, 0.15);
    }
    
    .ghbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: #ffffff !important; 
      background: linear-gradient(135deg, #706a90 0%, #2a2a2a 100%);
      border: none;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      box-shadow: 0 2px 8px rgba(139, 125, 184, 0.15);
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .ghbutton:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(139, 125, 184, 0.25);
      background: linear-gradient(135deg, #625d80 0%, #1d1d1d 100%);
    }
    
    .ghbutton:active {
      transform: translateY(0);
      box-shadow: 0 2px 6px rgba(139, 125, 184, 0.15);
    }
    
    .gmbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: #ffffff !important; 
      background: linear-gradient(135deg, #d47570 0%, #a03838 100%);
      border: none;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      box-shadow: 0 2px 8px rgba(255, 138, 128, 0.15);
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .gmbutton:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(255, 138, 128, 0.25);
      background: linear-gradient(135deg, #c86560 0%, #902828 100%);
    }
    
    .gmbutton:active {
      transform: translateY(0);
      box-shadow: 0 2px 6px rgba(255, 138, 128, 0.15);
    }
    
    @media (max-width: 768px) {
      .libutton, .gsbutton, .ghbutton, .gmbutton {
        padding: 0px 10px;
        font-size: 13px;
        height: 30px;
        margin: 0 3px;
      }
    }
  </style>

  <center>
  You can find this guy via
  </center>
  <a class="libutton" href="http://www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=thinkwee" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 6px;">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    Linkedin
  </a>
  <a class="gsbutton" href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=QvW2leIAAAAJ" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 6px;">
      <path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/>
    </svg>
    Google Scholar
  </a>
  <a class="ghbutton" href="https://github.com/thinkwee" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 6px;">
      <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
    </svg>
    GitHub
  </a>
  <a class="gmbutton" href="mailto:thinkwee2767@gmail.com">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 6px;">
      <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.904.732-1.636 1.636-1.636h3.819v.273L12 10.186l6.545-6.092v-.273h3.819c.904 0 1.636.732 1.636 1.636Z"/>
    </svg>
    Gmail
  </a>
  <p align="center">
  <img data-src="/img/wechat.bmp" alt="wechat" width="350">
  </p>
</center>


# About Me

<div style="text-align: center; width: 100%; overflow: hidden;">
<div style="display: inline-block; width: 32%; text-align: center; vertical-align: top;">
<img src="/img/bupt.png" alt="I'm in BUPT" style="width: 90%; max-width: 250px; height: auto;">
</div>
<div style="display: inline-block; width: 32%; text-align: center; vertical-align: top;">
<img src="/img/tencent.png" alt="I'm in Tecent" style="width: 90%; max-width: 250px; height: auto;">
</div>
<div style="display: inline-block; width: 32%; text-align: center; vertical-align: top;">
<img src="/img/tsinghua.png" alt="I'm in Tsinghua" style="width: 90%; max-width: 250px; height: auto;">
</div>
</div>

- Hello, I'm Wei Liu (刘维). Welcome to my blog (<span class="pv-counter"><span id="page-views">0</span> views</span>). You can search me on google with keyword "thinkwee", which means "The Thinking Wei".
- Past experience:
    -   2014-2021: Bachelor of Communication Engineering in BUPT, and Master of Computer Engineering in CIST Lab@BUPT.
    -   2021-2023: Application Research in the NLP&LLM Department in [Tencent](https://www.tencent.com/en-us/about.html).
    -   2023-2025: Working at [THUNLP](https://nlp.csai.tsinghua.edu.cn/)@Tsinghua University with Prof. [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/) and Prof. [Chen Qian](http://qianc62.github.io) on LLM Multi-Agent System.
    -   2025-now: Proud to be a PhD advised by [Prof. Yulan He](https://sites.google.com/view/yulanhe) and a member of [KCLNLP](https://kclnlp.github.io/)!

# Recent News

- **2025.5.16 Checkout KCLNLP's amazing works [here](https://x.com/kclnlp/status/1923409800009748788), with 15 papers accepted by ACL 2025 and 3 papers accepted by ICML 2025!**.
- **2025.5.21 Check out [NOVER](https://arxiv.org/abs/2505.16022), a novel verifier-free reinforcement learning framework for training Large Reasoning Model. Train your own R1-Zero-like reasoning model on ANY DATA!**
- **2025.6.9 Check out [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL), an awesome list of Reinforcement Learning-based Large Language Agent!**

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
-   \* denotes first/co-first author.
-   Personal Agentic AI:
    -   <span class="conf-neurips">NeurIPS 2024</span> <a href="https://arxiv.org/abs/2406.14928" class="bp">paper</a>  <a href="https://github.com/thinkwee/iAgents" class="bc">code</a> Autonomous Agents for Collaborative Task under Information Asymmetry\*
-   Inference Time Scaling:
    -   <span class="conf-arxiv">arXiv 2025</span> <a href="https://arxiv.org/abs/2505.16022" class="bp">paper</a>  <a href="https://github.com/thinkwee/NOVER" class="bc">code</a> NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning\*
    -   <span class="conf-iclr">ICLR 2025</span> <a href="https://arxiv.org/pdf/2406.07155" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Scaling Large-Language-Model-based Multi-Agent Collaboration
-   Multi-Agents System with LLMs:
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2307.07924" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Communicative Agents for Software Development
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2312.17025" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Experiential Co-Learning of Software-Developing Agents
    -   <span class="conf-acl">ACL 2025</span> <a href="https://arxiv.org/pdf/2406.08979" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Multi-Agent Software Development through Cross-Team Collaboration
    -   <span class="conf-arxiv">arXiv 2024</span> <a href="https://arxiv.org/pdf/2405.04219" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Iterative Experience Refinement of Software-Developing Agents
    -   <span class="conf-arxiv">arXiv 2025</span> <a href="https://arxiv.org/pdf/2505.06904" class="bp">paper</a>  <a href="https://github.com/xymou/EcoLANG" class="bc">code</a> EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation
-   Compression Intelligence in NLP: 
    -   <span class="conf-acl">ACL 2021</span> <a href="https://arxiv.org/pdf/2106.04847.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/UniKeyphrase" class="bc">code</a> UniKeyphrase: A Unified Extraction and Generation Framework for Keyphrase Prediction\*
    -   <span class="conf-aaai">AAAI 2022</span> <a href="https://ojs.aaai.org/index.php/AAAI/article/download/21402/version/19689/21151" class="bp">paper</a> <a href="https://github.com/m1594730237/FastAndConstrainedKeyphrase" class="bc">code</a> Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning
    -   <span class="conf-conll">CoNLL 2021</span> <a href="https://www.aclweb.org/anthology/K19-1077/" class="bp">paper</a> <a href="https://github.com/thinkwee/DPP_CNN_Summarization" class="bc">code</a> In Conclusion Not Repetition: Comprehensive Abstractive Summarization with Diversified Attention Based on Determinantal Point Processes\*
    -   <span class="conf-sigir">SIGIR 2019</span> <a href="http://ceur-ws.org/Vol-2414/paper20.pdf" class="bp">paper</a> CIST@CLSciSumm-19: Automatic Scientific Paper Summarization with Citances and Facets
    -   <span class="conf-emnlp">EMNLP 2020</span> <a href="https://www.aclweb.org/anthology/2020.sdp-1.25.pdf" class="bp">paper</a> CIST@CL-SciSumm 2020, LongSumm 2020: Automatic Scientific Document Summarization\*
    -   <span class="conf-ranlp">RANLP 2019</span> <a href="https://www.aclweb.org/anthology/W19-8904.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/multiling2019_wiki" class="bc">code</a> Multi-lingual Wikipedia Summarization and Title Generation On Low Resource Corpus\*
    -   <span class="conf-ccis">CCIS 2018</span> <a href="https://www.researchgate.net/publication/332432404_A_Multi-View_Abstractive_Summarization_Model_Jointly_Considering_Semantics_and_Sentiment" class="bp">paper</a> A Multi-View Abstractive Summarization Model Jointly Considering Semantics and Sentiment
    -   <span class="conf-arxiv">arXiv 2021</span> <a href="https://arxiv.org/pdf/2112.01147.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/co2sum" class="bc">code</a> CO2Sum: Contrastive Learning for Factual-Consistent Abstractive Summarization\*
    -   <span class="conf-arxiv">arXiv 2021</span> <a href="https://arxiv.org/pdf/2106.10084.pdf" class="bp">paper</a> <a href="https://github.com/thinkwee/SubjectiveBiasABS" class="bc">code</a> Subjective Bias in Abstractive Summarization\*
