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

  document.addEventListener('DOMContentLoaded', function() {
    const newsList = document.querySelector('.news-list');
    if (newsList) {
        newsList.scrollTop = newsList.scrollHeight;
    }
  });

  document.addEventListener('DOMContentLoaded', function() {
    var elements = document.querySelectorAll('[data-repo]');
    if (!elements || !elements.length) return;

    function formatStars(count) {
      if (typeof count !== 'number' || isNaN(count)) return 'N/A';
      if (count >= 1000000) return (count / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
      if (count >= 1000) return (count / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
      return String(count);
    }

    // Use localStorage for persistent caching across page loads
    var CACHE_KEY = 'github_stars_cache';
    var CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
    
    function getCache() {
      try {
        var cached = localStorage.getItem(CACHE_KEY);
        if (cached) {
          var data = JSON.parse(cached);
          // Check if cache is still valid
          if (Date.now() - data.timestamp < CACHE_DURATION) {
            return data.repos;
          }
        }
      } catch (e) {
        console.warn('Failed to read cache:', e);
      }
      return {};
    }

    function setCache(cache) {
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({
          timestamp: Date.now(),
          repos: cache
        }));
      } catch (e) {
        console.warn('Failed to save cache:', e);
      }
    }

    var cache = getCache();
    var pendingRequests = 0;
    var maxConcurrentRequests = 2; // Limit concurrent requests
    var requestQueue = [];

    function processQueue() {
      if (pendingRequests >= maxConcurrentRequests || requestQueue.length === 0) {
        return;
      }

      var task = requestQueue.shift();
      pendingRequests++;
      
      var repo = task.repo;
      var el = task.el;

      fetch('https://api.github.com/repos/' + repo)
        .then(function(res) { 
          if (res.status === 403) {
            throw new Error('Rate limit exceeded');
          }
          return res.json(); 
        })
        .then(function(data) {
          var value = (data && typeof data.stargazers_count === 'number') ? formatStars(data.stargazers_count) : 'N/A';
          cache[repo] = value;
          var starHistoryUrl = 'https://www.star-history.com/#' + repo + '&Date';
          el.innerHTML = '<a href="' + starHistoryUrl + '" target="_blank" class="dynamic-value" title="Live data from GitHub - Click to view star history">' + value + '</a>';
          el.classList.add('github-stars-loaded');
          setCache(cache);
        })
        .catch(function(err) {
          console.warn('Failed to fetch stars for ' + repo + ':', err.message);
          // Use fallback values if available
          var fallbackValue;
          if (repo === 'OpenBMB/ChatDev') {
            fallbackValue = '28k';
          } else if (repo === 'thinkwee/AgentsMeetRL') {
            fallbackValue = '490';
          } else {
            fallbackValue = 'N/A';
          }
          var starHistoryUrl = 'https://www.star-history.com/#' + repo + '&Date';
          el.innerHTML = '<a href="' + starHistoryUrl + '" target="_blank" class="dynamic-value fallback" title="Cached data (API rate limited) - Click to view star history">' + fallbackValue + '</a>';
          el.classList.add('github-stars-fallback');
        })
        .finally(function() {
          pendingRequests--;
          // Process next item in queue
          setTimeout(processQueue, 1000); // Add 1 second delay between requests
        });
    }

    elements.forEach(function(el) {
      var repo = el.getAttribute('data-repo');
      if (!repo) return;

      // Show loading state
      el.innerHTML = '<span class="dynamic-value loading">...</span>';

      // Check if we have cached value
      if (cache[repo]) {
        var starHistoryUrl = 'https://www.star-history.com/#' + repo + '&Date';
        el.innerHTML = '<a href="' + starHistoryUrl + '" target="_blank" class="dynamic-value cached" title="Cached data from GitHub - Click to view star history">' + cache[repo] + '</a>';
        el.classList.add('github-stars-cached');
        return;
      }

      // Add to request queue
      requestQueue.push({ repo: repo, el: el });
    });

    // Start processing queue
    processQueue();
    if (requestQueue.length > 0) {
      setTimeout(processQueue, 1000);
    }
  });
</script>

<style>
    .news-list {
        max-height: 200px;
        overflow-y: auto;
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .news-list::-webkit-scrollbar {
        width: 4px;
    }

    .news-list::-webkit-scrollbar-track {
        background: transparent;
    }

    .news-list::-webkit-scrollbar-thumb {
        background: #ccc;
    }

    .news-list::-webkit-scrollbar-thumb:hover {
        background: #888;
    }

    .news-item {
        padding: 6px 0;
        display: flex;
        align-items: flex-start;
        line-height: 1.8;
    }

    .news-item::before {
        content: "•";
        color: #888;
        margin-right: 12px;
        flex-shrink: 0;
    }

    .news-date {
        font-size: 13px;
        color: #888;
        margin-right: 8px;
        font-family: inherit;
        flex-shrink: 0;
    }

    .news-content {
        color: #555;
        font-size: 14px;
        flex: 1;
    }

    .news-content a {
        color: #555;
        text-decoration: none;
        border-bottom: 1px solid #ddd;
    }

    .news-content a:hover {
        border-bottom-color: #555;
    }

    .news-highlight {
        background: #f5f5f5;
        padding: 0 3px;
    }

    .news-content ul {
        margin: 4px 0 0 0;
        padding-left: 20px;
    }

    .news-content li {
        margin: 2px 0;
    }
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
        background-color: #888888; /* Darker blue color on hover */
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

    /* Dynamic GitHub stars styling */
    .dynamic-value {
        display: inline;
        font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
        border-bottom: 1px dotted #ccc;
        color: inherit;
        text-decoration: none;
    }
    
    a.dynamic-value {
        cursor: pointer;
    }
    
    span.dynamic-value.loading {
        cursor: default;
    }
</style>

<p align="center">
  <img src="/img/bg_blog.jpg" alt="I'm in London" width="600">
</p>
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
      color: black !important;
      background-color: #FFFFFF;
      border: 1px solid #CCCCCC;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .libutton:hover {
      background: linear-gradient(135deg, #8fa6c4 0%, #3d5470 100%);
      color: white !important;
      border: 1px solid transparent;
    }
    
    .libutton:active {
      transform: translateY(0);
    }
    
    .gsbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: black !important;
      background-color: #FFFFFF;
      border: 1px solid #CCCCCC;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .gsbutton:hover {
      background: linear-gradient(135deg, #6090d4 0%, #2d4a8a 100%);
      color: white !important;
      border: 1px solid transparent;
    }
    
    .gsbutton:active {
      transform: translateY(0);
    }
    
    .ghbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: black !important;
      background-color: #FFFFFF;
      border: 1px solid #CCCCCC;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .ghbutton:hover {
      background: linear-gradient(135deg, #706a90 0%, #2a2a2a 100%);
      color: white !important;
      border: 1px solid transparent;
    }
    
    .ghbutton:active {
      transform: translateY(0);
    }
    
    .gmbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: black !important;
      background-color: #FFFFFF;
      border: 1px solid #CCCCCC;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .gmbutton:hover {
      background: linear-gradient(135deg, #d47570 0%, #a03838 100%);
      color: white !important;
      border: 1px solid transparent;
    }
    
    .gmbutton:active {
      transform: translateY(0);
    }
    
    .aclbutton { 
      display: inline-flex; 
      align-items: center; 
      justify-content: center; 
      padding: 0px 12px; 
      text-align: center; 
      outline: none; 
      text-decoration: none !important; 
      color: black !important;
      background-color: #FFFFFF;
      border: 1px solid #CCCCCC;
      border-radius: 8px; 
      font-family: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
      font-size: 14px;
      font-weight: 500;
      height: 32px;
      transition: all 0.3s ease;
      cursor: pointer;
      margin: 0 5px;
    }
    
    .aclbutton:hover {
      background: linear-gradient(135deg, #f07a7f 0%, #c53a40 100%);
      color: white !important;
      border: 1px solid transparent;
    }
    
    .aclbutton:active {
      transform: translateY(0);
    }
    
    @media (max-width: 768px) {
      .libutton, .gsbutton, .ghbutton, .gmbutton, .aclbutton {
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
  <a class="aclbutton" href="https://aclanthology.org/people/wei-liu-kcl/" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 6px;">
      <path d="M19 2H8a2 2 0 0 0-2 2v1H5a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-1h1a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2Zm-3 16a1 1 0 0 1-1 1H5V7h10a1 1 0 0 1 1 1v10Zm3-3h-1V8a3 3 0 0 0-3-3H8V4h11v11Z"/>
    </svg>
    ACL Anthology
  </a>
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

<div>
<ul>
<li>Hello, I'm Wei Liu (刘维). Welcome to my blog <span class="pv-counter">(<span id="page-views">0</span> views)</span>. You can search me on google with keyword "thinkwee", which means "The Thinking Wei".</li>
</ul>
</div>

- I'm interested in:
  -   (before 2023) Compression Intelligence in NLP.
  -   (2023 - 2025) Inference Time Scaling and Agentic AI.
  -   (2025 - now) **AI in the Wild** (check out this [blog](https://thinkwee.top/2025/10/05/wild-era/#more)).
- Past Experience:
    -   2014 - 2021: Bachelor of Communication Engineering in BUPT, and Master of Computer Engineering in CIST Lab@BUPT.
    -   2021 - 2023: Application Research in the NLP&LLM Department in [Tencent](https://www.tencent.com/en-us/about.html).
    -   2023 - 2025: Working at [THUNLP](https://nlp.csai.tsinghua.edu.cn/)@Tsinghua University with Prof. [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/) and Prof. [Chen Qian](http://qianc62.github.io) on LLM Multi-Agent System.
    -   2025 - 2029(expected): Happy to join [KCLNLP](https://kclnlp.github.io/) as a PhD candidate advised by [Prof. Yulan He](https://sites.google.com/view/yulanhe) and [Prof. Yali Du](https://yalidu.github.io/)!

# Recent News

<ul class="news-list">
<li class="news-item">
<span class="news-date">2023.09.07</span>
<span class="news-content">🚀 <a href="https://github.com/OpenBMB/ChatDev" target="_blank">ChatDev</a> made #1 on Github Trending! It now has earn over <span class="news-highlight"><span id="chatdev-stars" data-repo="OpenBMB/ChatDev">...</span> stars!</span>
</li>
<li class="news-item">
<span class="news-date">2024.05.15</span>
<span class="news-content">🇹🇭 2 papers about LLM Multi-Agent System were accepted by ACL 2024!</span>
</li>
<li class="news-item">
<span class="news-date">2024.08.15</span>
<span class="news-content">🚀 I made a interactive websites showing popular MultiAgent Frameworks and show THUNLP MultiAgent Team's amazing works at <a href="https://thinkwee.top/multiagent_ebook" target="_blank">LLM MultiAgent EBook</a></span>
</li>
<li class="news-item">
<span class="news-date">2024.09.26</span>
<span class="news-content">🇨🇦 1 paper about Personal Agentic AI was accepted by NeurIPS 2024!</span>
</li>
<li class="news-item">
<span class="news-date">2025.01.22</span>
<span class="news-content">🇸🇬 1 paper about Scaling LLM Agents was accepted by ICLR 2025!</span>
</li>
<li class="news-item">
<span class="news-date">2025.05.15</span>
<span class="news-content">🇦🇹 1 paper about LLM Agent for Software Development was accepted by ACL 2025!</span>
</li>
<li class="news-item">
<span class="news-date">2025.02.21</span>
<span class="news-content">🇬🇧 I started my journey as a PhD student in the UK!</span>
</li>
<li class="news-item">
<span class="news-date">2025.05.16</span>
<span class="news-content">🎉 Find KCLNLP's amazing works <a href="https://x.com/kclnlp/status/1923409800009748788" target="_blank">here</a>, with <span class="news-highlight">15 papers accepted by ACL 2025</span> and <span class="news-highlight">3 papers accepted by ICML 2025</span>!</span>
</li>
<li class="news-item">
<span class="news-date">2025.06.09</span>
<span class="news-content">🚀 I made <a href="https://thinkwee.top/amr/" target="_blank">AgentsMeetRL</a>, an awesome list of Reinforcement Learning-based Large Language Agent, which has now earned over <span class="news-highlight"><span id="agentsmeetrl-stars" data-repo="thinkwee/AgentsMeetRL">...</span> stars!</span> welcome to the era of experience!</span>
</li>
<li class="news-item">
<span class="news-date">2025.08.20</span>
<span class="news-content">🎉 KCLNLP's new works <a href="https://x.com/kclnlp/status/1959938651753787832" target="_blank">here</a>, with <span class="news-highlight">8 papers accepted by EMNLP 2025</span> and <span class="news-highlight">1 paper accepted by COLM 2025</span>!</span>
</li>
<li class="news-item">
<span class="news-date">2025.08.20</span>
<span class="news-content">🇨🇳 <span class="news-highlight">2 papers accepted by EMNLP 2025</span>, Check out **[EcoLANG](https://arxiv.org/pdf/2505.06904)** and **[NOVER](https://arxiv.org/pdf/2505.16022)!**
<ul>
<li>NOVER extends RLVR to any task, making a step towards the general reasoning model. See my **[detailed blog](https://thinkwee.top/2025/09/13/general-reasoning/)**.</li>
<li>EcoLANG scales up the agentic social simulation faster and better with self-evolution on communication efficiency.</li>
</ul>
</span>
</li>
<li class="news-item">
<span class="news-date">2025.10.27</span>
<span class="news-content">🚀 <span class="news-highlight">Checkout DDRBench, the first benchmark for Agentic Deep Data Research!</span> DDRBench is a verfiable benchmark where LLMs are asked to dive into the database and explore whatever insight they think that is important. A fully autonomous Data2Insights task w/o any instruction/question. Find more on this **[blog](https://thinkwee.notion.site/ddrbench)!**</span>
</li>
</ul>

# Services
- Program Committee Member
    - ACL(2021,2022,2024)
    - EMNLP(2020,2023,2024,2025)
    - NeurIPS(2024,2025)
    - ICLR(2024,2025,2026)
    - CVPR(2025)
    - AAAI(2025)

# Industrial Experience
-   I spent 3 years of happy time at Tencent, where I bring NLP in the wild of Recommendation & Advertisement.
    -   Improving the NLU ability for News Feed Recommendation.
    -   Resolving the mismatch between commercial inclinations and content interests for Wechat Ads.
    -   Stability, Warm-Up, Efficiency-Quality Tradeoff, Interpretability & Explainability on Large Recommendation System.
    -   Diverse user interest modeling.

# Publications
\* denotes first/co-first author.

-   Personal Agentic AI:
    -   <span class="conf-neurips">NeurIPS 2024</span> <a href="https://arxiv.org/abs/2406.14928" class="bp">paper</a>  <a href="https://github.com/thinkwee/iAgents" class="bc">code</a> Autonomous Agents for Collaborative Task under Information Asymmetry\*
-   Inference Time Scaling:
    -   <span class="conf-emnlp">EMNLP 2025</span> <a href="https://arxiv.org/abs/2505.16022" class="bp">paper</a>  <a href="https://github.com/thinkwee/NOVER" class="bc">code</a> NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning\*
    -   <span class="conf-iclr">ICLR 2025</span> <a href="https://arxiv.org/pdf/2406.07155" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Scaling Large-Language-Model-based Multi-Agent Collaboration
-   Multi-Agents System with LLMs:
    -   <span class="conf-emnlp">EMNLP 2025</span> <a href="https://arxiv.org/pdf/2505.06904" class="bp">paper</a>  <a href="https://github.com/xymou/EcoLANG" class="bc">code</a> EcoLANG: Efficient and Effective Agent Communication Language Induction for Social Simulation
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2307.07924" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Communicative Agents for Software Development
    -   <span class="conf-acl">ACL 2024</span> <a href="https://arxiv.org/abs/2312.17025" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Experiential Co-Learning of Software-Developing Agents
    -   <span class="conf-acl">ACL 2025</span> <a href="https://arxiv.org/pdf/2406.08979" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Multi-Agent Software Development through Cross-Team Collaboration
    -   <span class="conf-arxiv">arXiv 2024</span> <a href="https://arxiv.org/pdf/2405.04219" class="bp">paper</a>  <a href="https://github.com/OpenBMB/ChatDev" class="bc">code</a> Iterative Experience Refinement of Software-Developing Agents
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

# Social Media
<p align="center">
<img data-src="/img/all.png" alt="social" width="800">
</p>
- I sometimes sync some articles from this blog on Wechat and Rednote, and I share other interesting things on these platforms. You can find me via above links.
