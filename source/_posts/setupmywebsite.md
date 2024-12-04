---
title: note for building a blog
tags:

- web
- hexo
- github
categories:
- Other
author: Thinkwee
date: 2017-01-16 12:01:00

---

***

Always wanted to build my own blog, previously thought of using WordPress, but suffered from laziness, didn't want to mess with the server. Later, I stumbled upon GitHub Pages, which automatically generates a website by uploading a js project, and everything is hosted on GitHub. The official instructions also recommend using this for writing blogs, so I started to try it. The general framework should be that GitHub Pages generates the website from your github.io project on GitHub, Hexo generates the static web page project from your blog content and custom settings, and then uploads it to your repository. To back up, we will establish two branches in the repository: one master for Hexo to upload static web page files, and one hexo for saving the local Hexo project. Below, I share some experiences and pitfalls encountered.


*       2017.2.8 update md writing software
*       2017.2.10 update mathjax cdn, add long gallery, update domain name, access分流(blog2.0)
*       2017.2.13 update optimization plugin, update top description, optimize long gallery, widen article width(blog3.0)
*       2017.3.30 update top description original address
*       2017.12.27 update异地恢复
*       2018.7.6 update a more comprehensive reference website
  
<!--more-->

![i07MIs.png](https://s1.ax1x.com/2018/10/20/i07MIs.png)

{% language_switch %}

{% lang_content en %}

Premise
=======

*   Install Node.js
*   Install git
*   Install your favorite text editing software, such as notepad++, for writing blogs. Alternatively, use plugins and third-party markdown writing software (recommended).
*   Optional: Install the Chrome plugin Minimalist Image Hosting or MPic (image hosting software). Install GeoGebra (for drawing graphs)

GitHub & Hexo Initialization
============================

Choosing hexo as the blog tool, rather than the officially recommended jekyll, is actually quite similar; one is based on Ruby, and the other on Node.js. Hexo is said to be faster. The final process was like this.

*   Registering a GitHub account, this is omitted
*   Create a new repository, the name must be "your\_username.github.io"
*   Create a new hexo branch for this repository and set hexo as the default branch
*   cd to a new local folder (used to store the blog project) in git bash
*   Sequentially execute npm install hexo, hexo init, npm install, npm install hexo-deployer-git
*   At this point, the Hexo blog project has been initialized in your folder. Find \_config.yml in the folder and modify the deploy parameter to master, see the following text.
*   https://your\_username.github.io/
*   The current state is that your blog project exists under the hexo branch, and all changes are saved to the Hexo branch on GitHub, but since the deploy parameter is set to master, using the hexo command to generate blog web pages will update the master branch, so both branches will be updated each time the blog is updated, one for the hexo project and one for updating web pages

**Afterward, all commands are executed in the Git Bash command-line environment, with the working directory being the Hexo blog directory, and note that you should switch to the hexo branch**

Hexo Configuration
==================

The directory format of the Hexo blog is as follows

\- \_config.yml - package.json - scaffolds - source | ├── \_drafts | └── \_posts - themes

*   config.yml sets the overall parameters of the blog here, such as the blog name, URL, and git.
*   \_posts stores your articles
*   theme, as the name implies, stores the blog interface theme
*   In the config.yml, configure the following key parameters:
    *   Blog Website Title
    *   Subtitle: Subtitle
    *   description: One-sentence summary
    *   author: Author Name
    *   language: zh-Hans Language Check the supported languages in the language section of the topic catalog based on the theme you choose
    *   timezone: Asia/Shanghai time zone, this has a standard, China should be written as Shanghai time zone
    *   https://your\_username.github.io/
    *   type: git
    *   repo: https://github.com/your\_username/your\_username.github.io.git
    *   **branch: master**

Update Hexo project
===================

Sequentially execute

*   git add . (Check if all files are updated)
*   git commit -m "Update Report" (submitting commit)
*   git push origin hexo (Upload updates to GitHub) The first update may encounter some errors
*   First, pull then push; otherwise, GitHub will report an error if it finds that some files on the local machine are not on the server
*   Permission denied: Go to the blog directory/.git/config, find the Url, and change the SSH link to HTML format: https://www.github.com/your\_username/your\_username.github.io.git
*   refusing to merge unrelated histories because there is no common ancestor branch to merge, this can only be forced to merge, execute command: git pull origin hexo --allow-unrelated-histories, then run ush all operations are performed under the Hexo branch, because master only stores static web page files, which do not require you to change, it is generated by Hexo

Hexo write blogs and update webpages
====================================

Main commands are as follows

*   hexo clean Clean cache and static files
*   hexo g or hexo generate to generate framework files; it is recommended to clean first and then generate before updating the website
*   hexo s or hexo server to open the local server for preview, enter localhost:4000 as the URL, and press Ctrl+C to close
*   hexo new "filename" creates a new blog post, this filename is not the title of the article, and after creating it, you can open it in the \_posts folder to write the blog
*   hexo d or hexo deploy, after writing the blog, clean & generate, then deploy the blog to GitHub Pages, the blog webpage is updated. There are a few points to note
*   After the first new directory is created, execute \`#npm install hexo-deployer-git --save\` to install git deployment; otherwise, the upload will not work
*   Hexo's grammar specification is very strict; tags: or level headings # must be followed by a space
*   I have some issues updating under the next topic, it's best to generate the server twice for previewing, and use server -s during the preview

Several Choices for Hexo Writing (Windows)
==========================================

*   MarkFlyElephant: Here, the Chrome app version of MarkFlyElephant is recommended, which can be opened offline. Advantages: The interface is simple and clear, editing is free, the preview function is complete, common operations (bold text, inserting images, links, code blocks, quotes) are convenient, and it can be bound to Evernote for article backup. Disadvantages: The operation is a bit 麻烦, and the text needs to be copied back. It is not convenient to open multiple articles, free trial for 10 days, 78 yuan per year, and it looks like this: ![i078zV.jpg](https://s1.ax1x.com/2018/10/20/i078zV.jpg) 
    
*   hexo admin plugin: Install the hexo-admin plugin for visual management of blogs, installation method:
    
        npm install --save hexo-admin
        hexo server -d
        open http://localhost:4000/admin/
        
    
    http://localhost:4000/admin/\* is a web management interface that can manage articles, and it can also write articles, with real-time preview, adding tags, etc., which is equivalent to extracting some functions and making them into a GUI. \*\*Advantages: multi-article management, convenient operation, automatic saving; when editing, you can change the font size and style according to the md syntax. Disadvantages: the preview is not complete. It does not support mathjax mathematical formula preview, the degree of customization is not high; there is a minor bug in editing; there is a small issue with the filename when creating a new article; it is not very good at supporting Chinese\*\*, and it looks like this: ![i073R0.jpg](https://s1.ax1x.com/2018/10/20/i073R0.jpg) 
    
*   Recommend installing both, use a third-party editor for writing new articles or articles that require inserted formula code, and use hexo admin if it's just for regular typing or updating previously written articles.
    
*   Typora: There are many markdown writing software options, most of which allow for editing and previewing simultaneously. Here, I highly recommend a minimalist design software that advocates for the "what you see is what you get" concept: Typora. \*\*Advantages: What you see is what you get, desktop software, complete features, and a convenient experience similar to Word (especially convenient for inserting tables and images). Disadvantages: Does not support synchronization and backup, HTML needs to be designated in a separate area, and editing is slightly less free.\*\* It looks like this: ![i071Gq.jpg](https://s1.ax1x.com/2018/10/20/i071Gq.jpg) 
    
*   Mac systems are recommended to use Mou or Sublime online, I have not used them and do not make any evaluations.
    
*   Always have Notepad++ on hand; there's no problem that Notepad++ can't solve.
    

Topic Setting
=============

*   Generally, Hexo theme creators will open-source their themes on GitHub, and you can directly clone them to your local machine using the command, then change the theme in config.yml to this theme name.
*   Each topic has its own config.yml file, used for further configuration of each topic, which can be modified according to the author's instructions or the readme on GitHub.

Writing Grammar
===============

*   Grammar supports Markdown, can use HTML, LaTeX, and other grammars, but must be declared at the beginning of the article

Insert Image
============

*   Insert images are divided into local and online link insertion; it is recommended to use online links
*   My method is to apply for a Qiniu Cloud account, create a new storage space, obtain the domain name, ak, sk, then install an extension in Chrome, Minimalist Image Hosting, configure the domain name, ak, sk obtained just now in this extension, and then directly click to upload the image, which is it. The extension will automatically generate the markdown statement for inserting the image or an online link, which can be directly written into the text. MPic is the same.

Insert Formula
==============

*   Surround the formula with a pair of $$, grammatical support for LaTeX
    
*   must declare in the beginning of the article like mathjax: true
    
*   Because hexo first uses marked.js for preprocessing and then processes it with MathJax, the backslash \\ may be escaped as \\, which can lead to incorrect display of formulas. There are two solutions: type two backslashes more, or the other solution is to modify the marked.js file, ignore the escaping, and you can search for the method on Baidu.
    
*   Suggest replacing CDN to accelerate:
    
        mathjax:
        enable: true
        per_page: true
        cdn: //cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML
        
    

Tags and Classification
=======================

*   The difference between tags and categories lies in that one article may be assigned to a single category, but may have multiple tags
*   For example, an article on machine learning is categorized under machine learning, but it can have multiple tags such as code, mathematics, and originality
*   categories: Category Name
*   tags: tag\_name or tags: \[tag1, tag2,...\], note not to use Chinese commas

Other beautification
====================

*   I use the Next theme, which allows settings like background, and a quick search on Baidu will find the methods
*   Change the font according to the theme. In "next," the font is changed directly in the "config," using Google Fonts. Due to issues with accessing it in China, the host uses "//fonts.css.network."

Home Article Quantity Setting
=============================

*   Install plugin
    
          npm install --save hexo-generator-index
          npm install --save hexo-generator-archive
          npm install --save hexo-generator-tag
        
    
*   Add the following field to the site configuration file config.yml
    
        index_generator:
        per_page: 5
        
        archive_generator:
        per_page: 20
        yearly: true
        monthly: true
        
        tag_generator:
        per_page: 10
        
    
*   index, archive, and tag prefixes represent the homepage, archive page, and tag page respectively.
    

Add comment feature
===================

*   Subject-dependent, I use the Next theme, which supports Duoshuo by default; you only need to write your own duoshou\_shortname in the theme configuration file
*   Nowadays, it's best to shut down those whatnots and open a background space on LeanCloud instead. You can use gitcomment, Disqus, or Liker. Currently, I use Valine, mainly because it allows comments without the need for login

Optimization Plugin
===================

*   hexo-generated html files have many redundancies; here, it is recommended to install a plugin to compress files, improving efficiency hexo-all-minifier

Top
===

*   Thank you Netcan\_Space for providing the solution, hoping the official theme will add this feature: Add Hexo sticky feature

cnpm
====

*   Use Taobao mirror to install plugins for faster speed, details on Baidu cnpm installation

Reactive Systems Science
========================

*   Using hexo-generator-feed for rss: hexo-generator-feed

Remote Recovery
===============

*   Recently reinstalled the system, restored the local blog, but after cloning from remote and operating locally with hexo, many problems were found, summarized as follows
    *   clone after directly installing dependencies, no need for hexo init, otherwise it will affect the blog's config.yml
    *   This time, the blog was actually completely rebuilt because there were issues with importing the previous settings. It was later discovered that there was no theme in the local blog backup at all! Because my theme was cloned from someone else's repository, and the entire blog was backed up through git, one repository cannot contain another, so the theme and all settings were never backed up. In the future, the .git folder under the theme directory should be deleted.
    *   Remember to install hexo-deployer-git
    *   next has been updated, but some dependencies for the new features still need to be checked, and you need to run npm install yourself
    *   Found that the previous blog was too flashy, so this time I simply removed all the additional features, focusing on writing
    *   The above 2 figures are outdated; you can check the file directory of the blog in my GitHub
    *   There are many modules that have not been backed up due to their long names, and I feel that Baidu Cloud is more reliable
    *   Upgrade MathJax to 2.7.5
    *   Adopt hexo-renderer-kramed and modify the inline.js escaping

This big shot is reliable
=========================

*   Website here: HEXO Website Memo
*   Mentioned is an excellent version control tool that resolves the issue of repositories under themes, with a direct introduction available at hexo-git-backup


{% endlang_content %}

{% lang_content zh %}

# 前提

- 安装好Node.js
- 安装好git
- 安装好一个你最喜欢的文字编辑软件，用于写博客，比如notepad++。或者使用插件和第三方markdown写作软件(推荐)。
- 可选：安装chrome插件极简图床或MPic(图床软件)。安装geogebra(画图像)

# GitHub&Hexo初始化

选择hexo作为博客工具，并没有用官方推荐的jeykell，其实差不多，一个基于ruby,一个基于node.js,hexo据说快一些
最后的过程是这样的

- 注册github账号，这个就省略了
- 新建一个repository，命名必须为“你的账户名.github.io"
- 对这个repository新建一个hexo分支并把hexo设置为默认分支
- git bash cd到你本地新建的一个文件夹（用于存放博客项目）
- 依次执行 npm install hexo、hexo init、npm install、npm install hexo-deployer-git
- 此时你的文件夹中就已经初始化了Hexo博客项目，在文件夹中找到      _config.yml，修改deploy参数为master, 见后文
- 此时你的网址就是：https://你的账户名.github.io/
- 现在的状态就是，你的博客项目存在hexo分支下，所有更改都会保存到github上Hexo分支中，但因为deploy参数为master,所以用hexo命令生成博客网页文件时会更新master分支，因此每次更新博客后两个分支都会更新，一个hexo项目，一个更新网页文件

**之后所有的命令都是在Git Bash命令行环境下，工作目录就是这个Hexo博客目录,并且注意切换到hexo分支**

# Hexo配置

Hexo博客的目录格式如下

├── _config.yml
├── package.json
├── scaffolds
├── source
|   ├── _drafts
|   └── _posts
└── themes

- config.yml在这里设置博客的总体参数，博客名，url，git什么的
- _posts里面存放你的文章
- theme顾名思义，存放博客界面主题
- 在config.yml中关键配置以下几个参数
  * title: 博客网站标题
  * subtitle:     副标题
  * description:     一句话简介
  * author:     作者名
  * language: zh-Hans    语言根据你选择的主题看，在主题目录的language里查看支持哪些语言
  * timezone: Asia/Shanghai 时区，这个有规范，中国就写上海时区
  * url: https://你的用户名.github.io/
  * type: git
  * repo: https://github.com/你的用户名/你的用户名.github.io.git
  * **branch: master**

# 更新Hexo项目

依次执行

- git add .  (检查所有文件是否更新)
- git commit -m "更新报告" (提交commit)
- git push origin hexo (上传更新到github)
  **第一次更新可能出现一些错误**
- 第一次先pull再push,否则GitHub发现本地没有服务器上的一些文件会报错
- permission denied 去博客目录/.git/config中，找到Url,将ssh链接改成html格式：https://www.github.com/你的用户名/你的用户名.github.io.git
- refusing to merge unrelated histories 因为没有公共祖先分支无法合并，这就只能强制合并了，执行命令：git pull origin hexo --allow-unrelated-histories，之后再ush
  **所有的操作都在Hexo分支下进行，因为master只存静态网页文件，不需要你更改，它是由Hexo生成的**

# Hexo 写博客及更新网页

主要命令如下

- hexo clean 清除缓存和静态文件
- hexo g或者hexo generate 生成框架文件 建议每次更新网站之前先clean再generate
- hexo s或者hexo server 打开本地服务器进行预览，网址输入localhost:4000即可，按crtl+c关闭
- hexo new"文件名" 新建博客，这个文件名不是文章标题名，新建之后在_post文件夹里打开就可以写博客了
- hexo d或者hexo deploy，写好博客，clean&generate之后，部署博客到GitHub Pages上，就更新了博客网页
  **其中有几点要注意**
- 第一次新建目录后执行#npm install hexo-deployer-git --save安装git分发，否则上传不了
- hexo的语法规范很严格，博客里tags:或者级标题符号#后面必须接一个空格
- 我在next主题下更新有点问题，最好是两次generate两次server进行预览，预览时用server -s

# Hexo写作的几种选择(Windows)

- **马克飞象**:这里推荐用马克飞象的chrome app版本，可以离线打开。**优点：界面简洁明了，编辑自由，预览功能完整，常见操作(加粗字体插入图片链接代码块引用)方便,可以绑定印象笔记给文章备份。缺点：操作麻烦一点，需要把文本拷回去。 多个文章打开不方便，免费试用10天，78元一年**,它长这个样子：
  ![i078zV.jpg](https://s1.ax1x.com/2018/10/20/i078zV.jpg)
- **hexo admin插件**:安装hexo-admin插件，可视化管理博客，安装方法：
  
  ```Github
  npm install --save hexo-admin
  hexo server -d
  open http://localhost:4000/admin/
  ```
  
  *http://localhost:4000/admin/*就是一个网页管理界面，可以管理文章，同样也能写文章，实时预览，加标签什么的，相当于把部分功能提出来做成了GUI，**优点：多文章管理，操作方便，自动保存,编辑时根据md语法可以在编辑文本改字号字体。缺点：预览不是很完全。不支持mathjax数学公式预览，自定义程度不高,编辑有一点小bug,新建文章时文件名有点小问题，对中文支持不是很好**,它长这个样子：
  ![i073R0.jpg](https://s1.ax1x.com/2018/10/20/i073R0.jpg)
- 推荐是两个都装上，写新文章或者写需要插入公式代码的文章用第三方编辑器，如果只是普通码字或者更新一下以前写过的文章可以用hexo admin。
- **Typora**:markdown写作软件有许多，大部分都是一边编辑一边预览，这里强推一款极简设计，提倡所见即所得概念的软件：**Typora**。**优点：所见即所得，桌面软件，功能完整，近似于word的便捷体验(尤其在插入表格图片方面很方便)。缺点:不支持同步备份，html需要单独划区域,编辑略不自由。**他长这个样子：
  ![i071Gq.jpg](https://s1.ax1x.com/2018/10/20/i071Gq.jpg)
- Mac系统网上推荐使用Mou或者Sublime，我没有用过，也不做评价。
- 常备一个notepad++,出现什么问题没有notepad++解决不了的。

# 主题设置

- 一般来说Hexo的主题制作人都会将主题开源到GitHub上，直接用命令clone将其拷到本地，然后在config.yml中将theme改成这个主题名即可
- 每一个主题下面有自己的config.yml，用于对每个主题进行进一步的配置，根据作者说明或者GitHub上的readme进行更改

# 写作语法

- 语法支持Markdown,可以用html，latex等其他语法，但必须在文章首部中声明

# 插入图片

- 插入图片分本地和在线链接插入，推荐使用在线链接
- 我的方法是申请七牛云账户，新建一个存储空间，拿到域名，ak,sk,然后chrome装一个插件，极简图床，在这个插件上配置好刚刚拿到的域名，ak,sk，之后直接点击上传图片，就可以了，插件会自动生成用于插入图片的markdown语句或者在线链接，直接写进正文即可。MPic同理。

# 插入公式

- 用一对$$将公式围起来，语法支持latex
- 必须在文章首部中声明如mathjax: true
- 因为hexo先用marked.js进行预处理，然后才通过mathjax处理，所以\\可能被转义为\，从而导致公式显示不正确，解决方法有两种，多打两个\\，另外一种是修改marked.js文件，无视转义，可以自行百度方法
- 建议更换cdn以加速:
  
  ```
  mathjax:
  enable: true
  per_page: true
  cdn: //cdn.bootcss.com/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML
  ```

# 标签与分类

- 标签与分类的区别在于1篇文章归于一类，但可能有多个标签
- 比如机器学习的一片文章归于机器学习这个category，但可以有代码、数学、原创等多个标签
- 分类需在首部加上categories: 分类名
- 标签需在首部加上tags: 标签名或者tags: [标签1,标签2....],**注意别打成中文逗号**    

# 其他美化

- 我用的是next主题，可以设置背景之类的，百度一下方法就能找到
- 更换字体，视主题而定，next里面是用config里直接改字体，用的是google fonts,由于国内访问有问题，所以host用//fonts.css.network

# 首页文章数量设置

- 安装插件
  
  ```Git
    npm install --save hexo-generator-index
    npm install --save hexo-generator-archive
    npm install --save hexo-generator-tag
  ```

- 在站点配置文件config.yml中添加如下字段
  
  ```Markdown
  index_generator:
  per_page: 5
  
  archive_generator:
  per_page: 20
  yearly: true
  monthly: true
  
  tag_generator:
  per_page: 10
  ```

- index, archive及tag开头分表代表主页，归档页面和标签页面。

# 添加评论功能

- ~~视主题而定，我用的next主题，默认支持多说，只需要在主题配置文件中写入自己的duoshou_shortname即可~~
- 现在多说那些啥的都关了，最好自己在leancloud上开后台空间，使用gitcomment、Disqus、来必力都行，现在我用的是valine，就冲它不用登录即可评论

# 优化插件

- hexo生成的html文件有许多冗余，这里推荐安装一款插件压缩文件，提高效率
  [hexo-all-minifier](https://github.com/chenzhutian/hexo-all-minifier)

# 置顶

- 感谢Netcan_Space提供解决方案，希望官方theme加入此功能：[添加Hexo置顶功能](http://www.netcan666.com/2015/11/22/%E8%A7%A3%E5%86%B3Hexo%E7%BD%AE%E9%A1%B6%E9%97%AE%E9%A2%98/)

# cnpm

- 使用淘宝镜像安装插件提速，详情百度cnpm安装

# RSS

- 使用hexo-generator-feed使用rss:[hexo-generator-feed](https://github.com/hexojs/hexo-generator-feed)

# 异地恢复

- 最近重装系统，重新恢复了本地博客，但是从远程clone下来并在本地hexo操作之后发现了许多问题，总结如下
  * clone之后直接安装依赖项，不需要hexo init，否则会情况博客的config.yml
  * 这一次其实是完全重新建了博客，因为以前的设置怎么导入怎么有问题，后来才发现当时的本地博客备份里根本没有theme!因为我的theme就是从别人的repository那里clone过来的，而整个博客又是通过git备份的，一个repository中不能包含另外1个repository，所以其实主题及所有设置一直没备份，以后得将主题目录下的.git文件夹删除
  * 记得安装hexo-deployer-git
  * next更新了，但是新功能的一些依赖还是需要看注释，自己npm install
  * 发现其实之前的博客太花哨了，这次干脆把所有附加功能都整没了，专注写作
  * 上面那2张图已经过时了，可以去我的github里看博客的文件目录
  * 有很多module因为名字太长所以也没有完成备份，感觉还是百度云靠谱
  * 升级mathjax到2.7.5
  * 改用hexo-renderer-kramed，并修改inline.js转义

# 这位大佬靠谱

- 网址在这：[HEXO建站备忘录](https://www.vincentqin.tech/posts/build-a-website-using-hexo/)
- 里面提到了一个非常好的版本管理的工具，解决了theme下repository包含的问题，可以直接看介绍[hexo-git-backup](https://github.com/coneycode/hexo-git-backup)

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
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>