---

title: Trie in Python
date: 2017-05-02 10:09:19
tags:

- code
- python
categories:
- Python

---

***

在Python中有字典这一数据结构，因此用Python实现字典树很方便

<!--more-->

![i07rz6.jpg](https://s1.ax1x.com/2018/10/20/i07rz6.jpg)

# 字典树(trie)

- 字典树主要用于词频统计，前缀后缀相关的算法，树的根节点不存任何字符，每一条边代表一个字符，其他每一个节点代表从根节点到此节点的所有边上字符构成的单词，存的内容根据需求而定。
- 字典树快的原因就是充分利用的单词的共同前缀，如果前缀都不一样，就不需要继续查找
- 一个单词不一定在叶子节点，因为它可能构成其他更长单词的前缀，因此如果用于词频统计，则可以插入完一个单词后在此单词最后一个节点中count++。如果仅仅用于判断某个词是否在字典树构成的字典中，则可以在插入完一个单词后，在最后一个节点中添加一个None节点，内容为单词本身

# 例子

- 以leetcode472为例，当中需要判断某个序列中某个单词是否能由它前面的单词构成
- 初始化trie
  
  ```Python
     trie = {}
  ```
- trie中插入单词
  
  ```Python
     def insert(trie, w):
         for c in w:
             trie = trie.setdefault(c, {})  # if c not in trie then set trie[c]={}
         trie[None] = w
  ```
  
  对每一个字符串，依次按字母索引，当索引不到时就对当前字母建立新节点
  遍历完单词后建立None节点存放单词(因为题目需要返回所有单词，因此此处存放单词,也可以存放出现次数
  这样一棵树最终完成时就如标题图所示，其中前四行是加入["cat", "cats","dog", "rat"]之后树的内容
- trie中查找某一个单词
  
  ```Python
     def prefixs(trie, w, lo):
         for i in range(lo, len(w)):
             trie = trie.get(w[i])
             if trie is None:
                 break
             prefix = trie.get(None)
         if prefix:
             yield i + 1, prefix
  ```
  
  因为题目需要，利用了生成器，这段函数是查找单词w中i从lo位置开始，i到单词尾这一段构成的字符串，是否在trie的字典集合中，返回所有符合结果的i+1。查找的方式与插入相同
  特别说明的是，trie的一大优势便是支持插入与查找同时进行

# 后缀树

- 如果将一个单词，长度为l，拆分成l个子单词插入到trie中，每个子单词都是这个单词的[i:l]构成的后缀，则这样的字典树称之为后缀树。
- 简单按上述方法建立后缀树会存在许多冗余信息，在空间上还可以进行优化
- 待续