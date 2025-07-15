---
title: OJ
date: 2017-03-27 19:47:54
tags:
-    code
-    python
-    c++
-    algorithm
categories:
-    Algo
photo: 
mathjax: true
---

算法刷题目录，方便自己查找回忆复习
之后(2018.9.27)只更新leetcode上的题了，也懒得整理源码了，leetcode上都存了，只记录思路吧

<!--more-->

！[mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170327/195153747.jpg)

{% language_switch %}

{% lang_content en %}

Leetcode
========

*   [LeetCode Algorithm List](https://leetcode.com/problemset/algorithms/)
*   Directly search for the serial number of each question

Sorting
-------

*   75: Given a sequence containing only 0, 1, and 2, sort it to form the form \[0,0,...,0,1,...,1,2,...,2\], and modify it on the original sequence. Drawing inspiration from the Lomuto partition algorithm, maintain three intervals \[0,i\], \[i,j\], \[j,k\] for the ranges of 0, 1, and 2, respectively. According to priority, if the current number is 2 or 1 or 0, first set nums\[k\]=2, k++, if the current number is 1 or 0, then set nums\[j\]=1, j++, and if the current number is 0, set nums\[i\]=0, i++. The further back it is, the more it can cover the previous ones, with higher priority.
*   Given an unordered sequence, find the maximum difference between adjacent elements after sorting the sequence in O(n) time complexity. Using bucket sort, let the maximum and minimum values in the sequence be max and min, respectively. It can be easily deduced that the maximum difference between adjacent elements must be greater than (max - min) / (n - 1), and let this value be gap, where n is the length of the sequence. Divide the value range \[min, max\] of the sequence into nums buckets with an interval of (max - min) / (n - 1), where nums is int((max - min) / gap + 1). Distribute the elements of the sequence into these buckets, storing only the maximum and minimum elements corresponding to each interval. The maximum interval must be at the boundaries between adjacent buckets (the maximum value of the previous bucket and the minimum value of the next bucket), because the maximum interval within a bucket is less than gap. Finally, traverse the buckets to find the result.
*   179: Water problem, given a sequence of numbers, requiring the combination of the numbers into the largest possible number. String sorting

Stack, Queue
------------

*   Merge K sorted linked lists into one sorted linked list, using a min-heap to store all the head nodes of the lists, and then inserting them in order into the new list
*   Arithmetic Expression, Stack, Python's Switch Syntax
*   224: Arithmetic expressions with parentheses, stack, an ingenious method for handling parentheses, if the sign outside the parentheses is negative, multiply by -1; otherwise, multiply by 1
*   373: Find the k groups of number pairs with the minimum sum in 2 ordered arrays, priority queue

Combinatorial Mathematics
-------------------------

*   47: Generate all combinations of n numbers, with repetitions, and sequentially insert numbers into the k+1 spaces of the already generated k-number combinations, use sets for deduplication directly, and in the Top Solution, deduplication is done by checking if the next number to be inserted is a repetition

Search and Find
---------------

*   In a matrix with a specific order, find using two binary searches
*   78: Generate subsets, recursively or iteratively, deciding for each number whether to include it in the subset or not for recursion
*   In a letter matrix, find a word, depth-first search
*   89: Generate Gray code, iterative, adding 1 or 0 at the front each time, Gray code is symmetrically above and below
*   140: Convert a string into a sentence by concatenating with a given dictionary and spaces, memoized search
*   153: Split an ordered array into two parts and concatenate them, then search, a special binary search
*   154:153 If a transformation of repeated digits is added, with a condition that the binary start and end are the adjacent non-repeated first digits
*   240:74 revised version, each row and column is ordered, count, and separately perform binary search on rows and columns, the number must be at the intersection of rows and columns that meet the conditions, and there are other solutions

树
-

*   98: Determine if a binary tree is a search tree, i.e., the left subtree is all less than the node, and the right subtree is all greater than the node; an in-order traversal will show if it is an increasing sequence, using recursion or a stack
*   101: Determine if a binary tree is symmetric, using recursion, note that the recursion is (left.left, right.right) and (left.right, right.left), i.e., the inner and outer pairs of child nodes, and an iterative version can also be implemented using a stack
*   106: Given the inorder and postorder traversals, find the preorder traversal. The last node in the postorder traversal is the root node, find this root node in the inorder traversal and remove it, the nodes to the left of the root node in the inorder traversal are the left subtree, and the nodes to the right are the right subtree; at this point, the last node in the postorder traversal is the root node of the left subtree, and recursion can be used.
*   107: Output the nodes of each level of the tree from bottom to top and from left to right, recording the answers in a two-dimensional array. If the one-dimensional length of the two-dimensional array, which represents the level, is less than the current level of the node being traversed, a new level is created to store the answers. It can be implemented directly using depth-first search (DFS), or by replacing recursion with a stack and a queue; if using a stack, it is DFS, and if using a queue, it is BFS.
*   Construct a binary search tree, recursively
*   114: Give a tree, compress it by preorder traversal to become a tree with only right nodes. The operation of compressing a node is defined as, setting the left subtree to empty, making the right subtree the original left subtree, and appending the original right subtree to the end of the original left subtree, with the tail node of the original left subtree being the last point reached during the preorder traversal of this subtree. Recursively, first compress the left node, then compress the right node, and then backtrack to compress the parent node.
*   144: Obtain the preorder traversal of a tree using a stack, pop a node to record its value, and then push the right node first, followed by the left node
*   Find the maximum value at each level of a binary tree, BFS

Graph Theory
------------

*   130: Water problem, replace all white pieces surrounded by black pieces in a diagram with black pieces, directly find all the white pieces on the edge and store them, then paint the entire board black and restore the white pieces. The stored is the top solution, the writing style is very pythonic.

Math problem
------------

*   Given two sorted sequences, find the median of the merged sequence with a time complexity of O(log(m+n)), so it cannot be done by comparing one by one. The significance of the median is to split the sequence into two parts of equal length, where the smallest number in the larger part is greater than the largest number in the smaller part. Based on this principle, split the two sequences into two parts, with sequence 1 split at position i and sequence 2 split at position j. It needs to be ensured: 1: If the lengths are the same, then $i+j=m-i+n-j$ ; 2: The union of the smaller parts of the two sequences has any number smaller than the union of the larger parts of the two sequences, because the split parts are still ordered, so this condition is $B[j-1] <= A[i] and A[i-1] <= B[j]$ . Once the position i is determined, the position j is also determined, so it is only necessary to perform a binary search to find the position i. Note the parity and boundary conditions.
*   57: Given a set of intervals, insert a new interval and merge. Simulation
*   122: Water, one line of code, find the maximum adjacent element difference in a sequence
*   142: Given a linked list, find the starting point of the loop in the list. It still uses the two-pointer technique, with one fast and one slow pointer starting from the beginning of the list. When they meet for the first time, it indicates that there is a loop and the difference in their steps is the length of the loop. It can also be deduced that the distance from the starting point of the list to the starting point of the loop is equal to the distance from the meeting point to the starting point of the loop, thus allowing the starting point of the loop to be found.
*   166: Write the quotient and divisor, including the display of repeating decimals, a mathematical analysis problem, continuously multiply the decimal by 10 and divide by the divisor, update the remainder, and when the remainder repeats, the decimal repeats
*   172: Find the number of zeros in n!, a problem in mathematical analysis. The zeros come from 5\*2, so it's about how many 5s and their multiples are in n
*   202: Two-pointer technique, a method for finding a loop
*   263: Mathematical Problem
*   264: Mathematical Problem
*   313: Mathematical Problem

String
------

*   Split a string into palindromic substrings; using Python's \[i::-1\] is very convenient and can be done in one line of code
*   242: Water topic, usage of Python dictionaries
*   Given a string, delete the minimum number of parentheses to make all parentheses in the string match, and output all possible cases. Note two points: do it in both normal and reverse order, as there are two schemes for deleting left and right parentheses; output all cases, and store them in a set.
*   451: Frequency statistics of letters, hash, Python dictionary
*   541: Partial string reversal, simulation, note the use of Python slicing

Greed
-----

*   134: Gas stations are arranged in a circle, given the amount of fuel each station can add and the fuel consumed between stations, determine from which station one can complete a full circle. The data guarantees that the answer is unique. Greedy approach: if it's not possible to complete the circle with the first i stations, then set the starting point to station i+1
*   402: Remove k digits from a large number to make the new number smallest, stack, greedy
*   Overlap Interval Problem, Greedy

Dynamic Programming
-------------------

*   Classic problem climbing ladder, Fibonacci sequence, dp
    
*   96: Given a sequence of numbers from 1 to n, ask how many different BSTs can be constructed. Let ans(i) be the number of different BSTs that can be constructed from a sequence of length i. We need to find the relationship between ans(i) and ans(i-1). Introduce the intermediate variable f(i,n), which represents the number of different BSTs that can be constructed with the i-th number as the root and a sequence of length n. Here, a recursive relationship can be found. The left subsequence of the root constructs the left subtree of the root, and there are left different left subtrees. The right subsequence right is the same. Then f(i,n) should be left \* right, i.e., f(i,n) = ans(i-1) \* ans(n-i). At the same time, with different i as the root, there will be completely different partitions. Therefore, $ans(n)=\sum _{i=1}^n f(i,n)$ , merging gives ans(i) = ans(0) \* ans(n-1) + ans(1) \* ans(n-2) + ... + ans(n-1) \* ans(0), with boundary ans(0) = ans(1) = 1.
    
*   139: Provide a dictionary and a string, and determine if the string can be completely composed of words from the dictionary. Define f\[i\] as true if the first i characters of the string can be completely composed of words. Then, traverse each word, with length k, if f\[i-k\] is true, then f\[i\] is true. This can also be converted into a graph and solved using depth-first search (DFS). An edge between i and j indicates that s\[i:j\] is a word, and we need to find a path from 0 to len(s).
    
*   174: The matrix contains positive and negative numbers, and the minimum path is sought without the intermediate value being 0. Dynamic programming, from the end to the beginning.
    
*   312: Given a sequence of numbers, ask how to sequentially eliminate numbers to get the maximum number of coins. The rule for getting coins is: eliminating a number yields the sum of the product of this number and its two adjacent numbers in coins. Interval dp (divide and conquer), the maximum number of coins that can be obtained in a segment is f\[x,y\], which depends on the last elimination position at i, the coins obtained being f\[x,i-1\] + num\[i-1\] \* num\[i\] \* num\[i+1\] + f\[i+1,y\]. Enumerate the position i within this interval, where the positions of the two subintervals on both sides are known, so the interval is enumerated from small to large. There are three layers of loops in total: the outer loop for the length of the interval, the middle loop for the starting position of the interval in the entire sequence, and the inner loop for enumerating the position i within the interval.
    
*   Determine the number of 1s in the binary representation of each number in \[1, num\], and by analyzing several numbers, it is found that for even number n, its binary representation is the binary of n/2 shifted one bit to the left, with the number of 1s unchanged. For odd number n, its binary representation is the binary of n/2 shifted one bit to the left and a 1 added at the end, which means there is one more 1. It is obvious that the state transition equation
    
    $$
    f(x)=
    \begin{cases}
    f(n) & x=2n \\
    f(n)+1 & x=2n+1 \\
    \end{cases}
    $$
    
*   397: If a number is even, divide it by 2; if it is odd, change it to the adjacent number. Ask how many times it takes to become 1. According to the problem, the state transition equation can be written as:
    
    $$
    f(x)=
    \begin{cases}
    f(n)+1 & x=2n \\
    min(f(2n+2)+1,f(2n)+1) & x=2n+1 \\
    \end{cases}
    $$
    
    Odd cases can be simplified to $min(f(2n)+1,f(n+1)+2)$ , so it can be solved by dynamic programming from 1 to n, but it may exceed time limit. The equation can be further simplified: If n % 4 = 3 and n != 3, then f(n) = f(n + 1) + 1. If n % 4 = 1 or n = 3, then f(n) = f(n - 1) + 1. Proof is here.
    
*   472: It is a variant of 139, still determining whether a string can be composed of words, but both the words and the string to be judged are in a dictionary. Each word needs to be checked if it can be completely composed of other words. Since there are no repeated words in the dictionary, the words are first sorted by length, and each word can only be composed of the words that come before it. The problem then transforms into 139, where the ith word is the string to be queried, and the first i-1 words form the dictionary required for the query. Dynamic programming is applied in the same way. The top solution utilizes a trie dictionary tree to accelerate, see the implementation of the dictionary tree in Python.
    

Divide and Conquer
------------------

*   247: Provide an expression without parentheses, and ask how many possible solutions there are when parentheses are added (without requiring duplicates), divide and conquer, take the i-th operation symbol, and recursively solve the partial solutions on the left and right sides of the symbol, usage of map in Python

Poj
===

*   [C++ source code](https://github.com/thinkwee/Poj_Test)
*   Dijkstra
*   Simulation
*   1094: Topological Sorting
*   1328: Greedy, switch to solve variables
*   1753: Enumeration, Bitwise Operations
*   1789: Prim, priority queue
*   1860: Bellman-Ford
*   2109: Greedy, High-Precision Multiplication
*   2965: Enumeration, Bitwise Operations
*   Modeling, Bellman-Ford
*   3295: Simulation, Stack

Intra-school competition
========================

*   [2017pre](http://code.bupt.edu.cn/contest/650/)
*   Finding three ordered numbers in a set that sum to 0, and then calculating the sum of their squares, original problem from LeetCode
*   D, Find the sum of consecutive prime numbers and factorization, POJ original problem, data is large, so there is no need to use a table, and directly judge each input
*   F, the title is misleading, no backtracking is needed, the characteristic equation can be written and solved using the recursive formula
*   Find the largest subset of a set with binding rules, where selecting a number requires selecting several other numbers, bfs
*   H, given the character transformation rules and cost, find the minimum cost to make two strings identical, flyod, note that two letters can be transformed into a third letter, not necessarily mutually, and the two strings may not be of equal length; if they are not of equal length, output -1 directly
*   I, in high school physics, seek the acceleration due to gravity, the equation is difficult to solve, use bisection method to approximate the answer

hiho
====

*   hiho
*   1505: Given a set, ask how many index quadruples (i, j, p, q) satisfy that i, j, p, q are all different, and i < j, p < q, Ai + Aj = Ap + Aq. The 2sum problem, with a small data range, can directly open a hash array, where sum\[i\] records how many groups of two numbers sum to i, and use the principle of inclusion-exclusion.
*   1506: The probability of getting heads m times out of n coin tosses, with different probabilities for each toss to land heads up. Dynamic programming, let dp\[i\]\[j\] be the probability of getting heads up j times after i tosses, with the state transition equation: dp\[i\]\[j\] = dp\[i-1\]\[j-1\] \* a\[i\] + dp\[i-1\]\[j\]; a\[i\] is the probability of getting heads up on the i-th toss, and special treatment is required for j=0.
*   1507: Incorrect record, given a tree, given the root node number, add an error edge in the middle, and find all possible error edges. Note that since only one error edge is added, the situation can be divided into two cases: 1. This error edge connects to the root node, output directly. 2. If the root node is normal, then this error edge must be connected to a node with an in-degree of 2, and there is only one such node. Find this node, remove the two edges connected to it, and start a depth-first search (dfs) from the root node. If the dfs traversal count is n, it means that the tree still holds after removing this edge, which is the error edge. If the count is less than n, it means that the tree is broken and forms a forest after removing this edge, and this edge is not an error edge. Note the case of repeated edges, at this time only one edge is removed, but both edges are output as error edges separately. Since the number of error edges is less than or equal to 2, the dfs count is less than or equal to 2, and the time complexity, including the construction of the graph with removed edges, is O(n).
*   1515: Given some score relationships between classmates (how many points higher or lower A is compared to B), answer q queries in the end. With weighted union-find, each set maintains the score difference between classmates and a certain root student in this set. Each time a relationship is input, merge the union-find sets of the two classmates, and update each union-find set once, where y is merged into x, the relationship value between x and y is s, and the update formula is d\[root of y\] = d\[x\] - d\[y\] - s. In the next iteration, update the values of the entire union-find set. Finally, perform the direct query.

{% endlang_content %}

{% lang_content zh %}

# Leetcode

- [Leetcode算法列表](https://leetcode.com/problemset/algorithms/)
- 直接搜每道题的序号即可

## 排序

- 75:给出一个只包含0，1，2的数列，排序形成[0,0,...,0,1,...,1,2,...,2]的形式，在原数列上修改。借鉴Lomuto partition algorithm，维护三个区间[0,i],[i,j],[j,k]为0，1，2的范围，按优先级，若当前数为2或1或0，先将nums[k]=2，k++，若当前数为1或0，则nums[j]=1,j++，若当前数为0,则nums[i]=0,i++，越往后能覆盖前面的，优先级越高。
- 164：给出一个无序数列，请在O(n)时间复杂度内找到数列有序化后相隔元素差最大值。桶排序，设数列中最大值最小值分别为max,min,易得这个相邻元素差最大值肯定大于(max-min)/(n-1)并设这个值为gap，n是数列长度。将数列的取值区间[min,max]以(max-min)/(n-1)为间隔分成nums个桶，nums是int((max - min) / gap + 1)，把数列各个元素分到桶中，每个桶只存对应区间内的最大元素和最小元素，把那么最大间隔肯定在相邻两个桶的边界处（前一个桶的最大值和后一个桶的最小值），因为桶内的最大间隔小于gap。最后遍历一遍桶就可以了。
- 179：水题，给出一个数列，要求把各个数连起来组合成一个最大的数。字符串排序

## 堆、栈、队列

- 23:合并K个有序链表为1个有序链表，用最小堆存所有链表头节点，依次取出插入到新链表中
- 150:算术表达式，栈，python的switch写法
- 224:带括号的算术表达式，栈,巧妙地方法处理括号，如果括号外是负号则符号乘-1否则乘1
- 373:在2对有序数组中找出有最小和的k组数对，优先队列

## 组合数学

- 47:生成n个数字的所有组合，数字有重复，依次插入数字到已生成k个数字组合的k+1个空中，排重直接用集合，Top Solution中用插入处后一个数字是否重复来排重

## 搜索与查找

- 74:在一个有有特定顺序的矩阵中查找，两次二分查找
- 78:生成子集，递归或者迭代，每个数字分加入子集或者不加入子集进行递归
- 79:在一个字母矩阵中查找单词，深搜
- 89:生成格雷码，迭代，每次在最前面加1或0，格雷码上下对称
- 140:将一个字符串按给定词典加空格成句子，记忆化搜索
- 153:一个有序数组拆成两部分再倒接一起，查找，特殊的二分查找
- 154:153如果加入重复数字的变形，加入一个条件，二分的start和end是相邻不重复的第一个
- 240:74的改版，每一行每一列有序，查数，对行和列分别做二分，数肯定在满足条件的行与列的交集处，还有其他解法

## 树

- 98:判断一颗二叉树是不是查找树，即左子树都小于节点，右子树都大于节点，中序遍历看看是不是递增数列即可，递归或者栈
- 101:判断一棵二叉树是不是对称的，递归，注意递归的是(left.left,right.right)和(left.right,right.left)即内外两队子节点，另外可以用栈实现迭代版本
- 106:给出中序遍历后序遍历，求前序遍历。后序遍历的最后一个是根节点，在中序遍历中找到这个根节点并在后序遍历中删除，根节点以左是左子树，以右是右子树，这时后序遍历的最后一个节点就是左子树的根节点，递归即可
- 107:从下往上从左往右输出树每一层的节点，用二维数组记录答案，如果二维数组的一维长度即层数比当前遍历的节点层数少就新建一层存答案。可以直接dfs，或者用栈和队列替代递归，如果用栈就是dfs，如果用队列就是bfs
- 108:构造一棵二叉查找树，递归
- 114:给一棵树，按前序遍历将其压扁，变成一棵只有右节点的树。将一个节点压扁操作定义为，把左子树置空，右子树变成原左子树，原右子树接在原左子树的尾节点上，原左子树的尾节点就是按前序遍历这个子树遍历到的最后一个点。递归，先压扁左节点，再压扁右节点，再回溯压父节点
- 144:求一棵树的前序遍历，利用栈，弹出一个节点记录值，并先压入右节点，再压入左节点
- 515:求二叉树每一层的最大值，BFS

## 图论

- 130:水题，把一个图中被黑子包围的白子全部替换成黑子，直接从边上找出所有白子并存起来，再将整个棋盘涂黑，把白子复原。存的是top solution，写法很pythonic

## 数学题

- 4:给出两个有序数列，求出两个数列合并后的中位数，要求时间复杂度O(log(m+n))，所以不能直接一个一个比较。中位数的意义是把序列拆成长度相等两部分，大部分中最小数比小部分中最大数大，根据这个原理将两个序列拆成两部分，设两个数列长度分别为m,n，设数列1在i位置拆开，数列2在j位置拆开，则需要保证：1：两个长度一样，则$i+j=m-i+n-j$；2：两个数列的小部分的并集的任意数比两个数列大部分的并集的任意数要小，因为拆开后的部分依然有序，因此这个条件就是$B[j-1] <= A[i] and A[i-1] <= B[j]$，i的位置确定，j的位置就确定，因此只需二分查找i的位置即可。注意奇偶性和边界判断。
- 57：给出一组区间，插入一个新区间并合并。模拟
- 122:水，一行代码，找一个序列中的最大相邻元素差
- 142:给出一个链表，找出链表中环的起点。依然是快慢钟算法，设一快一慢两个指针一起从链表开始走，第一次相遇时说明存在环且他们步数之差是环的长度，而且可以推算出链表起点到环起点的距离等于相遇点到环起点的距离，就可以找到环的起点
- 166:给出除数被除数写出结果，包括循环小数显示，数学分析题，小数不断乘10除除数，更新余数，余数出现重复即小数出现重复
- 172:求n!中0的个数，数学分析题，0来自5*2，就看n中有多少5及其倍数
- 202:快慢指针，求循环的一种方法
- 263:数学题
- 264:数学题
- 313:数学题

## 字符串

- 131:将一个字符串分成回文子串，用python的[i::-1]很好找，也是一行代码
- 242:水题，python字典的使用
- 301:给定字符串，删除最小数目的括号使得字符串中所有括号匹配，输出所有情况。注意两点：正序逆序各做一遍，因为有删左括号和右括号两种方案；所有情况都要输出，用集合存取所有情况
- 451:按频率统计字母，哈希，python字典
- 541:部分字符串翻转，模拟，注意python切片的使用

## 贪心

- 134:加油站排成一圈，给定各个加油站可以加的油和站与站之间路程所耗的油，问可以走完一圈需要从哪个加油站出发，数据保证答案唯一，贪心，如果前i个加油站走不完了，那么出发点就设成i+1
- 402:从一个大数中移除k个数字，使得新的数最小，栈，贪心
- 452:重叠区间问题，贪心

## 动态规划

- 70:经典问题爬梯子，斐波那契数列，dp
- 96:给一个1到n的数列，问可以构造出多少不同的bst。设ans(i)为长度为i的数列能构造不同的bst的个数，我们需要找到ans(i)与ans(i-1)的关系，引入中间变量f(i,n)，指以第i个数字为根构造bst，长度为n的数列能构造不同的bst的个数，这里可以找到递归关系，根左边的子数列构造出根的左子树，设有left种不同的左子树，右数列right同理，则f(i,n)应该是left\*right，即f(i,n)=ans(i-1)\*ans(n-i)，同时以不同的i为根会有完全不同的划分，则$ans(n)=\sum _{i=1}^n f(i,n)$，合并可以得到ans(i)=ans(0) \* ans(n-1) + ans(1) \* ans(n-2) + … + ans(n-1) \* ans(0)，边界ans(0)=ans(1)=1。 
- 139:给出词典，给出一个字符串，判断字符串是否可以完全由词典内的单词构成。设f[i]==true即前字符串前i位都可以完全由单词构成，则遍历每个单词，单词长度为k，若f[i-k]为真则f[i]为真，也可以转换为图用dfs做，i与j之间有一条边说明s[i:j]是一个单词，我们要找到一条路径从0到len(s)
- 174:矩阵中有正数负数，在中途不为0的情况下求最小路径。动态规划，从后往前
- 312:给出一个数列，问怎样依次消去数能得到最多硬币，得硬币的规则：消掉一个数得到这个数和相邻两个数累乘的数量的硬币。区间dp(分治)，一段区间内能得到的最多硬币f[x,y]是看其最后一次消去位置在i，得到的硬币就是f[x,i-1]+num[i-1]\*num[i]\*num[i+1]+f[i+1,y]，在这个区间内枚举位置i,其中需要左右两个小区间的位置已知，因此将区间从小到大枚举。总共三层循环：外循环区间长度，中循环区间在整个序列中的开始位置，内循环枚举区间内位置i。
- 338:求出[1,num]内每个数字二进制中有多少个1，列举几个数分析发现对偶数n，它的二进制就是n/2的二进制左移一位，1的个数不变，对奇数n，它的二进制就是n/2的二进制左移一位并末尾加1，即1的个数多一个，显然状态转移方程
  
  $$
  f(x)=
\begin{cases}
f(n) & x=2n \\
f(n)+1 & x=2n+1 \\
\end{cases}
  $$
- 397:一个数是偶数就除2，是奇数就变到相邻数，问最少几次变成1，根据题意可以写出状态转移方程:
  
  $$
  f(x)=
\begin{cases}
f(n)+1 & x=2n \\
min(f(2n+2)+1,f(2n)+1) & x=2n+1 \\
\end{cases}
  $$
  
  奇数情况可以化简为$min(f(2n)+1,f(n+1)+2)$，这样就可以从1到n动规了，但是会超时
  可以将方程进一步化简
  If n % 4 = 3 and n != 3, then f(n) = f(n + 1) + 1.
  If n % 4 = 1 or n = 3, then f(n) = f(n - 1) + 1.
  [证明在这里](https://discuss.leetcode.com/topic/59350/python-o-log-n-time-o-1-space-with-explanation-and-proof)    
- 472:是139的变式，依然是判断字符串是否可以由单词构成，但单词和要判断的字符串都在一个词典里，每个单词都需要判断是否完全能被其他单词组成。因为词典中无重复单词，所以先按单词长度排序，每个单词只能由它前面的单词构成，然后问题就转变成了139，第i个单词为要查询的字符串，前i-1个单词构成了查询所需的词典，一样的进行动态规划。top solution中利用了trie字典树加速，见[Python中实现字典树](http://thinkwee.top/2017/05/02/trie/#more)

## 分治

- 247:给出一个算式，无括号，问加括号的情况下有多少种可能的解(不要求去重)，分治，取第i个运算符号，对符号左边和右边分别递归求解部分解，python中map的用法

# Poj

- [C++源码](https://github.com/thinkwee/Poj_Test)
- 1062:Dijkstra
- 1068:模拟
- 1094:拓扑排序
- 1328:贪心，换求解变量
- 1753:枚举，位运算
- 1789:Prim，优先队列
- 1860:bellman-ford
- 2109:贪心，高精度乘法
- 2965:枚举，位运算
- 3259:建模，bellman-ford
- 3295:模拟，栈

# 校内赛

- [2017pre](http://code.bupt.edu.cn/contest/650/)
- A,集合中找有序三个数，满足和为0，求这些数平方和，Leetcode原题
- D,求一个数的连续质数和分解，poj原题，数据大，因此没必要打表，直接对每一个输入判断
- F,题目骗人，不需要回溯，递推公式可以写出特征方程求解
- G,找一个集合中的最大子集，有绑定规则，选一个数必须选其他几个数，bfs
- H,给出字符变换规则和代价，求把两个字符串变成一样的最小代价，flyod，注意两个字母可以变成第三个字母，不一定是互相变，两个字符串可能不等长，不等长直接输出-1
- I,高中物理，求重力加速度，方程不好解，二分法逼近答案    

# hiho

- hiho
- 1505：给定一个组，问有多少种下标四元组(i, j, p, q)满足i, j, p, q两两不同，并且i < j, p < q, Ai + Aj = Ap + Aq。2sum问题，数据范围不大，直接开哈希数组，sum[i]记录两个数之和为i有多少组，利用容斥定理解。 
- 1506：投掷硬币，投n次m次朝上的概率是多少，每次投正面朝上概率不同。动态规划，记dp[i][j]为投前i次j次朝上的概率，状态转移方程为:dp[i][j]=dp[i-1][j-1]a[i]+dp[i-1][j](1.0-a[i]);a[i]为第i次正面朝上的概率，注意对j=0进行特判。
- 1507：错误的记录，给出一棵树，给出根节点编号，在中间加一条错误边，找出所有可能的错误边。注意因为只加了一条错误边，因此情况可以分为两种：1.这条错误边连向根节点，直接输出。2.根节点正常，则这条错误边一定连在入度为2的节点上，且这样的节点只有一个，找到这个节点，将连接它的两条边分别去掉并从根节点开始dfs，如果dfs遍历完次数为n，说明这条边去掉后树依然成立，这是错误边，如果次数小于n，说明去掉这条边树就断了，形成森林，这条边不是错误边。注意边重复的情况，这时候只去掉其中一条，但是两条边是分别当错误边输出。因为错误边数小于等于2，所以dfs的次数小于等于2，加上构造去掉边的图，时间复杂度是O(n)。
- 1515：给定一些同学之间的分数关系（A比B高或低多少分），最后对q个查询给出答案。带权并查集，每个集维护同学到这个集某一个根同学的分数差。每次输入一条关系便合并两个同学所在的并查集，并对每个并查集做一次更新,加入y被x合并，x,y之间关系值是s,更新公式是d[root of y]=d[x]-d[y]-s，并在下一次循环中更新整个并查集的值。最后直接查询。

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