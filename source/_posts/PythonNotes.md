title: Python特性拾零
date: 2017-03-28 20:02:39
tags:
-	code
-	python
categories:
-	Python
---
***
Python的一些特性和语法
总结一些自己跳过的坑
Python3.5

<!--more-->


# 对象皆引用
-	不同于C++，Python只有传引用，不存在传值，在Python中，一切皆对象，变量只是对对象的引用
-	Python中不需要声明变量类型也是基于此
	```Python
		a=3
		b=a
	```
	a和b都只是引用一个整型值3，修改b，a的引用值也会变化
-	如果要拷贝，可以用b=a[:]

# string是常量
-	字符串不能更改，只能在原先字符串上更改后赋给一个新字符串，原字符串依然不变

# lambda匿名函数
-	简化函数书写，lambda 参量:计算式
-	主要用于排序或者reduce
-	lambda的参数是自由变量，是运行时绑定值，而不是定义时绑定值，如果要实现定义时绑定值，则定义之后在lambda中设置默认参数
	```Python
		x=10
		a=lambda y,x=x:x+y
		x=20
		print(a(10))
		
	>>> 20
	```
	如果不设置默认参数，上面的运行结果就是30
	
# 迭代器与生成器
-	通过重写对象的__iter__方法实现自定义迭代器，生成器yield实现迭代器的next方法
	```Python
		class Countdown:
			def __init__(self, start):
				self.start = start
		
			def __iter__(self):
				n = self.start
				while n > 0:
					yield n
					n -= 1
    
			def __reversed__(self):
				n = 1
				while n <= self.start:
					n += 1
    
    
		for rr in (Countdown(3)):
			print(rr)
		
	>>> 3
	    2
	    1
	```

# enumerate
-	同时输出迭代对象和索引，参数为索引开始号
	```Python
		for idx,val in enumerate(my_list,1):
			print(idx,val)
	```

# 函数
-	接收任意个参数
	```Python
	def avg(first,*rest):
		return (first+sum(rest))/(1+len(rest))
	```
	\*接任意数量的位置参数，也可以用\*\*接一个字典，代表任意数量的关键字参数，也可以混用\*和\*\*
	顺序(任意个位置参数，\*，最后一个位置参数，其他参数，\*\*)
-	函数返回多个值
	直接return a,b,c，实际上返回的是一个元祖

# 装饰器
-	一个装饰器就是一个函数，它接收一个函数作为参数并返回一个新的函数
	```Python
		import time
		from functools import wraps
		
		
		def timethis(func):
			@wraps(func)
			def wrapper(*args, **kwargs):
				start = time.time()
				result = func(*args, **kwargs)
				end = time.time()
				print(func.__name__, end - start)
				return result
		
			return wrapper
		
		@timethis
		def loop(n):
			while n > 0:
				n -= 1
		
		
		loop(100000)
	
	>>> loop 0.03971695899963379
	```
	在上例中，timethis是包装器，其定义中func是被包装的函数，args和kwargs是任意数量的位置参数和关键字参数，来保证被包装的函数能正确接收参数执行
	可以看到包装器中实现了一个wrapper装饰器函数，它运行了作为参数的func函数并计算打印了运行时间，一般装饰器函数返回原函数的执行结果
-	可以看到timethis中的@wraps本身也是一个装饰器，它用来注解底层包装函数，这样能够保留原函数的元信息，还能通过装饰器返回函数的__wrapped__属性直接访问到被装饰的函数，用来解除装饰

# 逗号的特殊作用
-	输出时换行变空格
-	转换类型为元组

# filter
-	接收一个函数和序列，将函数作用于序列中每一个元素上，根据返回值决定是否删除该元素
	```Python
		def is_odd(n):
			return n % 2 == 1

		filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15])
	```

# any
-	原型：
	```Python				
		def any(iterable):
		   for element in iterable:
			   if  element:
				   return False
		   return True
	```

# yield
-	一个带有yield的函数就是一个generator，它和普通函数不同，生成一个generator看起来像函数调用，但不会执行任何函数代码，直到对其调用next()（在for循环中会自动调用next()）才开始执行。虽然执行流程仍按函数的流程执行，但每执行到一个yield语句就会中断，并返回一个迭代值，下次执行时从yield的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield中断了数次，每次中断都会通过 yield 返回当前的迭代值。
	```Python
		>>> def g(n):
		...     for i in range(n):
		...             yield i **2
		...
		>>> for i in g(5):
		...     print i,":",
		...
		0 : 1 : 4 : 9 : 16 :
	```