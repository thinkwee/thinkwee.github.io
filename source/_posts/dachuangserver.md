---
title: Melodia服务器搭建
date: 2017-05-26 19:18:01
tags: [code,server,linux]
categories: Python
---
***
-	大创项目的服务器端，大创以及客户端介绍见[Melodia客户端](http://thinkwee.top/2017/03/09/dachuang/)
-	我们大创项目的服务器承担的功能比较少，只与android设备收发文件，用Python写了一个简单的服务器端

<!--more-->


# 功能
-	从android客户端接收的数据都是json格式，base64编解码。以我们自定义的特殊字符串结尾，服务器在建立与一个终端的连接后开一个处理的线程
-	客户端完成一次wav到midi转换需要两次通信，第一次json中request值为1，data中是wav文件，服务器负责生成md5编码的时间戳，并用时间戳命名一个wav文件存下，再调用我们的核心程序将wav转换，生成md5.mid和md5.png,即乐曲和曲谱，并回传给客户端md5值。第二次客户端发来的json中request值为2,data值为md5，服务器根据md5索引生成的midi文件回传给客户端

# 代码
```Python
	# -*- coding:utf-8 -*- 
	# ! usr/bin/python

	from socket import *
	import time
	import threading
	import os
	import md5
	import warnings

	Host = ''
	Port = 2017
	Addr = (Host, Port)
	midi_dict = {}

	warnings.filterwarnings("ignore")


	def md5_encode(src):
		m1 = md5.new()
		m1.update(src)
		return m1.hexdigest()


	def tcplink(sock, addr):
		sessnum = 0
		music_data = ''
		while True:
			data = sock.recv(1480)
			if data[-9:]=='endbidou1':
				print 'wav recv finished'
				music_data+=data
				music_data=music_data[:-9]
				midi_data = eval(music_data)
			sessnum = midi_data['request']  
				if midi_data['request'] == 1:
					flag_md5 = md5_encode(str(time.time()))
					print 'md5: ', flag_md5
					wav_name = flag_md5 + '.wav'
					with open(wav_name, 'w+') as f:
						f.write(midi_data['data'].decode('base64'))
						f.close()
					n = midi_data['config']['n'];
					m = midi_data['config']['m'];
					w = midi_data['config']['w'];
					midi_name = flag_md5 + '.mid'
					with open(midi_name, 'w') as f:
						f.close()
					shellmid = '../mldm/hum2midi.py -n '+str(n)+' -m '+str(m)+' -w '+str(w)+' -o ' + midi_name + ' ' + wav_name
			print "running wav2midi shell"
					retmid = os.system(shellmid)
					retmid >= 8
					if retmid == 0:
				print 'generate midi successful'
				shellpng = 'mono ../mlds/sheet '+midi_name+' '+flag_md5
				retpng = os.system(shellpng)
				if retpng == 0:
							sock.send(flag_md5.encode())
							print 'generate png successful'
							midi_dict[flag_md5] = midi_name
							break
				else:
				print 'generate png error'
				break
					else:
						print 'generate midi error'
						break
				elif midi_data['request'] == 2:
					flag = midi_data['data']
					if flag in midi_dict.keys():
						fo = open(flag+'.mid', 'rb')
						while True:
							filedata = fo.read(1024)
							if not filedata:
								break
							sock.send(filedata)
				print 'midi file sent'
						fo.close()
						break
					else:
						print 'can not find midi'
						break
				else:
					print 'json error'
			else:
				music_data += data
		sock.close()
		print 'session '+str(sessnum)+' for '+str(addr)+' finished'

	tcpSerSock = socket(AF_INET, SOCK_STREAM)
	tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
	tcpSerSock.bind(Addr)
	tcpSerSock.listen(5)

	while True:
		tcpCliSock, tcpCliAddr = tcpSerSock.accept()
		print 'add ', tcpCliAddr
		t = threading.Thread(target=tcplink, args=(tcpCliSock, tcpCliAddr))
		t.start()
	tcpSerSock.close()
```