---
title: Android:Melodia客户端
date: 2017-03-09 17:19:53
tags: [code,android]
categories: Android
---
***
学校大创项目简单的app
实现功能，录制声音存为wav，利用json与服务器通信，上传wav到服务器，服务器转为midi文件，从服务器下载midi和乐谱并播放，同时电子琴改装后也可以与服务器通信，由手机给电子琴辅助参数，电子琴通过arduino从服务器上读取乐曲中间键值文件播放。

<!--more-->

![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170309/172109957.gif)
封面图使用[qiao](https://github.com/qiao)的midi在线可视化工具[euphony](https://github.com/qiao/euphony)

# midi播放
调用MediaPlayer类播放，因为不可抗因素，只能用android5.1，没有midi库，就做简单的播放
-	MediaPlayer可以用外部存储，assert,自建raw文件夹或者uri四种方式访问媒体文件并播放
-	从raw文件夹中读取可以直接用player = MediaPlayer.create(this, R.raw.test1)
-	Uri或者外部存储读取new->setDataSource->prepare->start

# 录制声音并重放
参考[android中AudioRecord使用](http://blog.csdn.net/jiangliloveyou/article/details/11218555)
```Java
	private class RecordTask extends AsyncTask<Void, Integer, Void> {
		@Override
		protected Void doInBackground(Void... arg0) {
			isRecording = true;
			try {
				//开通输出流到指定的文件
				DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(pcmFile)));
				//根据定义好的几个配置，来获取合适的缓冲大小
				int bufferSize = AudioRecord.getMinBufferSize(audioRate, channelConfig, audioEncoding);
				//实例化AudioRecord
				AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.MIC, audioRate, channelConfig, audioEncoding, bufferSize);
				//定义缓冲
				short[] buffer = new short[bufferSize];

				//开始录制
				record.startRecording();

				int r = 0; //存储录制进度
				//定义循环，根据isRecording的值来判断是否继续录制
				while (isRecording) {
					//从bufferSize中读取字节，返回读取的short个数
					//这里老是出现buffer overflow，不知道是什么原因，试了好几个值，都没用，TODO：待解决
					int bufferReadResult = record.read(buffer, 0, buffer.length);
					//循环将buffer中的音频数据写入到OutputStream中
					for (int i = 0; i < bufferReadResult; i++) {
						dos.writeShort(buffer[i]);
					}
					publishProgress(new Integer(r)); //向UI线程报告当前进度
					r++; //自增进度值
				}
				//录制结束
				record.stop();
				convertWaveFile();
				dos.close();
			} catch (Exception e) {
				// TODO: handle exception
			}
			return null;
		}
	}
```

# pcm写头文件转成wav
因为录制的是裸文件，pcm格式，需要自己加上wav头
```Java
	private void WriteWaveFileHeader(FileOutputStream out, long totalAudioLen, long totalDataLen, long longSampleRate,
									 int channels, long byteRate) throws IOException {
		byte[] header = new byte[45];
		header[0] = 'R'; // RIFF
		header[1] = 'I';
		header[2] = 'F';
		header[3] = 'F';
		header[4] = (byte) (totalDataLen & 0xff);//数据大小
		header[5] = (byte) ((totalDataLen >> 8) & 0xff);
		header[6] = (byte) ((totalDataLen >> 16) & 0xff);
		header[7] = (byte) ((totalDataLen >> 24) & 0xff);
		header[8] = 'W';//WAVE
		header[9] = 'A';
		header[10] = 'V';
		header[11] = 'E';
		//FMT Chunk
		header[12] = 'f'; // 'fmt '
		header[13] = 'm';
		header[14] = 't';
		header[15] = ' ';//过渡字节
		//数据大小
		header[16] = 16; // 4 bytes: size of 'fmt ' chunk
		header[17] = 0;
		header[18] = 0;
		header[19] = 0;
		//编码方式 10H为PCM编码格式
		header[20] = 1; // format = 1
		header[21] = 0;
		//通道数
		header[22] = (byte) channels;
		header[23] = 0;
		//采样率，每个通道的播放速度
		header[24] = (byte) (longSampleRate & 0xff);
		header[25] = (byte) ((longSampleRate >> 8) & 0xff);
		header[26] = (byte) ((longSampleRate >> 16) & 0xff);
		header[27] = (byte) ((longSampleRate >> 24) & 0xff);
		//音频数据传送速率,采样率*通道数*采样深度/8
		header[28] = (byte) (byteRate & 0xff);
		header[29] = (byte) ((byteRate >> 8) & 0xff);
		header[30] = (byte) ((byteRate >> 16) & 0xff);
		header[31] = (byte) ((byteRate >> 24) & 0xff);
		// 确定系统一次要处理多少个这样字节的数据，确定缓冲区，通道数*采样位数
		header[32] = (byte) (1 * 16 / 8);
		header[33] = 0;
		//每个样本的数据位数
		header[34] = 16;
		header[35] = 0;
		//Data chunk
		header[36] = 'd';//data
		header[37] = 'a';
		header[38] = 't';
		header[39] = 'a';
		header[40] = (byte) (totalAudioLen & 0xff);
		header[41] = (byte) ((totalAudioLen >> 8) & 0xff);
		header[42] = (byte) ((totalAudioLen >> 16) & 0xff);
		header[43] = (byte) ((totalAudioLen >> 24) & 0xff);
		header[44] = 0;
		out.write(header, 0, 45);
	}
```

# json收发
根据我们的实际情况，发送时使用json，存三个参数和wav内容，因为录音的wav时长较短，可以把整个wav写入json中
json发送两次，第一次发送参数和文件，拿到md5编码的时间戳，第二次把这个时间戳加入json中请求相应的midi文件
```Java
    private JSONObject makejson(int request, String identifycode, String data) {
        if (identifycode == "a") {
            try {
                JSONObject pack = new JSONObject();
                pack.put("request", request);
                JSONObject config = new JSONObject();
                config.put("n", lowf);
                config.put("m", highf);
                config.put("w", interval);
                pack.put("config", config);
                pack.put("data", data);
                return pack;
            } catch (JSONException e) {
                e.printStackTrace();
            }
        } else {
            try {
                JSONObject pack = new JSONObject();
                pack.put("request", request);
                pack.put("config", "");
                pack.put("data", identifycode);
                return pack;
            } catch (JSONException e) {
                e.printStackTrace();
            }

        }
        return null;
    }
```
# socket通信
单开一个线程用于启动socket，再开一个线程写两次json收发
注意收发json时将json字符串用base64解码编码，java自己的string会存在错误
另外因为wav字符串较长，服务器接收时分块接收，正常做法是加一个字典项存wav长度，按长度读取wav，然后这里我们偷懒直接在文件尾加了一个特殊字符段用于判断是否接收完成，"endbidou"，不要问我是什么意思，做转换算法的兄弟想的
```Java
private class MsgThread extends Thread {
        @Override
        public void run() {
            File file = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/data/files/Melodia.wav");
            FileInputStream reader = null;
            try {
                reader = new FileInputStream(file);
                int len = reader.available();
                byte[] buff = new byte[len];
                reader.read(buff);
                String data = Base64.encodeToString(buff, Base64.DEFAULT);
                String senda = makejson(1, "a", data).toString();
                Log.i(TAG, "request1: " + senda);
                OutputStream os = null;
                InputStream is = null;
                DataInputStream in = null;
                try {
                    os = soc.getOutputStream();
                    BufferedReader bra = null;
                    os.write(senda.getBytes());
                    os.write("endbidou1".getBytes());
                    os.flush();
                    Log.i(TAG, "request1 send successful");
                    if (soc.isConnected()) {
                        is = soc.getInputStream();
                        bra = new BufferedReader(new InputStreamReader(is));
                        md5 = bra.readLine();
                        Log.i(TAG, "md5: " + md5);
                        bra.close();
                    } else
                        Log.i(TAG, "socket closed while reading");
                } catch (IOException e) {
                    e.printStackTrace();
                }
                soc.close();
                startflag = 1;

                StartThread st = new StartThread();
                st.start();

                while (soc.isClosed()) ;

                String sendb = makejson(2, md5, "request2").toString();
                Log.i(TAG, "request2: " + sendb);
                os = soc.getOutputStream();
                os.write(sendb.getBytes());
                os.write("endbidou1".getBytes());
                os.flush();
                Log.i(TAG, "request2 send successful");

                is = soc.getInputStream();
                byte buffer[] = new byte[1024 * 100];
                is.read(buffer);
                Log.i(TAG, "midifilecontent: " + buffer.toString());
                soc.close();
                File filemid = new File(Environment.getExternalStorageDirectory().getAbsolutePath() + "/data/files/Melodia.mid");
                FileOutputStream writer = null;
                writer = new FileOutputStream(filemid);
                writer.write(buffer);
                writer.close();
                Message msg = myhandler.obtainMessage();
                msg.what = 1;
                myhandler.sendMessage(msg);
            } catch (IOException e) {
                e.printStackTrace();
            }


        }
    }
```

# 录音特效
录音图像动画效果来自Github：[ShineButton](https://github.com/ChadCSong/ShineButton)
另外录音按钮做了个效果，按住录音，松开完成，往外滑一定距离取消
```Java
	fabrecord.setOnTouchListener(new View.OnTouchListener() {
				@Override
				public boolean onTouch(View v, MotionEvent event) {
					switch (event.getAction()) {
						case MotionEvent.ACTION_DOWN:
							uploadbt.setVisibility(View.INVISIBLE);
							if (isUploadingIcon) {
								isPressUpload = false;
								uploadbt.performClick();
								isPressUpload = true;
								isUploadingIcon = !isUploadingIcon;
							}

							Log.i(TAG, "ACTION_DOWN");
							if (!shinebtstatus) {
								shinebt.performClick();
								shinebtstatus = true;
							}
							ox = event.getX();
							oy = event.getY();

							isRecording = true;
							recLen = 0;
							recTime = 0;
							pb.setValue(0);
							fabrecord.setImageResource(R.drawable.ic_stop_white_24dp);
							Snackbar.make(fabrecord, "开始录音", Snackbar.LENGTH_SHORT)
									.setAction("Action", null).show();

							recorder = new RecordTask();
							recorder.execute();
							handler.postDelayed(runrecord, 0);

							break;
						case MotionEvent.ACTION_UP:
							handler.removeCallbacks(runrecord);
							Log.i(TAG, "ACTION_UP");
							if (shinebtstatus) {
								shinebt.performClick();
								shinebtstatus = false;
							}
							float x1 = event.getX();
							float y1 = event.getY();
							float dis1 = (x1 - ox) * (x1 - ox) + (y1 - oy) * (y1 - oy);

							isRecording = false;
							pb.setValue(0);
							fabrecord.setImageResource(R.drawable.ic_fiber_manual_record_white_24dp);
							if (dis1 > 30000) {
								Snackbar.make(fabrecord, "取消录音", Snackbar.LENGTH_SHORT)
										.setAction("Action", null).show();
							} else {
								if (!isUploadingIcon) {
									uploadbt.setVisibility(View.VISIBLE);
									isPressUpload = false;
									uploadbt.performClick();
									isPressUpload = true;
									isUploadingIcon = !isUploadingIcon;
								} else {

								}

								Snackbar.make(fabrecord, "录音完成", Snackbar.LENGTH_SHORT)
										.setAction("Action", null).show();
								handler.postDelayed(runreplay, 0);
								replay();
							}
							break;
						case MotionEvent.ACTION_MOVE:
							float x2 = event.getX();
							float y2 = event.getY();
							float dis2 = (x2 - ox) * (x2 - ox) + (y2 - oy) * (y2 - oy);
							if (dis2 > 30000) {
								fabrecord.setImageResource(R.drawable.ic_cancel_white_24dp);
							} else {
								fabrecord.setImageResource(R.drawable.ic_stop_white_24dp);
							}
							break;
					}
					return true;
				}
			});
```
# 展示乐谱
-	本来是想通过socket收发图片，后来觉得太麻烦于是把方案改成Apache对每一次转换生成相应的图片链接，通过时间戳md5直接在线访问，如果需要分享图片则先存到本地再分享
```Java
    public void init() {
        md5 = getArguments().getString("md5");
        final String imageUri = "服务器地址" + md5 + "_1.png";
        Log.i("play", "pngfile: " + imageUri);
        new Handler().postDelayed(new Runnable() {
            public void run() {
                //execute the task
                imageLoader.displayImage(imageUri, showpic);
            }
        }, 2000);

    }
```

# 与电子琴通信
-	类似于上传服务器，也是socket通信，电子琴改装了之后从手机客户端接收八度、速度两个参数，arduino接收到参数就播放，并由arduino断开连接
```Java
	pianobt.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					if (!isconnected) {
						pianoaddr = etpianoaddr.getText().toString();
						pianoport = Integer.valueOf(etpianoport.getText().toString());
						param[0] = 0x30;
						StartThread st = new StartThread();
						st.start();
						while (!isconnected) ;
						MsgThread ms = new MsgThread();
						ms.start();
						YoYo.with(Techniques.Wobble)
								.duration(300)
								.repeat(6)
								.playOn(seekBaroctave);
						while (soc.isConnected()) ;
						try {
							soc.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
						isconnected = false;
						Log.i("piano", "socket closed");
					}


				}
			});

			samplebt.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					pianoaddr = etpianoaddr.getText().toString();
					pianoport = Integer.valueOf(etpianoport.getText().toString());
					param[0] = 0x31;
					StartThread st = new StartThread();
					st.start();
					while (!isconnected) ;
					MsgThread ms = new MsgThread();
					ms.start();
					YoYo.with(Techniques.Wobble)
							.duration(300)
							.repeat(6)
							.playOn(seekBaroctave);
					while (soc.isConnected()) ;
					try {
						soc.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
					isconnected = false;
					Log.i("piano", "socket closed");

				}
			});


		}

		private class StartThread extends Thread {
			@Override
			public void run() {
				try {
					soc = new Socket(pianoaddr, pianoport);
					if (soc.isConnected()) {//成功连接获取soc对象则发送成功消息
						Log.i("piano", "piano is Connected");
						if (!isconnected)
							isconnected = !isconnected;

					} else {
						Snackbar.make(pianobt, "启动电子琴教学失败", Snackbar.LENGTH_SHORT)
								.setAction("Action", null).show();
						Log.i("piano", "Connect Failed");
						soc.close();
					}
				} catch (IOException e) {
					Snackbar.make(pianobt, "启动电子琴教学失败", Snackbar.LENGTH_SHORT)
							.setAction("Action", null).show();
					Log.i("piano", "Connect Failed");
					e.printStackTrace();
				}
			}
		}

		private class MsgThread extends Thread {
			@Override
			public void run() {
				try {
					OutputStream os = soc.getOutputStream();
					os.write(param);
					os.flush();
					Log.i("piano", "piano msg send successful");
					Snackbar.make(pianobt, "正在启动启动电子琴教学", Snackbar.LENGTH_SHORT)
							.setAction("Action", null).show();

					soc.close();
				} catch (IOException e) {
					Log.i("piano", "piano msg send successful failed");
					Snackbar.make(pianobt, "启动电子琴教学失败", Snackbar.LENGTH_SHORT)
							.setAction("Action", null).show();
					e.printStackTrace();
				}

			}
		}
```

# 乐谱分享
-	显示乐谱的是Github上一个魔改的ImageView:[PinchImageView](https://github.com/boycy815/PinchImageView)
-	定义其长按事件，触发一个分享的intent
```Java
	showpic.setOnLongClickListener(new View.OnLongClickListener() {
				@Override
				public boolean onLongClick(View v) {
					Bitmap drawingCache = getViewBitmap(showpic);
					if (drawingCache == null) {
						Log.i("play", "no img to save");
					} else {
						try {
							File imageFile = new File(Environment.getExternalStorageDirectory(), "saveImageview.jpg");
							Toast toast = Toast.makeText(getActivity(),
									"", Toast.LENGTH_LONG);
							toast.setGravity(Gravity.TOP, 0, 200);
							toast.setText("分享图片");
							toast.show();
							FileOutputStream outStream;
							outStream = new FileOutputStream(imageFile);
							drawingCache.compress(Bitmap.CompressFormat.JPEG, 100, outStream);
							outStream.flush();
							outStream.close();

							Intent sendIntent = new Intent();
							sendIntent.setAction(Intent.ACTION_SEND);
							sendIntent.putExtra(Intent.EXTRA_STREAM, Uri.fromFile(imageFile));
							sendIntent.setType("image/png");
							getActivity().startActivity(Intent.createChooser(sendIntent, "分享到"));

						} catch (IOException e) {
							Log.i("play", "share img wrong");
						}
					}
					return true;
				}
			});
```