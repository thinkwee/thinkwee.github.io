---
title: A summary of my Android apps:BuptRoom
date: 2017-01-16 11:56:39
tags: [code,android]
categories: Android

---

***

# Introduction


Write an app to query the school's empty classrooms
Pull information from the school's registration website, classify and display it, and add some miscellaneous things
After all, it's my first time writing Android, so I want to try everything
Download here: [BuptRoom](https://fir.im/buptroom)
repository address: [A simple Beiyou self-study room query system](https://github.com/thinkwee/BuptRoom)
It took about 3 weekends to complete the first version, and then I spent about 1 month updating miscellaneous things
After that, I spent about 1 month updating miscellaneous things
Many things written in an unstandardized manner, and I just looked up and used them temporarily
Summarize the experience of writing the App:

<!--more-->

{% language_switch %}

{% lang_content en %}

# Learning content

- Android basic architecture, components, lifecycle
- Fragment
- Java library and library calls
- Github usage
- Deploy app
- Image processing methods
- A stupid way to pull web content
- Utilize GitHub third-party libraries
- Color knowledge
- Android Material Design
- Simple optimization
- Multithreading and Handler

# Solved problems

Mainly solved the following problems:

- Android 6.0 and above versions seem to require dynamic permission verification, and the current version only supports 5.0 and below, used permisson:

```Java
    <uses-permission android:name="android.permission.INTERNET"></uses-permission>
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW"></uses-permission>
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"></uses-permission>
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"></uses-permission>
    <uses-permission android:name="android.permission.MOUNT_UNMOUNT_FILESYSTEMS"></uses-permission>
```

- The webpage is a jsp dynamic webpage, which cannot be simply parsed, so I finally used loadurl in webview to execute javascript commands, and need to download the jsoup-1.9.2.jar package and add it to the library file

```Java
    final class MyWebViewClient extends WebViewClient {
        public boolean shouldOverrideUrlLoading(WebView view, String url) {
            view.loadUrl(url);
            return true;
        }
        public void onPageStarted(WebView view, String url, Bitmap favicon) {
            Log.d("WebView","onPageStarted");
            super.onPageStarted(view, url, favicon);
        }
        public void onPageFinished(WebView view, String url) {
            Log.d("WebView","onPageFinished ");
            view.loadUrl("javascript:window.handler.getContent(document.body.innerHTML);");
            super.onPageFinished(view, url);
        }
    }
```

- Write a handler to respond to javascript commands, so that the body content in the html file is obtained in String form

```Java
    final  class JavascriptHandler{
        @JavascriptInterface
        public void getContent(String htmlContent){
            Log.i(Tag,"html content: "+htmlContent);
            document= Jsoup.parse(htmlContent);
            htmlstring=htmlContent;
            content=document.getElementsByTag("body").text();
            Toast.makeText(MainActivity.this,"加载完成",Toast.LENGTH_SHORT).show();
        }
    }
```

- After that, string processing, according to the format provided by the registration website

```Java
    Remove commas
    String contenttemp=content;
    content="";
    String[] contentstemp=contenttemp.split(",");
    for (String temp:contentstemp){
        content=content+temp;
    }

    Group
    contents=content.split(" |:");
    String showcontent="";
    count=0;
    int tsgflag=0;
    int cishu=0;
    j12.clear();
    j34.clear();
    j56.clear();
    j78.clear();
    j9.clear();
    j1011.clear();
    if (keyword.contains("图书馆")) tsgflag=1;
    for (String temp:contents){
        if (temp.contains(keyword)){
            cishu++;
            SaveBuidlingInfo(count,cishu,tsgflag);
        }
        count++;
    }

    SaveBuildingInfo is to classify and store classrooms by building, and then classify and store them by time period in j12,j34.....
    while (1 == 1) {
        if (contents[k].contains("楼") || contents[k].contains("节") || contents[k].contains("图"))
            break;
        ;
        switch (c) {
            case 1:
                j12.add(contents[k]);
                break;
            case 2:
                j34.add(contents[k]);
                break;
            case 3:
                j56.add(contents[k]);
                break;
            case 4:
                j78.add(contents[k]);
                break;
            case 5:
                j9.add(contents[k]);
                break;
            case 6:
                j1011.add(contents[k]);
                break;
            default:
                break;
        }
        k++;
    }
```

- The NavigationView is a simple one, with no special design, because there are no multiple interfaces, just use the refresh TextView to fake multiple interfaces

- Tried the MaterialDesign components, added some things about system time

```Java
    final Calendar c = Calendar.getInstance();
     c.setTimeZone(TimeZone.getTimeZone("GMT+8:00"));
     mYear = String.valueOf(c.get(Calendar.YEAR)); // 获取当前年份
     mMonth = String.valueOf(c.get(Calendar.MONTH) + 1);// 获取当前月份
     mDay = String.valueOf(c.get(Calendar.DAY_OF_MONTH));// 获取当前月份的日期号码
     mWay = String.valueOf(c.get(Calendar.DAY_OF_WEEK));
     mHour= c.get(Calendar.HOUR_OF_DAY);
     mMinute= c.get(Calendar.MINUTE);

     if (mHour>=8&&mHour<10){
         nowtime="现在是一二节课";
     }else
     if (mHour>=10&&mHour<12){
         nowtime="现在是三四节课";
     }else
     if ((mHour==13&&mMinute>=30)||(mHour==14)||(mHour==15&&mMinute<30)){
         nowtime="现在是五六节课";
     }else
     if ((mHour==15&&mMinute>=30)||(mHour==16)||(mHour==17&&mMinute<30)){
         nowtime="现在是七八节课";
     }else
     if ((mHour==17&&mMinute>=30)||(mHour==18&&mMinute<30)){
         nowtime="现在是第九节课";
     }else
     if ((mHour==18&&mMinute>=30)||(mHour==19)||(mHour==20&&mMinute<30)){
         nowtime="现在是十、十一节课";
     }else
    nowtime="现在是休息时间";

     if("1".equals(mWay)){
         mWay ="天";
         daycount=6;
     }else if("2".equals(mWay)){
         mWay ="一";
         daycount=0;
     }else if("3".equals(mWay)){
         mWay ="二";
         daycount=1;
     }else if("4".equals(mWay)){
         mWay ="三";
         daycount=2;
     }else if("5".equals(mWay)){
         mWay ="四";
         daycount=3;
     }else if("6".equals(mWay)){
         mWay ="五";
         daycount=4;
     }else if("7".equals(mWay)){
         mWay ="六";
         daycount=5;
     }
     Timestring=mYear + "年" + mMonth + "月" + mDay+"日"+"星期"+mWay;

     FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
     fab.setOnClickListener(new View.OnClickListener() {
         @Override
         public void onClick(View view) {
             Snackbar.make(view, "今天是"+Timestring+"\n"+nowtime+"  "+interesting[daycount], Snackbar.LENGTH_SHORT)
                     .setAction("Action", null).show();
         }
     });
```

# What I learned on GitHub

In addition to trying to use other GitHub libraries, I learned a lot, including color palettes, shake modules, fir update modules, sliding card interfaces, etc.
Some GitHub repository links are here

- 滑动卡片界面：[Android-SwipeToDismiss](https://github.com/romannurik/Android-SwipeToDismiss)
- fir更新模块:[UpdateDemo](https://github.com/hugeterry/UpdateDemo)

Some of them are directly written in the code, and I forgot the original address....

- Sensor call for shake
  
  ```Java
  public class ShakeService extends Service {
  public static final String TAG = "ShakeService";
  private SensorManager mSensorManager;
  public boolean flag=false;
  private ShakeBinder shakebinder= new ShakeBinder();
  private String htmlbody="";
  
  @Override
  public void onCreate() {
     // TODO Auto-generated method stub
     super.onCreate();
     mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
     Log.i(TAG,"Shake Service Create");
  }
  
  @Override
  public void onDestroy() {
     // TODO Auto-generated method stub
     flag=false;
     super.onDestroy();
     mSensorManager.unregisterListener(mShakeListener);
  }
  
  @Override
  public void onStart(Intent intent, int startId) {
     // TODO Auto-generated method stub
     super.onStart(intent, startId);
     Log.i(TAG,"Shake Service Start");
  }
  
  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
     // TODO Auto-generated method stub
     mSensorManager.registerListener(mShakeListener,
             mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
             //SensorManager.SENSOR_DELAY_GAME,
             50 * 1000); //batch every 50 milliseconds
     htmlbody=intent.getStringExtra("htmlbody");
  
     return super.onStartCommand(intent, flags, startId);
  }
  
  private final SensorEventListener mShakeListener = new SensorEventListener() {
     private static final float SENSITIVITY = 10;
     private static final int BUFFER = 5;
     private float[] gravity = new float[3];
     private float average = 0;
     private int fill = 0;
  
     @Override
     public void onAccuracyChanged(Sensor sensor, int acc) {
     }
  
     public void onSensorChanged(SensorEvent event) {
         final float alpha = 0.8F;
  
         for (int i = 0; i < 3; i++) {
             gravity[i] = alpha * gravity[i] + (1 - alpha) * event.values[i];
         }
  
         float x = event.values[0] - gravity[0];
         float y = event.values[1] - gravity[1];
         float z = event.values[2] - gravity[2];
  
         if (fill <= BUFFER) {
             average += Math.abs(x) + Math.abs(y) + Math.abs(z);
             fill++;
         } else {
             Log.i(TAG, "average:"+average);
             Log.i(TAG, "average / BUFFER:"+(average / BUFFER));
             if (average / BUFFER >= SENSITIVITY) {
                 handleShakeAction();//如果达到阈值则处理摇一摇响应
             }
             average = 0;
             fill = 0;
         }
     }
  };
  
  protected void handleShakeAction() {
     // TODO Auto-generated method stub
     flag=true;
     Toast.makeText(getApplicationContext(), "摇一摇成功", Toast.LENGTH_SHORT).show();
     Intent intent= new Intent();
     intent.putExtra("htmlbody",htmlbody);
     intent.addFlags(FLAG_ACTIVITY_NEW_TASK);
     intent.setClassName(this,"thinkwee.buptroom.ShakeTestActivity");
     startActivity(intent);
  }
  
  @Override
  public IBinder onBind(Intent intent) {
     // TODO Auto-generated method stub
     return shakebinder;
  }
  class ShakeBinder extends Binder{
  
  }
  }
  ```

```
# Independent network pull and multi-threading
-     In the previous structure, network pull was integrated in the welcome activity, in order to add a refresh function in the main interface and call the network pull at any time, I wrote the network pull as a separate class, which can be called when needed
-     However, in the welcome activity, the welcome animation and network pull are in two separate threads (to prevent the animation from being blocked), so there may be a situation where the welcome animation is completed and the main interface is entered, but the network pull is not completed, and the content pulled cannot be passed to the main interface. The final solution is to set a 2s timeout for the network pull. If it is not pulled, an incorrect parameter is passed to the activity that starts the main interface, prompting a refresh
```Java
        webget = new Webget();
        webget.init(webView);
        HaveNetFlag = webget.WebInit();

        new Handler().postDelayed(new Runnable() {
            public void run() {
                //execute the task
                ImageView img = (ImageView) findViewById(R.id.welcomeimg);
                Animation animation = AnimationUtils.loadAnimation(WelcomeActivity.this, R.anim.enlarge);
                animation.setFillAfter(true);
                img.startAnimation(animation);
            }
        }, 50);

        new Handler().postDelayed(new Runnable() {
            public void run() {
                //execute the task
                WrongNet = webget.getWrongnet();
                HaveNetFlag = webget.getHaveNetFlag();
                htmlbody = webget.getHtmlbody();
                Log.i("welcome", "2HaveNetFlag: " + HaveNetFlag);
                Log.i("welcome", "2Wrongnet: " + WrongNet);
                Log.i("welcome", "2html: " + htmlbody);
            }
        }, 2000);

        new Handler().postDelayed(new Runnable() {

            @Override
            public void run() {
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                intent.putExtra("WrongNet", WrongNet);
                intent.putExtra("HtmlBody", htmlbody);
                startActivity(intent);
                WelcomeActivity.this.finish();

            }

        }, 2500);
    }
```


{% endlang_content %}

{% lang_content zh %}

![i0IHL4.png](https://s1.ax1x.com/2018/10/20/i0IHL4.png)

# 学习的内容

- Android基本架构，组件，生命周期
- Fragment的使用
- Java库与库之间的调用
- Github的使用
- 部署app
- 图像处理的一些方法
- 一个愚蠢的拉取网页内容的方式
- GitHub第三方库的利用
- 颜色方面的知识
- Android Material Design
- 简单的优化
- 多线程与Handler

# 解决的问题

主要解决了这么几个问题

- Android6.0以上的版本貌似权限需要动态验证，现在写的只支持5.0及以下版本，用到的permisson:

```Java
    <uses-permission android:name="android.permission.INTERNET"></uses-permission>
    <uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW"></uses-permission>
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"></uses-permission>
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"></uses-permission>
    <uses-permission android:name="android.permission.MOUNT_UNMOUNT_FILESYSTEMS"></uses-permission>
```

- 网页是jsp动态网页，不能简单地parse，最后采用在webview中loadurl，执行javascript命令，需下载jsoup-1.9.2.jar这个包添加到库文件中

```Java
    final class MyWebViewClient extends WebViewClient {
        public boolean shouldOverrideUrlLoading(WebView view, String url) {
            view.loadUrl(url);
            return true;
        }
        public void onPageStarted(WebView view, String url, Bitmap favicon) {
            Log.d("WebView","onPageStarted");
            super.onPageStarted(view, url, favicon);
        }
        public void onPageFinished(WebView view, String url) {
            Log.d("WebView","onPageFinished ");
            view.loadUrl("javascript:window.handler.getContent(document.body.innerHTML);");
            super.onPageFinished(view, url);
        }
    }
```

- 写一个handler响应javascript命令,这样在content中就拿到String形式的html文件中body内容

```Java
    final  class JavascriptHandler{
        @JavascriptInterface
        public void getContent(String htmlContent){
            Log.i(Tag,"html content: "+htmlContent);
            document= Jsoup.parse(htmlContent);
            htmlstring=htmlContent;
            content=document.getElementsByTag("body").text();
            Toast.makeText(MainActivity.this,"加载完成",Toast.LENGTH_SHORT).show();
        }
    }
```

- 之后是字符串处理，根据教务处给的格式精简分类

```Java
    去逗号
    String contenttemp=content;
    content="";
    String[] contentstemp=contenttemp.split(",");
    for (String temp:contentstemp){
        content=content+temp;
    }

    分组
    contents=content.split(" |:");
    String showcontent="";
    count=0;
    int tsgflag=0;
    int cishu=0;
    j12.clear();
    j34.clear();
    j56.clear();
    j78.clear();
    j9.clear();
    j1011.clear();
    if (keyword.contains("图书馆")) tsgflag=1;
    for (String temp:contents){
        if (temp.contains(keyword)){
            cishu++;
            SaveBuidlingInfo(count,cishu,tsgflag);
        }
        count++;
    }

    SaveBuildingInfo是按教学楼分类存取一天教室，其中再按时间段分类存到j12,j34.....
    while (1 == 1) {
        if (contents[k].contains("楼") || contents[k].contains("节") || contents[k].contains("图"))
            break;
        ;
        switch (c) {
            case 1:
                j12.add(contents[k]);
                break;
            case 2:
                j34.add(contents[k]);
                break;
            case 3:
                j56.add(contents[k]);
                break;
            case 4:
                j78.add(contents[k]);
                break;
            case 5:
                j9.add(contents[k]);
                break;
            case 6:
                j1011.add(contents[k]);
                break;
            default:
                break;
        }
        k++;
    }
```

- 界面上套了一个NavigationView，没有什么特别设计的，因为没有设置多界面，就靠刷新TextView来伪装多个界面

- 尝试了MaterialDesign组件，加入一点系统时间方面的东西

```Java
    final Calendar c = Calendar.getInstance();
     c.setTimeZone(TimeZone.getTimeZone("GMT+8:00"));
     mYear = String.valueOf(c.get(Calendar.YEAR)); // 获取当前年份
     mMonth = String.valueOf(c.get(Calendar.MONTH) + 1);// 获取当前月份
     mDay = String.valueOf(c.get(Calendar.DAY_OF_MONTH));// 获取当前月份的日期号码
     mWay = String.valueOf(c.get(Calendar.DAY_OF_WEEK));
     mHour= c.get(Calendar.HOUR_OF_DAY);
     mMinute= c.get(Calendar.MINUTE);

     if (mHour>=8&&mHour<10){
         nowtime="现在是一二节课";
     }else
     if (mHour>=10&&mHour<12){
         nowtime="现在是三四节课";
     }else
     if ((mHour==13&&mMinute>=30)||(mHour==14)||(mHour==15&&mMinute<30)){
         nowtime="现在是五六节课";
     }else
     if ((mHour==15&&mMinute>=30)||(mHour==16)||(mHour==17&&mMinute<30)){
         nowtime="现在是七八节课";
     }else
     if ((mHour==17&&mMinute>=30)||(mHour==18&&mMinute<30)){
         nowtime="现在是第九节课";
     }else
     if ((mHour==18&&mMinute>=30)||(mHour==19)||(mHour==20&&mMinute<30)){
         nowtime="现在是十、十一节课";
     }else
    nowtime="现在是休息时间";

     if("1".equals(mWay)){
         mWay ="天";
         daycount=6;
     }else if("2".equals(mWay)){
         mWay ="一";
         daycount=0;
     }else if("3".equals(mWay)){
         mWay ="二";
         daycount=1;
     }else if("4".equals(mWay)){
         mWay ="三";
         daycount=2;
     }else if("5".equals(mWay)){
         mWay ="四";
         daycount=3;
     }else if("6".equals(mWay)){
         mWay ="五";
         daycount=4;
     }else if("7".equals(mWay)){
         mWay ="六";
         daycount=5;
     }
     Timestring=mYear + "年" + mMonth + "月" + mDay+"日"+"星期"+mWay;

     FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
     fab.setOnClickListener(new View.OnClickListener() {
         @Override
         public void onClick(View view) {
             Snackbar.make(view, "今天是"+Timestring+"\n"+nowtime+"  "+interesting[daycount], Snackbar.LENGTH_SHORT)
                     .setAction("Action", null).show();
         }
     });
```

# 在GitHub上学到的

此外还尝试引用了其他的一些GitHub库，学习了许多，包括调色盘，摇一摇模块，fir更新模块，滑动卡片界面等等
部分GitHub repository链接在这里

- 滑动卡片界面：[Android-SwipeToDismiss](https://github.com/romannurik/Android-SwipeToDismiss)
- fir更新模块:[UpdateDemo](https://github.com/hugeterry/UpdateDemo)

还有一些直接写在代码里了，忘记原地址了....

- 摇一摇的传感器调用
  
  ```Java
  public class ShakeService extends Service {
  public static final String TAG = "ShakeService";
  private SensorManager mSensorManager;
  public boolean flag=false;
  private ShakeBinder shakebinder= new ShakeBinder();
  private String htmlbody="";
  
  @Override
  public void onCreate() {
     // TODO Auto-generated method stub
     super.onCreate();
     mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
     Log.i(TAG,"Shake Service Create");
  }
  
  @Override
  public void onDestroy() {
     // TODO Auto-generated method stub
     flag=false;
     super.onDestroy();
     mSensorManager.unregisterListener(mShakeListener);
  }
  
  @Override
  public void onStart(Intent intent, int startId) {
     // TODO Auto-generated method stub
     super.onStart(intent, startId);
     Log.i(TAG,"Shake Service Start");
  }
  
  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {
     // TODO Auto-generated method stub
     mSensorManager.registerListener(mShakeListener,
             mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
             //SensorManager.SENSOR_DELAY_GAME,
             50 * 1000); //batch every 50 milliseconds
     htmlbody=intent.getStringExtra("htmlbody");
  
     return super.onStartCommand(intent, flags, startId);
  }
  
  private final SensorEventListener mShakeListener = new SensorEventListener() {
     private static final float SENSITIVITY = 10;
     private static final int BUFFER = 5;
     private float[] gravity = new float[3];
     private float average = 0;
     private int fill = 0;
  
     @Override
     public void onAccuracyChanged(Sensor sensor, int acc) {
     }
  
     public void onSensorChanged(SensorEvent event) {
         final float alpha = 0.8F;
  
         for (int i = 0; i < 3; i++) {
             gravity[i] = alpha * gravity[i] + (1 - alpha) * event.values[i];
         }
  
         float x = event.values[0] - gravity[0];
         float y = event.values[1] - gravity[1];
         float z = event.values[2] - gravity[2];
  
         if (fill <= BUFFER) {
             average += Math.abs(x) + Math.abs(y) + Math.abs(z);
             fill++;
         } else {
             Log.i(TAG, "average:"+average);
             Log.i(TAG, "average / BUFFER:"+(average / BUFFER));
             if (average / BUFFER >= SENSITIVITY) {
                 handleShakeAction();//如果达到阈值则处理摇一摇响应
             }
             average = 0;
             fill = 0;
         }
     }
  };
  
  protected void handleShakeAction() {
     // TODO Auto-generated method stub
     flag=true;
     Toast.makeText(getApplicationContext(), "摇一摇成功", Toast.LENGTH_SHORT).show();
     Intent intent= new Intent();
     intent.putExtra("htmlbody",htmlbody);
     intent.addFlags(FLAG_ACTIVITY_NEW_TASK);
     intent.setClassName(this,"thinkwee.buptroom.ShakeTestActivity");
     startActivity(intent);
  }
  
  @Override
  public IBinder onBind(Intent intent) {
     // TODO Auto-generated method stub
     return shakebinder;
  }
  class ShakeBinder extends Binder{
  
  }
  }
  ```

```
# 独立网络拉取，并使用多线程
-    在之前的结构中网络拉取整合在欢迎界面的activity中，为了在主界面中添加刷新功能，随时调用网络拉取，我把网络拉取单独写成了一个类，需要的时候调用
-    然而在欢迎界面中显示欢迎动画和网络拉取在两个独立的线程中（为了使得动画不卡顿），这样就出现了可能欢迎动画做完了进入主界面时网络拉取还没有完成，传不了拉取的内容到主界面，最后的解决方案是设置网络拉取2s超时，若没拉取到则传一个错误的参数到启动主界面的activity中，提示刷新
```Java
        webget = new Webget();
        webget.init(webView);
        HaveNetFlag = webget.WebInit();

        new Handler().postDelayed(new Runnable() {
            public void run() {
                //execute the task
                ImageView img = (ImageView) findViewById(R.id.welcomeimg);
                Animation animation = AnimationUtils.loadAnimation(WelcomeActivity.this, R.anim.enlarge);
                animation.setFillAfter(true);
                img.startAnimation(animation);
            }
        }, 50);

        new Handler().postDelayed(new Runnable() {
            public void run() {
                //execute the task
                WrongNet = webget.getWrongnet();
                HaveNetFlag = webget.getHaveNetFlag();
                htmlbody = webget.getHtmlbody();
                Log.i("welcome", "2HaveNetFlag: " + HaveNetFlag);
                Log.i("welcome", "2Wrongnet: " + WrongNet);
                Log.i("welcome", "2html: " + htmlbody);
            }
        }, 2000);

        new Handler().postDelayed(new Runnable() {

            @Override
            public void run() {
                Intent intent = new Intent(WelcomeActivity.this, MainActivity.class);
                intent.putExtra("WrongNet", WrongNet);
                intent.putExtra("HtmlBody", htmlbody);
                startActivity(intent);
                WelcomeActivity.this.finish();

            }

        }, 2500);
    }
```

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
