---
title: 机器学习入门:Mnist(SVM,PCA)
categories: 机器学习
tags:
  - code
  - machinelearning
mathjax: true
date: 2017-03-16 10:35:04
photos: http://ojtdnrpmt.bkt.clouddn.com/blog/20170308/184051988.JPG	
---
在mnist手写数字识别集上测试特征降维前后用SVM进行识别的准确度和时间
使用主成分分析方法进行特征降维

<!--more-->

# Mnist
-	Mnist的训练集有60000条数据，每条数据是8*8点阵图像，代表一个手写数字，因此有64维，在线性SVM中训练需要很长时间，如果通过PCA降维可以在损失少量精确度的情况下大大缩短训练时间

# SVM

# PCA

# 代码
```Python
    from tools.data_util import DataUtils
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt
    import numpy as np
    
    
    def init():
        trainfile_X = r'E:\Machine Learning\MLData\mnist\train-images.idx3-ubyte'
        trainfile_y = r'E:\Machine Learning\MLData\mnist\train-labels.idx1-ubyte'
        testfile_X = r'E:\Machine Learning\MLData\mnist\t10k-images.idx3-ubyte'
        testfile_y = r'E:\Machine Learning\MLData\mnist\t10k-labels.idx1-ubyte'
        train_X = DataUtils(filename=trainfile_X).getImage()
        train_y = DataUtils(filename=trainfile_y).getLabel()
        test_X = DataUtils(testfile_X).getImage()
        test_y = DataUtils(testfile_y).getLabel()
        ss = StandardScaler()
        train_X = ss.fit_transform(train_X)
        test_X = ss.transform(test_X)
        return train_X, train_y, test_X, test_y
    
    
    def LSVC(train_X, train_y, test_X, test_y):
        lsvc = LinearSVC()
        lsvc.fit(train_X, train_y)
        predict_y = lsvc.predict(test_X)
    
        print(lsvc.score(test_X, test_y))
        print(classification_report(test_y, predict_y))
    
    
    def PrincipalComponentAnalysis(train_X, train_y, test_X, test_y):
        estimator = PCA(n_components=20)
    
        train_X_pca = estimator.fit_transform(train_X)
        test_X_pca = estimator.transform(test_X)
        lsvc = LinearSVC()
        lsvc.fit(train_X_pca, train_y)
        predict_y = lsvc.predict(test_X_pca)
    
        print(lsvc.score(test_X_pca, test_y))
        print(classification_report(test_y, predict_y))
    
    
    def main():
        train_X, train_y, test_X, test_y = init()
        print(train_X.shape[0])
        # LSVC(train_X, train_y, test_X, test_y)
        # PrincipalComponentAnalysis(train_X, train_y, test_X, test_y)
    
    
    if __name__ == "__main__":
        main()
```

# 结果
-	降维前
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170316/155543898.JPG)
-	降维后
	![mark](http://ojtdnrpmt.bkt.clouddn.com/blog/20170316/155614953.JPG)
-	原始64维的数据在我的机器上跑了将近20分钟，降到20维后3分钟就输出了结果，对比一下各项性能指标，下降了4%左右
	
