'''
Created on 2018年5月15日

@author: luomingqiang
'''
# coding:utf-8
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plot
from numpy import *
import numpy as np
from sklearn.linear_model import LogisticRegression as logic
plot.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
'''
x = x + alpha * derivedFunction(x)
'''
def gradient(function, derivedFunction, ascent) :
    alpha = 0.01; presision = 0.0000000001
    yOld = 0.0; yNew = 1.0
    x = 1.0
    while abs(yNew - yOld) > presision :
        yOld = function(x)
        if ascent :
            x = x + alpha * derivedFunction(x)
        else :
            x = x - alpha * derivedFunction(x)
        yNew = function(x)
        print(yNew)
    return yNew

def loadDataSet() :
    dataMatrix = []; labelMatrix = []
    fr = open('testSet.txt')
    for line in fr.readlines() :
        dataArray = line.strip().split()
        dataMatrix.append([1.0, float(dataArray[0]), float(dataArray[1])])
        labelMatrix.append(int(dataArray[2]))
    fr.close()
    return dataMatrix, labelMatrix

def plotDataSet() :
    dataMatrix, labelMatrix = loadDataSet()
    dataArray = array(dataMatrix)
    n = shape(dataMatrix)[0]
    xcord1 = []; ycord1 = []; xcord2 = []; ycord2 = []
    for i in range(n) :
        if int(labelMatrix[i]) == 1 :
            xcord1.append(dataArray[i, 1])
            ycord1.append(dataArray[i, 2])
        else :
            xcord2.append(dataArray[i, 1])
            ycord2.append(dataArray[i, 2])
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = .5)
    plot.title('DataSet')
    plot.xlabel('x')
    plot.ylabel('y')
    plot.show()

def sigmoid(inX) :
    return 1.0 / (1 + exp(-inX))

def gradAscent2(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                        #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                            #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                        #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                                #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化                                        #存储每次更新的回归系数
    for j in range(numIter):                                           
        dataIndex = list(range(m))
        for i in range(m):           
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights 

def colicTest():
    frTrain = open('horseColicTraining.txt')                                        #打开训练集
    frTest = open('horseColicTest.txt')                                                #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)

def colicSklearnTest() :
    trainfr = open('horseColicTraining.txt')
    trainMat = []; trainLabels = []
    for line in trainfr.readlines() :
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1) :
            lineArr.append(float(curLine[i]))
        trainMat.append(lineArr)
        trainLabels.append(float(curLine[-1]))
    classfier = logic(solver = 'sag', max_iter = 5000).fit(trainMat, trainLabels)
    testFr = open('horseColicTest.txt')
    testMat = []; testLabels = []
    for line in testFr.readlines() :
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1) :
            lineArr.append(float(curLine[i]))
        testMat.append(lineArr)
        testLabels.append(float(curLine[-1]))
    rate = classfier.score(testMat, testLabels) * 100
    print('正确率为: %f%%' % rate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def gradAscent(dataMat, labels) :
    dataMat = mat(dataMat)
    labels = mat(labels).transpose()
    m, n = shape(dataMat)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n, 1))
    weightsArray = array([])
    for k in range(maxCycles) :
        h = sigmoid(dataMat * weights)
        error = labels - h
        weights = weights + alpha * dataMat.transpose() * error
        weightsArray = append(weightsArray, weights)
    weightsArray = weightsArray.reshape(maxCycles, n)
    return weights.getA(), weightsArray

def randGradAscent(dataMat, labels, times) :
    dataMat = array(dataMat)
    m, n = shape(dataMat)
    weights = ones(n)
    # weightsArray = array([])
    for i in range(times) :
        dataIndex = list(range(m))
        for j in range(m) :
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMat[randIndex] * weights))
            error = labels[randIndex] - h
            weights = weights + alpha * error * dataMat[randIndex]
            # weightsArray = append(weightsArray, weights, axis = 0)
            del(dataIndex[randIndex])
    # weightsArray = weightsArray.reshape(times * m,n)
    return weights

def randGradAscentTest() :
    trainfr = open('horseColicTraining.txt')
    
    trainMat = []; trainLabels = []
    for line in trainfr.readlines() :
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1) :
            lineArr.append(float(curLine[i]))
        trainMat.append(lineArr)
        trainLabels.append(float(curLine[-1]))
    weights = randGradAscent(trainMat, trainLabels, 500)
    errorCount = 0.0; numTestVec = 0.0
    testFr = open('horseColicTest.txt')
    for line in testFr.readlines() :
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(curLine) - 1) :
            lineArr.append(float(curLine[i]))
        z = sum(array(lineArr) * weights)
        prob = int(1.0 if sigmoid(z) > 0.5 else 0.0)
        if prob != int(curLine[-1]) :
            errorCount += 1.0
            print('classifier error: ', curLine)
    print('错误率为: %.2f%%' % (errorCount * 100.0 / numTestVec) )
    return

def plotBestFit(weights) :
    dataMatrix, labelMatrix = loadDataSet()
    dataArray = array(dataMatrix)
    n = shape(dataMatrix)[0]
    xcord1 = []; ycord1 = []; xcord2 = []; ycord2 = []
    for i in range(n) :
        if int(labelMatrix[i]) == 1 :
            xcord1.append(dataArray[i, 1])
            ycord1.append(dataArray[i, 2])
        else :
            xcord2.append(dataArray[i, 1])
            ycord2.append(dataArray[i, 2])
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's', alpha = .5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green', alpha = .5)
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plot.title('DataSet')
    plot.xlabel('x')
    plot.ylabel('y')
    plot.show()

def plotWeights(weights_array1, weights_array2):
    #设置汉字格式
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plot.subplots(nrows = 3, ncols = 2,sharex = False, sharey = False, figsize = (20, 10))
    x1 = arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
    plot.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plot.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
    plot.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1')
    plot.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plot.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
    plot.setp(axs0_title_text, size=20, weight='bold', color='black') 
    plot.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
    plot.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1')
    plot.setp(axs2_xlabel_text, size=20, weight='bold', color='black') 
    plot.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plot.show()






































