'''
Created on 2018年5月19日

@author: luomingqiang
'''
import numpy as np
def loadDataSet() :
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines() :
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmode(inX) :
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels) :
    dataMatrix = np.mat(dataMatIn) # 将数据转成m * n矩阵
    labelMat = np.mat(classLabels).transpose() # 将类别转成 m * 1矩阵
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1)) # 初始化回归系数为 n * 1矩阵
    for k in range(maxCycles) :
        h = sigmode(dataMatrix * weights) # 按当前回归系数对数据进行分类，得到m * 1的结果矩阵
        error = labelMat - h # 得出误差，m * 1矩阵
        weights = weights + alpha * (dataMatrix.transpose() * error) # 根据误差调整回归系数
    return weights

def stocGradAscent(dataMatIn, classLabels) :
    dataMatrix = np.array(dataMatIn)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    alpha = 0.01
    maxCycles = 200
    for j in range(maxCycles) :
        for i in range(m) :
            h = sigmode(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * dataMatrix[i] * error
    return weights

def stocGradAscent1(dataMatIn, classLabels, maxCycles = 150) :
    dataMatrix = np.array(dataMatIn)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(maxCycles) :
        indexList = list(range(m))
        for i in range(m) :
            alpha = 20 / (1.0 + j + i) + 0.001
            randIndex = int(np.random.uniform(0, len(indexList)))
            dataIndex = indexList[randIndex]
            h = sigmode(sum(dataMatrix[dataIndex] * weights))
            error = classLabels[dataIndex] - h
            weights = weights + alpha * dataMatrix[dataIndex] * error
            del(indexList[randIndex])
    return weights

def classifyVector(inX, weights) :
    prob = sigmode(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def colicTest() :
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines() :
        lineArr = line.strip().split('\t')
        lineData = []
        for i in range(len(lineArr) - 1) :
            lineData.append(float(lineArr[i]))
        trainingSet.append(lineData)
        trainingLabels.append(float(lineArr[-1]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0.0; numTestVec = 0.0
    for line in frTest.readlines() :
        numTestVec += 1.0
        lineArr = line.strip().split('\t')
        lineData = []
        for i in range(len(lineArr) - 1) :
            lineData.append(float(lineArr[i]))
        if classifyVector(np.array(lineData), trainWeights) != int(lineArr[-1]) :
            errorCount += 1.0
    errorRate = float(errorCount / numTestVec)
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest() :
    sumErrorRate = 0.0
    for i in range(10) :
        sumErrorRate += colicTest()
    print('after %d iterations the average error rate is: %f' % (10, float(sumErrorRate / 10.0)))

def plotBestFit(weights) :
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n) :
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else :
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()