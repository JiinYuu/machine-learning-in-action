'''
Created on 2018年6月9日

@author: luomingqiang
'''
import numpy as np

def loadDataSet(fileName) :
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines() :
        lineArr = []
        curLine = line.split('\t')
        for i in range(numFeat) :
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr) :
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0 :
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def standRegres2(xMat, yMat) :
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0 :
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0) :
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m) :
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0 :
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0) :
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m) :
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHat) :
    return ((yArr - yHat) ** 2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2) :
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0 :
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr) :
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts) :
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100) :
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    n = np.shape(xMat)[1]
    ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    returnMat = np.ones((numIt, n))
    for i in range(numIt) :
        print(ws.T)
        minError = np.inf
        for j in range(n) :
            for s in [1, -1] :
                wsTest = ws.copy()
                wsTest[j] += s * eps
                yHat = xMat * wsTest
                error = rssError(yMat.A, yHat.A)
                if error < minError :
                    minError = error
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat