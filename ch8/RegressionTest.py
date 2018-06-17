'''
Created on 2018年6月9日

@author: luomingqiang
'''
import ch8.Regression as reg
import numpy as np
# import matplotlib.pyplot as plt

# xArr, yArr = reg.loadDataSet('ex0.txt')
# print(xArr, '\n', yArr)

# ws = reg.standRegres(xArr, yArr)
# print(ws)
# 
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# yHat = np.mat(xMat * ws)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# yHat = xMat * ws
# print(np.corrcoef(yHat.T, yMat))
# print(reg.lwlr(xArr[0], xArr, yArr, 0.01))
# print(reg.lwlr(xArr[0], xArr, yArr, 10.0))
# yHat = reg.lwlrTest(xArr, xArr, yArr, 0.01)
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd][:, 0, :]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[srtInd])
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s = 2, c = 'red')
# plt.show()
# abX, abY = reg.loadDataSet('abalone.txt')
# yHat01 = reg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = reg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = reg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
# error01 = reg.rssError(abY[0:99], yHat01.T)
# error1 = reg.rssError(abY[0:99], yHat1.T)
# error10 = reg.rssError(abY[0:99], yHat10.T)
# print(error01, error1, error10)
# yHat01 = reg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
# yHat1 = reg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
# yHat10 = reg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
# error01 = reg.rssError(abY[100:199], yHat01.T)
# error1 = reg.rssError(abY[100:199], yHat1.T)
# error10 = reg.rssError(abY[100:199], yHat10.T)
# print(error01, error1, error10)
# ws = reg.standRegres(abX[0:99], abY[0:99])
# yHat = np.mat(abX[100:199]) * ws
# errorStand = reg.rssError(abY[100:199], yHat.T.A)
# print(errorStand)
# abX, abY = reg.loadDataSet('abalone.txt')
# ws = reg.ridgeTest(abX, abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ws)
# plt.show()
xArr, yArr = reg.loadDataSet('abalone.txt')
reg.stageWise(xArr, yArr, 0.001, 5000)
xMat = np.mat(xArr)
yMat = np.mat(yArr)
xMat = reg.regularize(xMat)
yMean = np.mean(yMat, 0)
yMat = yMat - yMean
weights = reg.standRegres2(xMat, yMat.T)
print(weights.T)