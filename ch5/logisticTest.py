'''
Created on 2018年5月19日

@author: luomingqiang
'''
import ch5.logistic as logistic
import numpy as np

dataArr, labelMat = logistic.loadDataSet()
weights = logistic.gradAscent(dataArr, labelMat)
print(weights)

m, n = np.shape(np.mat(dataArr));

errorCount = 0.0;
for i in range(m) :
    inX = dataArr[i]
    label = logistic.classifyVector(inX, weights)
    if label != int(labelMat[i]) :
        errorCount += 1.0
print(errorCount)
# weights = logistic.stocGradAscent(dataArr, labelMat)
# print(weights)
# logistic.plotBestFit(weights)
# 
# weights = logistic.stocGradAscent1(dataArr, labelMat)
# print(weights)
# logistic.plotBestFit(weights)
# 
# logistic.multiTest()