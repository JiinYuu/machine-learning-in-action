'''
Created on 2018年5月25日

@author: luomingqiang
'''
from ch6 import SvmMLiA as svm
# import numpy as np

# dataArr, labelArr = svm.loadDataSet('testSet.txt')
# print(dataArr, '\n', labelArr)
 
# b, alphas = svm.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
 
# print(b, '\n', alphas[alphas > 0])

# b, alphas = svm.smoP(dataArr, labelArr, 0.6, 0.001, 40)
# print(b, "\n", alphas[alphas > 0])
# 
# ws = svm.calcWs(alphas, dataArr, labelArr)
# print(ws)
# 
# dataMat = np.mat(dataArr)
# wmat = np.mat(ws)
# print(dataMat[0] * wmat + b)
# print(dataMat[1] * wmat + b)
# print(dataMat[2] * wmat + b)
# print(dataMat[3] * wmat + b)
# svm.testRbf(0.1)
svm.testDigits(('rbf', 0.1))