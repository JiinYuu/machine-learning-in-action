'''
Created on 2018年6月2日

@author: luomingqiang
'''
import ch7.AdaBoost as ada
# import numpy as np

# D = np.mat(np.ones((5, 1)) / 5)
# dataMat, classLabels = ada.loadSimpData()
# print(dataMat, '\n', classLabels)
# bestStump, minError, bestClassEst = ada.buildStump(dataMat, classLabels, D)
# print(bestStump, '\n', minError, '\n', bestClassEst)
# 
# classifierArray = ada.adaBoostTrainDS(dataMat, classLabels, 9)
# print(classifierArray)
# 
# print(ada.adaClassify([0, 0], classifierArray))
# print(ada.adaClassify([[5, 5], [0, 0]], classifierArray))

# dataArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
# print(dataArr, '\n', labelArr)
# classifierArr = ada.adaBoostTrainDS(dataArr, labelArr, 50)
# testData, testLabel = ada.loadDataSet('horseColicTest2.txt')
# prediction10 = ada.adaClassify(testData, classifierArr)
# print(prediction10)
# errArr = np.mat(np.ones((67, 1)))
# print(classifierArr)
# print(errArr[prediction10 != np.mat(testLabel).T].sum())
dataArr, labelArr = ada.loadDataSet('horseColicTraining2.txt')
classifierArr, aggClassEst = ada.adaBoostTrainDS(dataArr, labelArr, 50)
ada.plotROC(aggClassEst.T, labelArr)