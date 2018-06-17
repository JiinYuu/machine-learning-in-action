'''
Created on 2018年5月15日

@author: luomingqiang
'''
import ch5.logRegres as logic

# function = lambda x : -(x * x) + 4 * x
# derivedFunction = lambda x : -2 * x + 4
# 
# maxY = logic.gradient(function, derivedFunction, True)
# print(maxY)
# 
# function = lambda x : (x * x) + 4 * x
# derivedFunction = lambda x : 2 * x + 4
# minY = logic.gradient(function, derivedFunction, False)
# print(minY)
# 
# dataMatrix, labels = logic.loadDataSet()
# print(dataMatrix)
# print(labels)
# logic.plotDataSet()
# 
# weights = logic.gradAscent(dataMatrix, labels)
# print(weights)
# logic.plotBestFit(weights)
# 
# weights = logic.gradAscent1(dataMatrix, labels, 200)
# print(weights)
# logic.plotBestFit(weights)
# dataMatrix, labels = logic.loadDataSet()
# weights1, weightsArray1 = logic.gradAscent(dataMatrix, labels)
# weights2, weightsArray2 = logic.randGradAscent(dataMatrix, labels, 200)
# logic.plotWeights(weightsArray1, weightsArray2)
logic.colicSklearnTest()