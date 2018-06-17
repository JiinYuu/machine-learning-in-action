'''
Created on 2018年5月11日

@author: luomingqiang
'''
from math import log
import operator

def calcShannonEnt(dataSet) :
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet : # 循环每一行数据
        currentLabel = featVec[-1] # 取最后一个元素，即该条数据类别
        if currentLabel not in labelCounts.keys() :
            labelCounts[currentLabel] = 0 # 如果当前类别还没出现过，则初始化当前类别元素数量为0
        labelCounts[currentLabel] += 1 # 将当前类别元素数量+1
    shannonEnt = 0.0 # 信息熵
    for key in labelCounts : # 循环所有类别
        prob = float(labelCounts[key]) / numEntries # 当前类别的出现概率
        shannonEnt -= prob * log(prob, 2) # 总信息熵 = 总信息熵 + 当前类别信息熵，因为log（x < 1）的值为负数，所以这里用-=
    return shannonEnt

def createDataSet() :
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value) :
    retDataSet = []
    for featVec in dataSet :
        if featVec[axis] == value :
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet) :
    numFeatures = len(dataSet[0]) - 1 # 特征总数
    baseEntropy = calcShannonEnt(dataSet) # 未分类前的信息熵
    bestInfoGain = 0.0; bestFeatrue = -1
    for i in range(numFeatures) : # 循环所有特征
        featList = [example[i] for example in dataSet] # 取出当前特征的所有取值
        uniqueVals = set(featList) # 去重
        newEntropy = 0.0
        for value in uniqueVals : # 计算按当前特征划分数据集后的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain) : # 更新最佳分类特征和最佳信息增益
            bestInfoGain = infoGain
            bestFeatrue = i
    return bestFeatrue

def majorityCnt(classList) :
    classCount = {}
    for vote in classList :
        if vote not in classCount.keys() :
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount

def createTree(dataSet, labels) :
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList) :
        return classList[0]
    if len(dataSet[0]) == 1 :
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = { bestFeatLabel : {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals :
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec) :
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys() :
        if testVec[featIndex] == key :
            if type(secondDict[key]).__name__ == 'dict' :
                classLabel = classify(secondDict[key], featLabels, testVec)
            else :
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename) :
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()
    
def grabTree(filename) :
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)




























