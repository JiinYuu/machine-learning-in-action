'''
Created on 2018年5月11日

@author: luomingqiang
'''
import ch3.trees as trees
myDat, labels = trees.createDataSet()
print(myDat)
print(trees.calcShannonEnt(myDat))
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))
print(trees.splitDataSet(myDat, 1, 1))
print(trees.chooseBestFeatureToSplit(myDat))
myTree = trees.createTree(myDat, labels)
print(myTree)
trees.storeTree(myTree, 'classifier.txt')
myTree = trees.grabTree('classifier.txt')
print(myTree)