'''
Created on 2018年5月11日

@author: luomingqiang
'''
import ch3.treePlotter as plotter
import ch3.trees as trees
# 
# myData, labels = trees.createDataSet()
# myTree = trees.createTree(myData, lables)
# plotter.createPlot(myTree)
# myTree = plotter.retrieveTree(0)
# print(myTree)
# print(plotter.getNumLeafs(myTree))
# print(plotter.getTreeDepth(myTree))
# print(trees.classify(myTree, labels, [1, 0]))
# print(trees.classify(myTree, labels, [1, 1]))
# plotter.createPlot(myTree)

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
print(lensesTree)
trees.storeTree(lensesTree, 'classifier.txt')
lensesTree = trees.grabTree('classifier.txt')
print(trees.classify(lensesTree, ['age', 'prescript', 'astigmatic', 'tearRate'], ['pre', 'myope', 'no', 'normal']))

plotter.createPlot(lensesTree)
