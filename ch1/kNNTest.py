'''
Created on 2018年5月4日

@author: luomingqiang
'''
import ch1.kNN
import matplotlib.pyplot as plt
from numpy import array

group, labels = ch1.kNN.createDataSet();
print(group, '\n', labels)

clazz = ch1.kNN.classify0([1, 1], group, labels, 3)
print(clazz)

datingDataMat, datingLabels = ch1.kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat, '\n', datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

normMat, ranges, minValue = ch1.kNN.autoNorm(datingDataMat)
print(normMat, '\n', ranges, '\n', minValue)

# ch1ch2tingClassTest()

# ch1ch2assifyPerson()

testImgVec = ch1.kNN.img2vector('testDigits/0_13.txt')
print(testImgVec[0, 0 : 31], '\n', testImgVec[0, 32 : 63])

ch1.kNN.handWritingClassTest()