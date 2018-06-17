'''
Created on 2018年5月13日

@author: luomingqiang
'''
# -*- coding: utf-8 -*-
from numpy import ones, log, array, random
import thulac

thu1 = thulac.thulac(seg_only = True)

def loadDataSet() :
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet) :
    vocabSet = set([])
    for document in dataSet :
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] = 1
        else :
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet) :
    returnVec = [0] * len(vocabList)
    for word in inputSet :
        if word in vocabList :
            returnVec[vocabList.index(word)] += 1
        else :
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory) :
    numTrainDocs = len(trainMatrix) # 总样本数
    numWords = len(trainMatrix[0]) # 词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # p（1），因为是二分类问题，所以可以直接sum / 总数
    p0Num = ones(numWords); p1Num = ones(numWords) # 类别为0或1的样本各词条出现的次数向量，为了避免*0，所以初始数量为1
    p0Denom = 2.0; p1Denom = 2.0 # 类别为0的样本各词条出现总次数，为了避免 /0，所以初始化处理为2
    for i in range(numTrainDocs) :
        if trainCategory[i] == 1 :
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else :
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom) # 类别为1的样本各词条出现的条件概率：p(w|c1)，由于概率乘积很小，所以取log2
    p0Vect = log(p0Num / p0Denom) # 类别为0的样本各词条出现的条件概率：p(w|c0)，由于概率乘积很小，所以取log2
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1) :
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) # 待分类样本属于p1的概率，由于采用了log，所以*变+
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) # 待分类样本属于p0的概率，由于采用了log，所以*变+
    return 1 if (p1 > p0) else 0

def testingNB() :
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMatrix = []
    for postinDoc in listOPosts :
        trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))
    p0Vec, p1Vec, pAb = trainNB0(array(trainMatrix), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0Vec, p1Vec, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0Vec, p1Vec, pAb))

def textParse(bigString) :
    listOfTokens = thu1.cut(bigString, text = True).split()
    return [tok.lower() for tok in listOfTokens if len(tok) > 1]

def spamTest() :
    docList = []; classList = []; fullText = []
    for i in range(1, 26) :
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10) :
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMatrix = []; trainClass = []
    for docIndex in trainingSet :
        trainMatrix.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0Vec, p1Vec, pSpam = trainNB0(array(trainMatrix), array(trainClass))
    errorCount = 0
    for docIndex in testSet :
        word2Vec = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(word2Vec, p0Vec, p1Vec, pSpam) != classList[docIndex] :
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount / len(testSet)))
    
def calcMostFreq(vocabList, fullText) :
    import operator
    freqDict = {}
    for token in vocabList :
        freqDict[token] = fullText.count(token)
        sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse = True)
    return sortedFreq[:30]

def localWords(feed1, feed0) :
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen) :
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
#     top30Words = calcMostFreq(vocabList, fullText)
#     for pairW in top30Words :
#         if pairW[0] in vocabList :
#             vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen)); testSet = []
    for i in range(20) :
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMatrix = []; trainClass = []
    for docIndex in trainingSet :
        trainMatrix.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0Vec, p1Vec, pSpam = trainNB0(array(trainMatrix), array(trainClass))
    errorCount = 0
    for docIndex in testSet :
        word2Vec = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(word2Vec, p0Vec, p1Vec, pSpam) != classList[docIndex] :
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount / len(testSet)))
    return vocabList, p0Vec, p1Vec

def getTopWords(inner, outer, innerTxt, outerTxt) :
    vocabList, pInner, pOuter = localWords(inner, outer)
    topInner = []; topOuter = []
    for i in range(len(pInner)) :
        if pInner[i] > -6.0 : topInner.append((vocabList[i], pInner[i]))
        if pOuter[i] > -6.0 : topOuter.append((vocabList[i], pOuter[i]))
    sortedInner = sorted(topInner, key = lambda pair : pair[1], reverse = True)
    print("-----------------------------", innerTxt, "------------------------------")
    for item in sortedInner :
        print(item)
    sortedOuter = sorted(topOuter, key = lambda pair : pair[1], reverse = True)
    print("-----------------------------", outerTxt, "------------------------------")
    for item in sortedOuter :
        print(item)
















































