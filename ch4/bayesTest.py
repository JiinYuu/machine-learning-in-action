'''
Created on 2018年5月13日

@author: luomingqiang
'''
import ch4.bayes as bayes
import feedparser as feedparser

# listOPosts, listClasses = bayes.loadDataSet()
# myVocabList = bayes.createVocabList(listOPosts)
# print(myVocabList)
# print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
# print(bayes.setOfWords2Vec(myVocabList, listOPosts[1]))
# 
# trainMatrix = []
# for postinDoc in listOPosts :
#     trainMatrix.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
# p0V, p1V, pAb = bayes.trainNB0(trainMatrix, listClasses)
# print(p0V, '\n', p1V, '\n', pAb)
# 
# bayes.testingNB()
# 
# bayes.spamTest()
# 
# ny = feedparser.parse('https://www.zhihu.com/rss')
# print(ny['entries'])
# print(len(ny['entries']))
sports = feedparser.parse('http://www.people.com.cn/rss/sports.xml')
finance = feedparser.parse('http://www.people.com.cn/rss/finance.xml')
# vocabLst, pInner, pOuter = bayes.localWords(inner, outer)
# vocabLst, pInner, pOuter = bayes.localWords(inner, outer)
bayes.getTopWords(sports, finance, 'sports', 'finance')