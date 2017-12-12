# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 下午3:36
# @Author  : 伊甸一点
# @FileName: bayes.py
# @Software: PyCharm
# @Blog    : http://zpfbuaa.github.io

import numpy as np
import re
import random
import feedparser

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec

listPosts, listClasses = loadDataSet()

myVocabList = createVocabList(listPosts)
print ('my vocablist is '+ str(myVocabList))

trainMat2 = [] # create a matrix
for postinDoc in listPosts:
    trainMat2.append(setOfWords2Vec(myVocabList,postinDoc))

print ('total documnets = '+str(len(trainMat2)))
for i in range(0,len(trainMat2)):
    print ('article is '+str(listPosts[i]))
    print (str(trainMat2[i])+'\n')

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom+= sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

p0V, p1V, pAb = trainNB0(trainMat2,listClasses)

print (p0V)
print (p1V)
print (pAb)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()


def BagofWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString): #TODO: parse word can be much more complex
    word_pt = re.compile('\\W*')
    listOfToken = word_pt.split(bigString)
    return [tok.lower() for tok in listOfToken if len(tok) > 2] # just take the word(length > 2)

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('data/email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('data/email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    print 'the vocabList length is',len(vocabList)
    trainingSet = range(50)
    testSet = []
    for i in range(10): # choose 10 document as testSet
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) # remove the data in trainingSet
    traingMat = []
    trainClasses = []
    for docIndex in trainingSet:
        traingMat.append(BagofWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(traingMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = BagofWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1
    print ('the error rate is:',float(errorCount)/len(testSet))

spamTest()

ny = feedparser.parse('https://newyork.craigslist.org/search/m4m?format=rss')
print len(ny['entries'])