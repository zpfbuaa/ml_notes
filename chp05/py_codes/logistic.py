# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 下午6:09
# @Author  : 伊甸一点
# @FileName: logistic.py
# @Software: PyCharm
# @Blog    : http://zpfbuaa.github.io

import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(inZ):
    return 1.0/(1+np.exp(-inZ))

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('../data/testSet.txt')
    for line in fr.readlines():
        split_result = line.strip().split()
        dataMat.append([1.0, float(split_result[0]), float(split_result[1])])
        labelMat.append(int(split_result[2]))
    return dataMat,labelMat # return value type are [list, list]

"""
what is this mean?
"""

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()# convert row vector to column vector
    column, row = np.shape(dataMatrix)
    alpha = 0.001 # 步长
    maxCycles = 500 # 迭代次数
    weights = np.ones((row,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error # update w = w + alpha * x * h(wx)(y - h(wx))
    weights = np.array(weights)
    return weights

def plotBestFit(weights,name):
    #weights = weights.getA()
    #print type(weights)
    #weights = weights
    dataMat, labelMat = loadDataSet()
    dataArray = np.array(dataMat) # convert to array
    n = np.shape(dataMat)[0] # the input data dimension
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in range(int(n)):
        if int(labelMat[i])==1:
            x1.append(dataArray[i, 1]); y1.append(dataArray[i, 2])
        else:
            x2.append(dataArray[i, 1]); y2.append(dataArray[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1,s=30,c='red',marker='s')
    ax.scatter(x2, y2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]* x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Training Algorithm '+name)
    plt.savefig(name + '.jpg')
    plt.show()
    plt.close()


dataMat, labelMat = loadDataSet()
weights1 = gradAscent(dataMat,labelMat)
print ('weights1',weights1)
plotBestFit(weights1,'gradAscent')

def stocGradAscent0(dataMatrix, classLabels):
    column, row = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(row)
    for i in range(column):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]

    return weights

dataMat, labelMat = loadDataSet()
weights2 = stocGradAscent0(np.array(dataMat), labelMat)
print ('weights2',weights2)
plotBestFit(weights2,'stocGradAscent0')

def stocGradAscent1(dataMatrix, classLabels, numIter):
    column, row = np.shape(dataMatrix)
    weights = np.ones(row)
    for j in range(numIter):
        dataIndex = range(column)
        for i in range(column):
            alpha = 4/(1.0+j+i)+0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

weights3 = stocGradAscent1(np.array(dataMat), labelMat,150)
print ('weights3',weights3)
plotBestFit(weights3,'stocGradAscent1')

def classifyVector(inX, weights):
    prob = sigmoid(sum( inX * weights))
    if prob > 0.5 : return 1.0
    else : return 0.0

def colicTest():
    frTrain = open('../data/horseColicTraining.txt')
    frTest = open('../data/horseColicTest.txt')
    trainingtSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))

        trainingtSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingtSet), trainingLabels, 500) # get trainWeights and set training rounds = 500
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!= int(currLine[21]):
            errorCount+=1
    errorRate = (float(errorCount)/numTestVec)
    print ('the error rate of this test is: %f'%errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print 'after %d iteration the average error rate is: %f' %(numTests,errorSum/float(numTests))

multiTest()