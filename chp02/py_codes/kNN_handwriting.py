import numpy as np
import os
import operator


IMAGE_SIZE = 32
TRAINING_ROOT = '../data/trainingDigits/'
TEST_ROOT = '../data/testDigits/'

# kNN algorithm
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5
    sortedDistIndicies = distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# imgae is 32*32
def img2vector(filename):
    finalVector = np.zeros((1,1024))
    f = open(filename)
    for i in range(IMAGE_SIZE):
        line = f.readline()
        for j in range(IMAGE_SIZE):
            finalVector[0,i*IMAGE_SIZE+j] = int(line[j])
    return finalVector

vect = img2vector(TEST_ROOT+'0_13.txt')
print vect


def handwritingTest():
    DataSetLabel = []
    trainingFileList = os.listdir(TRAINING_ROOT)
    traingNumber = len(trainingFileList)
    trainingMat = np.zeros((traingNumber,IMAGE_SIZE*IMAGE_SIZE))
    for i in range(traingNumber):
        fileName = trainingFileList[i]
        label = int(fileName.split('.')[0].split('_')[0])
        DataSetLabel.append(label)
        trainingMat[i,:] = img2vector(TRAINING_ROOT+fileName)

    testFileList = os.listdir(TEST_ROOT)
    errorCount = 0.0
    testNumber = len(testFileList)
    for i in range(testNumber):
        fileName = testFileList[i]
        label = int(fileName.split('.')[0].split('_')[0])
        testVector = img2vector(TEST_ROOT+fileName)
        result = classify0(testVector, trainingMat, DataSetLabel, 3)
        if(result!=label):
            print('error file_name %s, actual_label is %d, predict_label is %d'%(fileName,label,result))
            errorCount+=1
    print('error_number is %f, error_ration is %f'%(errorCount,errorCount/testNumber))

handwritingTest()