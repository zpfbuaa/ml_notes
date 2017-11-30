#encoding=utf-8
import numpy as np
import operator
import matplotlib.pyplot as plt
import math
def createDataSet():
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

group,labels = createDataSet()
print(group)
print(labels)

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

print (classify0([0,0],group,labels,3))

def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    numberOfLines = len(lines)
    kaseList = np.zeros((numberOfLines,3))
    labelsList = []
    idx = 0
    for line in lines:
        line = line.strip()
        split_line = line.split('\t')
        kaseList[idx,:] = split_line[0:3]
        labelsList.append(int(split_line[-1]))
        idx+=1
    return kaseList, labelsList

def modifydata(filename):
    f = open(filename,'r')
    lines = f.readlines()
    f.close()
    idx = 0
    numberOfLines = len(lines)
    for idx in range(numberOfLines):
        line = lines[idx]
        if('largeDoses' in line):
            lines[idx] = line.replace('largeDoses','3')
        elif('smallDoses' in line):
            lines[idx] = line.replace('smallDoses','2')
        elif('didntLike' in line):
            lines[idx] = line.replace('didntLike','1')
    f_w = open(filename, 'w')
    f_w.writelines(lines)
    f_w.close()

# no need to change the data we have datingTestSet2
#modifydata('../data/datingTestSet.txt')

datingDateMat, datingLabels = file2matrix('../data/datingTestSet2.txt')
print(datingDateMat[0:10])
print(datingLabels[0:10])

donlike=[]
littlelike=[]
mostlike=[]
for i in range(len(datingLabels)):
    if(datingLabels[i]==1):
        donlike.append(datingDateMat[i])
    elif(datingLabels[i]==2):
        littlelike.append(datingDateMat[i])
    elif(datingLabels[i]==3):
        mostlike.append(datingDateMat[i])

plt.subplot(3, 2, 1)
plt.scatter(datingDateMat[:,0],datingDateMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.subplot(3, 2, 2)
plt.scatter(datingDateMat[:,0],datingDateMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.subplot(3, 2, 3)
plt.scatter(datingDateMat[:,1],datingDateMat[:,0],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.subplot(3, 2, 4)
plt.scatter(datingDateMat[:,1],datingDateMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.subplot(3, 2, 5)
plt.scatter(datingDateMat[:,2],datingDateMat[:,0],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.subplot(3, 2, 6)
plt.scatter(datingDateMat[:,2],datingDateMat[:,1],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.savefig('dating_result.jpg')
plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    rangeVals = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    column_num = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(column_num,1))
    normDataSet = normDataSet/np.tile(rangeVals,(column_num,1))
    return normDataSet, rangeVals, minVals

normSet, ranges, minvals = autoNorm(datingDateMat)
print normSet
print ranges
print minvals

def datingClassTest(filename='../data/datingTestSet2.txt'):
    test_ratio=0.1
    dataSet, dataLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(dataSet)
    column_num = dataSet.shape[0]
    numOfTest = int(test_ratio*column_num)
    errorCount = 0.0
    for i in range(numOfTest):
        result = classify0(normMat[i,:],normMat[numOfTest:column_num,:],dataLabels[numOfTest:column_num],3)
        if(result!=dataLabels[i]):
            errorCount+=1
            print('actual_label is %d, predict_label is %d'%(dataLabels[i],result))
    print('total error number:%f and error_ratio:%f'%(errorCount,errorCount/numOfTest))

datingClassTest()

def datingPerson(filename='../data/datingTestSet2.txt'):
    result_list=['not at all', 'in small doses', 'in large doses']
    fly_miles = float(raw_input("请输入每年获得的飞行常客里程数:"))
    game_ratio = float(raw_input("请输入完游戏视频所耗时间比例:"))
    cream_consumed = float(raw_input("请输入每周消费的冰激凌公升数:"))
    datingDateMat,datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDateMat)
    test_array = ([fly_miles,game_ratio,cream_consumed])
    result = classify0((test_array-minvals)/ranges,normMat,datingLabels,3)
    print("This person in your eyes: ",result_list[result])

datingPerson()