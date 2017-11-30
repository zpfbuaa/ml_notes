from math import log
import operator
def calShannonEntropy(dataSet):
    numberOfData = len(dataSet)
    labelDic = {}
    for item_data in dataSet:
        # take the last value as label
        label = item_data[-1]
        if label not in labelDic.keys():
            labelDic[label]=0
        labelDic[label]+=1
    shannon = 0.0
    for key in labelDic:
        prob = float(labelDic[key])/numberOfData
        shannon -= prob*log(prob,2)
    return shannon

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'yes']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

myMat, labels = createDataSet()

print 'calShannon:',calShannonEntropy(myMat)

# split_dataSet
def splitDataSet(dataSet, axis, value):
    splitDataSet = []
    for item_data in dataSet:
        if item_data[axis] == value :
            splitVector = item_data[:axis]
            splitVector.extend(item_data[axis+1:])
            splitDataSet.append(splitVector)
    return splitDataSet

print 'split_result1:', splitDataSet(myMat,0,1)
print 'split_result2:',splitDataSet(myMat,0,0)

def chooseBestFeature(dataSet):
    # feature numbers, the last one is label, just choose the first_data as the sample
    numberFeatures = len(dataSet[0])-1
    # calculation the ShannonEntropy as the base line
    baseEntorpy = calShannonEntropy(dataSet)

    #define the best result
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numberFeatures):
        # find all the feature
        featureList = [item[i] for item in dataSet]
        # remove dumplings
        uniqueVals = set(featureList)
        newEntropy = 0.0
        # calculate entropy_result for each feature_split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calShannonEntropy(subDataSet)
        infoGain = baseEntorpy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

myMat2, labels2 = createDataSet()
chooseBestFeature(myMat2)

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    # all the class
    classList = [item[-1] for item in dataSet]
    # first return case: all data are same label
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # can't divide the data complicity
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # find the bestFeature axis and label
    bestFeatureAxis = chooseBestFeature(dataSet)
    bestFeatureLabel = labels[bestFeatureAxis]
    # create tree by recursion
    mytree = {bestFeatureLabel:{}}
    # remove the feature which is used
    del(labels[bestFeatureAxis])
    # get the total values of this feature_axis
    featureValues = [feature[bestFeatureAxis] for feature in dataSet]
    uniqueFeatureVals = set(featureValues)
    #
    for values in uniqueFeatureVals:
        # Python: parameters are passed by reference, so get a new param
        subLabels = labels
        mytree[bestFeatureLabel][values] = createTree(splitDataSet(dataSet,bestFeatureAxis,values),subLabels)
    return mytree

myMat3, labels3 = createDataSet()
print 'createtree:',createTree(myMat3,labels3)


