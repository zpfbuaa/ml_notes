from math import log
import operator
import matplotlib.pyplot as plt
import pickle

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
        subLabels = labels[:]
        mytree[bestFeatureLabel][values] = createTree(splitDataSet(dataSet,bestFeatureAxis,values),subLabels)
    return mytree

myMat3, labels3 = createDataSet()
print 'createtree:',createTree(myMat3,labels3)

# get numbers of tree leaf
def getTreeLeafs(myTree):
    numberOfLeafs = 0
    firstStr = myTree.keys()[0]
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__=='dict':
            numberOfLeafs += getTreeLeafs(secondDic[key])
        else:
            numberOfLeafs+=1
    return numberOfLeafs

# get tree depth
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDic[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth : maxDepth = thisDepth
    return maxDepth

def retriveTree(i):
    listOfTrees = [ {'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                    {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

mytree = retriveTree(0)
print 'leafs:',getTreeLeafs(mytree)
print 'depth:',getTreeDepth(mytree)




decisionNode = dict(boxstyle = "sawtooth",fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')
def plotNode(nodeTxt, centerPt, parentPt, nodeType):

    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                        va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(mytree, parentPt, nodeTxt):
    leafs = getTreeLeafs(mytree)
    depth = getTreeDepth(mytree)
    firstStr = mytree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(leafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDic = mytree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDic.keys():
        if type(secondDic[key]).__name__=='dict':
            plotTree(secondDic[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDic[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree,save_name):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getTreeLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    fig.savefig(save_name)

mytree = retriveTree(0)
#mytree['no surfacing'][3]='maybe'
print mytree
createPlot(mytree,'tree_result.jpg')
#createPlot(mytree)

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

mytree = retriveTree(0)
myDat, labels = createDataSet()
print classify(mytree,labels,[0,0])
print classify(mytree,labels,[1,0])
print classify(mytree,labels,[1,1])

def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)

storeTree(mytree,'classifierStorage.txt')
print grabTree('classifierStorage.txt')


fr = open('../data/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
print lenses
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
print lensesLabels
lensesTree = createTree(lenses, lensesLabels)
print lensesTree
createPlot(lensesTree,'lenses_result.jpg')
storeTree(lensesTree,'lensesTree.txt')
print grabTree('lensesTree.txt')