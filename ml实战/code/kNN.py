from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #获取dataset一维的长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #tile函数，复制inX，后面元组为高维到低维的长度
    sqDiffMat = diffMat**2 #元素平方
    sqDistances = sqDiffMat.sum(axis=1) #axis=1为行累加，=0为列累加
    distances = sqDistances**0.5 #元素开根号
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3)) #数组赋值为0
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #strip函数，删除头尾中传进参数中含有的字符，为空则默认删除空白符，lstrip删除头，rstrip删除尾
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1;
    return returnMat,classLabelVector

def str2int(str):
    if (str == 'largeDoses'):
        return 3
    if (str == 'smallDoses'):
        return 2
    if (str == 'didntLike'):
        return 1

def autoNorm(dataSet):
    minVals = dataSet.min(0) #(0)表示从列中选取最小值，而不是行中
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #shape(dataSet)获取dataset的维度大小
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1)) #矩阵除法使用linalg.slove（matA，matB）
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ('the classifier came back with:%d, the real answer is:%d' % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print ("the total error rate is:%f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person:" ,resultList[int(classifierResult) - 1])

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr =  fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #获取目录下文件名列表
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % filenameStr)
    testFileList  = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        filenameStr = testFileList[i]
        fileStr = filenameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % filenameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d"\
               % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of error is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))




        
