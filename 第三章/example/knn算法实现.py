from numpy import *
import operator

def createDataSet():
    group = array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30]])    #样本点数据
    labels = ['A','A','A','D','D','D']
    return group,labels

# 使用KNN进行分类
def KNNClassify(newInput, dataSet, labels, k):  
    numSamples = dataSet.shape[0] # shape[0] 表示行数
 
    # 计算欧氏距离
    diff = tile(newInput, (numSamples, 1)) - dataSet # 计算元素属性值的差
    squaredDiff = diff ** 2 # 对差值取平方  
    squaredDist = sum(squaredDiff, axis = 1) # 按行求和 
    distance = squaredDist ** 0.5  

    # 对距离进行排序  
    # argsort() 返回按照升序排列的数组的索引
    sortedDistIndices = argsort(distance)

    classCount = {} # 定义字典
    for i in range(k):  
        # 选择前k个最短距离 
        voteLabel = labels[sortedDistIndices[i]]  
 
        # 累计标签出现的次数  
        # 如果在标签在字典中没有出现的话, get()会返回0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  

    # 返回得到的投票数最多的分类
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
    return maxIndex


dataSet, labels = createDataSet()
 
testX = array([15, 50])
k = 3
outputLabel = KNNClassify(testX, dataSet, labels, k)
print("Your input is:", testX, "and classified to class: ", outputLabel)
 
testX = array([80, 70])
outputLabel = KNNClassify(testX, dataSet, labels, k)
print("Your input is:", testX, "and classified to class: ", outputLabel)
