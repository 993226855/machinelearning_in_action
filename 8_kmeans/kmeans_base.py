from numpy import *

def loadDataSet(fileName):
    '''加载数据'''
    f = open(fileName)
    dataMat=[]
    for line in f.readlines():
        curLine = line.strip().split('\t')
        lineMat =[]
        for i in curLine:
            lineMat.append(float(i))
        dataMat.append(lineMat)
    return mat(dataMat)

# kmeans算法的流程
# ①随机指定几个簇中心
# ②计算各点到中心点的距离，并标注类别
# ③计算各簇的质心坐标
# ④质心坐标取代随机簇心坐标，循环执行②

'''距离计算函数'''
def distEclud(Xi,Xj):
    delta=Xi-Xj
    return sqrt(sum(pow(array(delta),2)))

'''随机定义几个簇心'''
def randCent(dataSet,k):
    n =shape(dataSet)[1]
    centriods=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        maxJ=max(dataSet[:,j])
        rangeJ=float(maxJ-minJ)
        centriods[:,j]=minJ+rangeJ*random.rand(k,1)
    # center={}
    # for i in range(centriods.shape[0]):
    #     center[i+1]=centriods[i,:]
    return centriods
def updateCent(dataSet,labelMat,k):
    labels = list(unique(array(labelMat)).astype('int16'))
    labels.sort()
    newCent=mat(zeros((k,dataSet.shape[1])))
    for label in labels:
        labelIndex = nonzero(labelMat.A == label)[0]
        label_dataSet = dataSet[labelIndex]
        newCent[label] = mean(label_dataSet.A,axis=0)
    return newCent

def kmeans(dataSet,k):
    n = dataSet.shape[0]
    intialCenter = randCent(dataSet, k)
    # 初始化每个样本属于哪个簇
    labelMat = mat(zeros((n, 2)))
    stop = True
    while stop:
        stop = False
        for i in range(n):
            sample = dataSet[i,:]
            mindist=inf
            mindistIndex=0
            for j in range(k):
                temp_center=intialCenter[j,:]
                temp_dist=distEclud(sample,temp_center)
                if temp_dist<mindist:
                    mindist=temp_dist
                    mindistIndex=j
            labelMat[i,:]=mindistIndex,mindist**2
            if labelMat[i,0]!=mindistIndex:
                stop=True
        intialCenter = updateCent(dataSet, labelMat[:,0], k)
    return intialCenter,labelMat

import matplotlib.pyplot as plt
def showKmeansPlot(dataSet,labelMat,updateCenter):
    # fig, ax = plt.subplots()
    labels = list(unique(array(labelMat).astype('int16')))
    labels.sort()
    colors={0:'red',1:'blue',2:'green',3:'pink',4:'grey',5:'gray',6:'orange',7:'cyan',8:'olive',9:'purple'}
    for label in labels:
        labelIndex = nonzero(labelMat.A == label)[0]
        label_dataSet = dataSet[labelIndex]
        for i in range(label_dataSet.shape[0]):
            plt.plot(label_dataSet[i, 0], label_dataSet[i, 1], marker='o',color=colors[label])#colors[label]
    for i in range(updateCenter.shape[0]):
        plt.plot(updateCenter[i, 0], updateCenter[i, 1], marker='*', color=colors[i])  # colors[i]
    plt.show()

fileName='E:/Python-Workspace/machinelearning/8_kmeans/data/testSet.txt'
data=loadDataSet(fileName)
K=[1,2,3,4,5,6,7,8,9,10]
dist=[]
for k in K:
    updateCenter,labelMat = kmeans(data,k)
    d=sum(array(labelMat[:,1]))
    dist.append(d)
    # showKmeansPlot(data,labelMat[:,0],updateCenter)
plt.plot(K,dist,'r')
plt.show()