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
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy
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
                temp_dist=distSLC(sample,temp_center)
                if temp_dist<mindist:
                    mindist=temp_dist
                    mindistIndex=j
            labelMat[i,:]=mindistIndex,mindist**2
            if labelMat[i,0]!=mindistIndex:
                stop=True
        intialCenter = updateCent(dataSet, labelMat[:,0], k)
    return intialCenter,labelMat

import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('E:/Python-Workspace/machinelearning/8_kmeans/data/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = kmeans(datMat, numClust)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('E:/Python-Workspace/machinelearning/8_kmeans/data/Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

clusterClubs()