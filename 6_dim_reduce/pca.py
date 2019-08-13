# 看了《机器学习实战》对PCA又有了新的认识理解
# 伪代码：
# 1、去除平均值
# 2、计算协方差矩阵
# （现在终于理解为什么要求协方差矩阵了：
# PCA就是提取前N维的方差信息，所以先求协方差）
# 3、计算协方差矩阵的特征值和特征向量
# 4、将特征值从大到小排序
# 5、保留最上面的N个特征向量
# 6、将数据转换到上述N个特征向量构建的新空间中

# 导入包
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np

def pca(data, topNfeat=9999999):
    '''
    主成分分析方法
    :param data: dataframe
    :param topNfeat: 选择几个特征
    :return:
    '''
    dataMat=data
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def loadData(fileName,delim='\t'):
    '''
    加载数据
    :param fileName:
    :param delim:
    :return: 返回数据matrix
    '''
    file = open(fileName)
    stringArr = [line.strip().split(delim) for line in file.readlines()]
    floatArr=[]
    for line in stringArr:
        float_line=[]
        for ele in line:
            float_line.append(float(ele))
        floatArr.append(float_line)
    return mat(floatArr)

def replaceNanWithMean(fileName,delim):
    '''
    替换nan值为该列的均值
    :param fileName:
    :param delim:
    :return:
    '''
    dataMat=loadData(fileName,delim)
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal=mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        # a=dataMat[:,i].A  #matrix.A表示矩阵转数组
        # a1 = dataMat[:, i]
        # b=~isnan(a) # ~表示补集
        # c=nonzero(b)
        # d=dataMat[c[0],i]
        # meanVal = mean(d)
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i]=meanVal
    return dataMat

# # 案例一
# import os
# filePath='E:\\Python-Workspace\\machinelearning\\6_dim_reduce\\data\\testSet.txt'
# data = loadData(filePath,delim='\t')
# lowDimData,reconData = pca(data,1)
# lowDimData.shape
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(array(data)[:,0],array(data)[:,1],marker='^',s=90)
# ax.scatter(reconData[:,0].flatten().A[0],
#            reconData[:, 1].flatten().A[0],marker='*',c='red',s=90)

# 案例二：半导体数据降维
secomFilePath='E:\\Python-Workspace\\machinelearning\\6_dim_reduce\\data\\secom.data'
secomData=replaceNanWithMean(secomFilePath,' ')
lowDimData,reconData = pca(secomData,5)
#below is a quick hack copied from pca.pca()
meanVals = mean(secomData, axis=0)
meanRemoved = secomData - meanVals #remove mean
covMat = cov(meanRemoved, rowvar=0)
eigVals,eigVects = linalg.eig(mat(covMat))
eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
eigValInd = eigValInd[::-1]#reverse
sortedEigVals = eigVals[eigValInd]
total = sum(sortedEigVals)
varPercentage = sortedEigVals/total*100

percent_it=0
percent_its=[]
for i in varPercentage:
    percent_it+=i
    percent_its.append(percent_it)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, 21), varPercentage[:20], marker='^')
ax.plot(range(1, 21), percent_its[:20], marker='*')
plt.xlabel('Principal Component Number')
plt.ylabel('Percentage of Variance')
plt.show()


