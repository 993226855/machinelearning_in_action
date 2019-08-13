'''导入包'''
from numpy import *
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import log

# 1、选择最优的特征和数据分割点
# 2、基于最优的特征和数据分割点将数据分成左右两部分
# 3、基于2中的左右两份数据构建左、右树
# 4、因为3中又是到了构建左、右树，那么就可以递归执行1--3的步骤

def loadDataset(fileName):
    dataset=[]
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split('\t')
        curLineList=[]
        for filed in curLine:
            curLineList.append(float(filed))
            dataset.append(curLineList)
    return dataset

def binSplitData(dataSet,splitfeature,splitvalue):
    '''
    将数据分成两部分
    :param dataSet:原始数据，矩阵
    :param splitfeature:分割列
    :param splitvalue:分割值
    :return:分割出来的两部分数据
    '''
    mat0=dataSet[nonzero(dataSet[:, splitfeature] > splitvalue)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, splitfeature]<= splitvalue)[0], :]
    return mat0,mat1
def regLeaf(dataSet):
    '''返回Y的均值'''
    return mean(dataSet[:,-1])
def regErr(dataSet):
    '''计算数据集的方差总和，反映了数据的离散程度'''
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType,errType,ops):
    '''
    该函数的目标是找到数据集切分的最佳位置
    :param dataSet:矩阵
    :param leafType:
    :param errType:
    :param ops:=(tolS,tolN),分别是容忍误差，最小样本数
    :return:最优分割样本，以及最优分割样本的分割值
    '''
    tolS,tolN=ops[0],ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n=shape(dataSet)
    S=errType(dataSet)
    bestS=inf
    bestIndex = 0
    bestValue=0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):#TypeError: unhashable type: 'matrix'
            mat0,mat1=binSplitData(dataSet,featIndex,splitVal)
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS=errType(mat0)*errType(mat1)
            if newS<bestS:
                bestIndex=featIndex
                bestValue=splitVal
                bestS=newS
    if (S-bestS)<tolS:
        return None,leafType(dataSet)
    mat0, mat1 = binSplitData(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue,



def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    leftSet,rightSet=binSplitData(dataSet,feat,val)
    ws_origin = leafType(dataSet)
    ws_left = leafType(rightSet)
    ws_right = leafType(leftSet)
    X_min = dataSet[:,0].min()-0.01
    X_max = dataSet[:, 0].max()+0.01

    X_xlim = mat(linspace(X_min, X_max, 100)).T
    m, n = X_xlim.shape
    X_mat = mat(ones((m, 2)))
    X_mat[:, 1] = X_xlim
    Y_mat=X_mat * ws_origin

    X_xlim1 = mat(linspace(X_min, val, 50)).T
    m, n = X_xlim1.shape
    X_mat1 = mat(ones((m, 2)))
    X_mat1[:, 1] = X_xlim1
    Y_mat1 = X_mat1 * ws_left

    X_xlim2 = mat(linspace(val,X_max, 50)).T
    m, n = X_xlim2.shape
    X_mat2 = mat(ones((m, 2)))
    X_mat2[:, 1] = X_xlim2
    Y_mat2 = X_mat2 * ws_right

    plt.plot(X_xlim, Y_mat, c='b')
    plt.plot(X_xlim1, Y_mat1, c='r')
    plt.plot(X_xlim2,Y_mat2,c='g')
    plt.show()

    # retTree['left']=createTree(leftSet,leafType,errType,ops)
    # retTree['right'] = createTree(rightSet, leafType, errType,ops)
    return retTree

# input data
# data = loadDataset('E:/Python-Workspace/machinelearning/10_regressionTree/data/ex00.txt')
# dataMat=mat(data)

# plot data in plane
# for i in range(dataMat.shape[0]):
#     plt.plot(dataMat[i,0],dataMat[i,1],c='r',marker='o')#
# plt.show()

# 建模
# tree = createTree(dataMat,ops=(1,100))
# print(tree)

