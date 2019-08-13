# 看了《机器学习实战》对SVD又有了新的认识理解
# 导入包
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
# import numpy as np

def svd(data,percent=0.8):
    '''
    主成分分析方法
    :param data: list
    :return:
    '''
    U,Sigma,VT = linalg.svd(array(data))
    total_value = sum(Sigma)
    total=0
    sigma_N=0
    for s in Sigma:
        total+=s
        if total/total_value>percent:
            sigma_N += 1
            break
        sigma_N+=1
    # 重构原始数据
    new_sigma=Sigma[:sigma_N]
    Sigma_mat=mat(zeros((sigma_N,sigma_N)))
    for i in arange(sigma_N):
        Sigma_mat[i,i]=Sigma[i]
    new_U=U[:,:sigma_N]
    new_VT=VT[:sigma_N,:]
    data_transform = mat(new_U)*Sigma_mat*mat(new_VT)
    return U,Sigma,VT,new_U,new_sigma,new_VT,data_transform

def loadData():
    data=[[1,1,1,0,0],
          [2, 2, 2, 0, 0],
          [1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0],
          [5, 5, 5, 0, 0],
          [1, 1, 0, 2, 2],
          [0, 0, 0, 3, 3],
          [0, 0, 0, 1, 1]]
    return data

# # 案例一
# import os
# data=loadData()
# U,Sigma,VT,new_U,new_sigma,new_VT,data_transform = svd(data)
#
# print('U: \n{0}'.format(U))
# print('new_U: \n{0}'.format(new_U))
# print('Sigma: \n{0}'.format(Sigma))
# print('new_sigma: \n{0}'.format(new_sigma))
# print('VT: \n{0}'.format(VT))
# print('new_VT: \n{0}'.format(new_VT))
# print('data: \n{0}'.format(data))
# print('data_transform: \n{0}'.format(data_transform))

# 实践案例--基于协同过滤的推荐引擎
# 推荐引擎的基本思路是：通过计算物品或者用户的相似度来给用户推荐未曾买过的商品。
# ①计算物品、用户的相似度
# ②物品评分
# ③但是实际数据中是非常大的，计算相似度非常消耗计算资源，如何高效的计算呢？
# ④通过SVD将高维数据降维至低维，从而减少计算量

# 相似度计算
from numpy import *
from numpy import linalg as la

'''欧拉距离计算'''
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

'''皮尔逊计算'''
def pearsSim(inA,inB):
    if len(inA)<3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

'''余弦相似度'''
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

# 推荐系统的工作过程是：给定一个用户，系统会为此用户返回N个最好的推荐菜
# 第一、寻找用户没有评级的菜肴，及用户-菜品表中0的值
# 第二、对没有评级的菜品打分，也就是相似度的计算
# 第三、对这些物品的评分进行排序，返回前N个物品

def standEst(dataMat,user,simMeas,item):
    '''
    对输入的商品计算相似度：输入一个未评分的商品，计算该商品与其他商品的相似度，并以此计算评分
    :param dataMat:
    :param user:待推荐的用户
    :param simMeas:相似度计算模型
    :param item:未评分的商品，也是待推荐的商品
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0:
            continue
        # 寻找两个用户都评级的物品
        overlap = nonzero(logical_and(dataMat[:,item].A>0,
            dataMat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overlap, item],
                                 dataMat[overlap, j])
        simTotal+=similarity
        ratSimTotal += similarity * userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    # 寻找用户没有评级的菜肴
    unrateItems=nonzero(dataMat[user,:].A==0)[1]
    if len(unrateItems)==0:
        return 'you rated everything'
    itemScores=[]
    for item in unrateItems:
        estimatedScore=estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

# 上述的推荐步骤是基于庞大的原始数据，你会发现原始数据中包含很多0值，也就是数据是稀疏矩阵
# 无效数据太多，会极大的影响计算速度，因此SVD就派上用场了。下面会将SVD用于数据降维，降低计算开销

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat,user,simMeas,item):
    '''
    对输入的商品计算相似度：输入一个未评分的商品，计算该商品与其他商品的相似度，并以此计算评分
    :param dataMat:
    :param user:待推荐的用户
    :param simMeas:相似度计算模型
    :param item:未评分的商品，也是待推荐的商品
    :return:
    '''
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT=la.svd(dataMat)
    new_Sigma=mat(eye(4)*Sigma[:4])#这里的4表示选择前4个主成分，其实需要事先实验

    xformedItems=dataMat.T*U[:,:4]*new_Sigma.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity = simMeas(xformedItems[item, :].T,
                             xformedItems[j, :].T)
        print('the %d and %d similarity is :%f' % (item,j,similarity))
        simTotal+=similarity
        ratSimTotal += similarity * userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal

data = loadExData2()
recommend(data,1,estMethod=svdEst)




