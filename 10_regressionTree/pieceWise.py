from numpy import *
import sys
sys.path.append("E:/Python-Workspace/machinelearning/10_regressionTree/")
import treeCreate as tc


def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n'
                        'try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    err =sum(power(Y - yHat, 2))
    return err

import numpy as np
import matplotlib.pyplot as plt

# input data
data = tc.loadDataset('E:/Python-Workspace/machinelearning/10_regressionTree/data/exp2.txt')
dataMat=np.mat(data)
# plot data in plane
for i in range(dataMat.shape[0]):
    plt.scatter(dataMat[i,0],dataMat[i,1],c='b',s=7,marker='o')#
# plt.show()

# modelling
tree=tc.createTree(dataMat,leafType=modelLeaf,errType=modelErr,ops=(1,10))
# print(tree)
# 结果输出
# {'spInd': 0, 'spVal': matrix([[ 0.285477]]), 'right': matrix([[
# 3.46877936], [ 1.18521743]]), 'left': matrix([[ 1.69855694e-03],
# [ 1.19647739e+01]])}
# 最后createTree() 构建的树的叶节点分别由 y=3.468+1.1852x | y=0.0016985+11.96477x模型表示
# 对比起之前regLeaf()返回的这一段区间内y值的均值，也就是形成的是一条平行于X轴的直线
