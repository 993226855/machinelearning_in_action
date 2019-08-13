from numpy import *
import sys
sys.path.append("E:/Python-Workspace/machinelearning/10_regressionTree/")
import treeCreate as tc

# Prepruning 预剪枝（边建立树边剪枝）
# 在treeCreate脚本中chooseBestSplit函数中就包含的tolS and tolN两个参数就是提前限制树的生长
# The trees built are sensitive to the settings we used for tolS and tolN.

#Postpruning 后剪枝（用测试数据来检测树模型的效果，决定是否将叶节点合并）

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = tc.binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = tc.binSplitDataSet(testData, tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
                       sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

data = tc.loadDataset('E:/Python-Workspace/machinelearning/10_regressionTree/data/ex00.txt')
print(data[0])
print(data[1])


