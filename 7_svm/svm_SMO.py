'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    '''
    随机选择j,j为αj的选择依据
    :param i:得到 αi
    :param m:样本大小
    :return:j
    '''
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def kernelTrans(X, A, kTup):
    '''
    计算核值，转换到高维到空间
    calc the kernel or transform data to a higher dimensional space
    :param X:特征矩阵
    :param A:一个一个的样本
    :param kTup:核函数类型
    :return:
    '''
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin':
        K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

# 该类仅仅作为存储数据的类，下面的方法不属于他
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

'''预测值与实际值之差'''
def calcEk(oS, k):
    fXk = (multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b).astype('float64')
    # fXk1 = multiply(oS.alphas, oS.labelMat).T
    # fXk2 = fXk1 * oS.K[:, k] + oS.b
    fXk3 = oS.labelMat[k].astype('float64')
    # fXk = fXk2.astype('float64')
    Ek = fXk - fXk3
    return Ek

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H");
            return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]  #changed for kernel
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T \
        #       - oS.X[i, :] * oS.X[i, :].T \
        #       - oS.X[j,:] * oS.X[j,:].T
        if eta >= 0:
            print("eta>=0");
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough");
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS

def calcWs(alphas,dataArr,classLabels):
    supportVectorsIndex = nonzero(alphas.A > 0)[0]
    supportVectors      = dataArr[supportVectorsIndex]
    supportVectorLabels = classLabels[supportVectorsIndex]
    supportVectorAlphas = alphas[supportVectorsIndex]
    m,n = shape(supportVectors)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(supportVectorAlphas[i]*supportVectorLabels[i],supportVectors[i,:].T)
    return w

def SVM_accuracy(svm, test_x, test_y):
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_y=test_y.transpose()
    for i in range(numTestSamples):
        kernelValue = kernelTrans(svm.X, test_x[i, :], kTup=('lin', 0))
        predict = kernelValue.T * multiply(svm.labelMat, svm.alphas) + svm.b
        if sign(predict) == sign(test_y[i,0]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    return accuracy

import matplotlib.pyplot as plt
# show your trained svm model only available with 2-D data
def showSVM(svm):
    if svm.X.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    # draw all samples
    for i in range(svm.X.shape[0]):
        if svm.labelMat[i] == -1:
            plt.plot(svm.X[i, 0], svm.X[i, 1], 'or')
        elif svm.labelMat[i] == 1:
            plt.plot(svm.X[i, 0], svm.X[i, 1], 'ob')
    # mark support vectors
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.X[i, 0], svm.X[i, 1], 'oy')
    # draw the classify line
    w = calcWs(svm.alphas,svm.X,svm.labelMat)
    min_x = min(svm.X[:, 0])[0, 0]
    max_x = max(svm.X[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    y_min_x_h = float(-svm.b - w[0] * min_x+1) / w[1]
    y_max_x_h = float(-svm.b - w[0] * max_x+1) / w[1]
    y_min_x_l = float(-svm.b - w[0] * min_x-1) / w[1]
    y_max_x_l = float(-svm.b - w[0] * max_x-1) / w[1]

    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-r')
    plt.plot([min_x, max_x], [y_min_x_h, y_max_x_h],':', '-g')
    plt.plot([min_x, max_x], [y_min_x_l, y_max_x_l],':', '-g')
    plt.show()
## step 1: load data
print("step 1: load data...")
fileName = 'E:/Python-Workspace/machinelearning/7_svm/data/testSet.txt'
dataSet,labels=loadDataSet(fileName)
train_x = dataSet[0:81]
train_y = labels[0:81]
test_x = dataSet[80:101]
test_y = labels[80:101]
## step 2: training...
print("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 50
oS = smoP(train_x, train_y, C, toler, maxIter)
## step 3: testing
print("step 3: testing...")
accuracy = SVM_accuracy(oS, mat(test_x), mat(test_y))
## step 4: show the result
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showSVM(oS)