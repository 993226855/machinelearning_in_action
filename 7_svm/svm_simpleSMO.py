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

########################简单SMO算法实现########################################
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

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''# dataMatIn: 数据; classLabels: Y值; C:V值; toler:容忍程度; maxIter: 最大迭代次数'''
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    # 进行初始化
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        # m : 数据样本数
        for i in range(m):
            # 先计算FXi，这里只用了线性的 kernel，相当于不变。
            # f(x)=αiyiK(xi,x)+b
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # Ei:预测值与实际值的差值
            Ei = fXi - float(labelMat[i])
            #if checks if an example violates KKT conditions
            # KTT条件如下：yi(f(xi)-yi)
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 随机再选一个不等于i的数j,目的是制造αi、αj
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                #αi--αj对
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 控制边界条件
                # 定义了上(H)下(L)界取值范围
                # yi != yj==>L,H;
                # yi = yj==>L,H;
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print("L==H");
                    continue

                # 算出 eta = -(K11 + K22 - 2K12)
                # 这里需要添加一个负号
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0");
                    continue
                # 这里就用一个减号
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                # 控制上下界
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough");
                    continue
                # 算出αi
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1 #alphaPairsChanged就是一个标记符，记录alpha对变化了多少次
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1 #记录迭代次数
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def calcWs(alphas,dataArr,classLabels):
    X = dataArr; labelMat = classLabels
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

import matplotlib.pyplot as plt
# show your trained svm model only available with 2-D data
def showSVM(train_x,train_y,b,alphas):
    train_x,train_y = mat(train_x),mat(train_y).transpose()
    if train_x.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    # draw all samples
    for i in range(train_x.shape[0]):
        if train_y[i] == -1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')
        elif train_y[i] == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')
    # mark support vectors
    supportVectorsIndex = nonzero(alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(train_x[i, 0], train_x[i, 1], 'oy')
    # draw the classify line
    w = calcWs(alphas, train_x, train_y)

    min_x = min(train_x[:, 0])[0, 0]
    max_x = max(train_x[:, 0])[0, 0]
    y_min_x = float(-b - w[0] * min_x) / w[1]
    y_max_x = float(-b - w[0] * max_x) / w[1]

    y_min_x_h = float(-b - w[0] * min_x+1) / w[1]
    y_max_x_h = float(-b - w[0] * max_x+1) / w[1]

    y_min_x_l = float(-b - w[0] * min_x-1) / w[1]
    y_max_x_l = float(-b - w[0] * max_x-1) / w[1]

    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.plot([min_x, max_x], [y_min_x_h, y_max_x_h], '-r')
    plt.plot([min_x, max_x], [y_min_x_l, y_max_x_l], '-r')
    plt.show()


################## test svm #####################
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
kTup=('lin', 0)
# svmClassifier = smoP(train_x, train_y, C, toler, maxIter)
b,alphas = smoSimple(train_x, train_y, C, toler, maxIter)
## step 3: testing
# print("step 3: testing...")
# accuracy = testSVM(oS, mat(test_x), mat(test_y))
## step 4: show the result
print("step 4: show the result...")
# print('The classify accuracy is: %.3f%%' % (accuracy * 100))
showSVM(train_x,train_y,b,alphas)