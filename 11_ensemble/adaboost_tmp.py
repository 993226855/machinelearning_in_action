import numpy as np

class AdaBoost:
    '''
    '''
    def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):
        retArray = np.ones((dataMatrix.shape[0],1))
        if threshIneq=='lt':
            retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
        else:
            retArray[dataMatrix[:, dimen] > threshVal] = -1.0
        return retArray

    def buildStump(self,dataArr,classLabels,D):
        dataMatrix =np.mat(dataArr)
        labelMat=np.mat(classLabels).T
        m,n=dataMatrix.shape
        numSteps=10.0
        bestStump ={}
        bestClassEst=np.mat(np.zeros((m,1)))
        minError = np.inf
        for i in range(n):
            rangeMin=dataMatrix[:,i].min()
            rangeMax = dataMatrix[:, i].max()
            stepSize=(rangeMax-rangeMin)/numSteps
            for j in range(-1,int(numSteps)+1):
                for inequal in ['lt','gt']:
                    threshVal=(rangeMin+float(j)*stepSize)
                    predictedVals=\
                        self.stumpClassify(dataMatrix,i,threshVal,inequal)
                    errArr=np.mat(np.ones((m,1)))
                    errArr[predictedVals==labelMat]=0
                    weightedError=D.T*errArr
                    if weightedError<minError:
                        minError=weightedError
                        bestClassEst=predictedVals.copy()
                        bestStump['dim']=i
                        bestStump['thresh']=threshVal
                        bestStump['ineq']=inequal
        return bestStump,minError,bestClassEst

    def adaBoostTrainDS(self,dataArr,classLabels,numIt=40):
        weakClassArr=[]
        m=dataArr.shape[0]
        D=np.mat(np.ones((m,1))/m)
        aggClassEst=np.mat(np.zeros((m,1)))
        for i in range(numIt):
            bestStump, error, classEst=\
                self.buildStump(dataArr,classLabels,D)
            print("D:{}".format(D.T))
            alpha=float(0.5*np.log((1.0-error)/np.max(error,np.inf)))
            bestStump['alpha']=alpha
            weakClassArr.append(bestStump)
            print("classEst:{}".format(classEst.T))
            expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
            D=np.multiply(D,np.exp(expon))
            D=D/D.sum()
            aggClassEst+=alpha*classEst
            aggErrors=np.multiply(np.sign(aggClassEst)!=
                                  np.mat(classLabels).T,np.ones((m,1)))
            errorRate=aggErrors.sum()/m
            print("errorRate:{}".format(errorRate))
            if errorRate==0.0:
                break
        return weakClassArr