import numpy as np

from dt import DecisionTree
from losses import MSELoss, CrossEntropyLoss,ClassifyLoss

class AdaBoost:
    '''
    '''
    def __init__(
        self,
        n_iter,
        max_depth=None,
        classifier=True,
        # learning_rate=1,
        # loss="crossentropy",
        # step_size="constant",
        # split_criterion="entropy",
    ):
        # self.loss = loss
        self.weights = None
        self.learners = None
        self.out_dims = None
        self.n_iter = n_iter
        self.base_estimator = None
        self.max_depth = max_depth
        # self.step_size = step_size
        self.classifier = classifier
        # self.learning_rate = learning_rate
        # self.split_criterion = split_criterion

    def fit(self, X, Y):
        # if self.loss == "mse":
        #     loss = MSELoss()
        # elif self.loss == "0_1_err":
        #     loss = ClassifyLoss()
        N, M = X.shape
        self.learners = np.empty((self.n_iter, 1), dtype=object)
        self.alpha = np.ones((self.n_iter, 1))#
        self.weights = np.mat(np.ones((N, 1)) / N)#样本权重
        Y_pred = np.zeros((N, 1))

        for i in range(0, self.n_iter):#迭代几次则拟合几棵树
            # use MSE as the surrogate loss when fitting to negative gradients
            t = DecisionTree(
                classifier=False, max_depth=self.max_depth, criterion="entropy"
            )
            t.fit(X, Y)
            self.learners[i] = t

            Y_pred = t.predict(X)
            errArr = np.mat(np.ones((N, 1)))
            errArr[Y_pred == Y] = 0
            weightedError = self.weights * errArr
            self.alpha[i] = float(0.5 * np.log((1.0 - weightedError) / np.max(weightedError, np.inf)))

            expon = np.multiply(-1 * self.alpha[i] * np.mat(Y).T, Y_pred)
            self.weights = np.multiply(self.weights, np.exp(expon))
            self.weights = self.weights / self.weights.sum()

    def predict(self, X):
        Y_pred = np.zeros((X.shape[0],1))
        for i in range(self.n_iter):
            Y_pred += self.alpha[i] * self.learners[i].predict(X)
        if self.classifier:
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred

