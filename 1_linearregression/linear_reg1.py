import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
#如何封装一个类class
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_diabetes

class lr_model():
    def __init__(self):
        pass
    #def prepare_data(self):
    #def initialize_params(self, dims):
    #def linear_loss(self, X, y, w, b):
    #def linear_train(self, X, y, learning_rate, epochs):
    #def predict(self, X, params):
    # def linear_cross_validation(self, data, k_fold, randomize=True):
    #     if randomize:
    #         data = list(data)
    #         shuffle(data)
    #     slices = [data[i::k_fold] for i in range(k_fold)]
    # #表示data[i:j:k_fold],i到j每隔k_fold取一个数字，这里没有j表示i到最后
    #     for i in range(k_fold):
    #         validation = slices[i]
    #         train = [data for s in slices if s is not validation for data in s]
    #         train = np.array(train)
    #         validation = np.array(validation)
    #         yield train, validation

if __name__ == '__main__':
    lr = lr_model()
    data = lr.prepare_data()
    for train, validation in lr.linear_cross_validation(data, 5):
        X_train = train[:, :10]
        y_train = train[:, -1].reshape((-1, 1))
        X_valid = validation[:, :10]
        y_valid = validation[:, -1].reshape((-1, 1))

        loss5 = []
        loss, params, grads = lr.linear_train(X_train, y_train, 0.001, 100000)
        loss5.append(loss)
        score = np.mean(loss5)
        print('five kold cross validation score is', score)
        y_pred = lr.predict(X_valid, params)
        valid_score = np.sum(((y_pred - y_valid) ** 2)) / len(X_valid)
        print('valid score is', valid_score)