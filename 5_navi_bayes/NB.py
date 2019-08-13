import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import log
import pandas as pd

'''naive bayes'''
def trainNB(train_X_y,target_col,test_X):
    num_train = len(train_X_y)
    # 先假设只有两种类型0,1
    y_class_count = train_X_y[target_col].value_counts()
    prob_y={}
    for yi in y_class_count.keys():
        prob_yi = y_class_count[yi]/num_train
        # p(x|yk)=p(x1|yk)p(x2|yk)...p(xi|yk)
        prob_x_yi = 1
        for xcoli in test_X.columns:
            train_X_yi = train_X_y[train_X_y[target_col] == yi]
            yi_num = len(train_X_yi)
            xi_yi_num = len(train_X_yi[train_X_yi[xcoli] == test_X[xcoli][0]])
            prob_xi_yi = xi_yi_num/yi_num
            prob_x_yi=prob_x_yi*prob_xi_yi
        prob_y[yi]=prob_x_yi*prob_yi
    return prob_y

def createData():
    train={'X1':[1,1,1,1,1,2,2,2,2,2],
          'X2':['S','M','M','S','S','S','M','M','L','L'],
          'Y':[-1,-1,1,1,-1,-1,-1,1,1,1]}
    test = {'X1': [2],
            'X2': ['S']}
    return pd.DataFrame(train),pd.DataFrame(test)

train_X_y,test_X=createData()
prob_y=trainNB(train_X_y,'Y',test_X)
print(prob_y)