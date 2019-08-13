'''导入包'''
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from math import log

'''计算熵'''
def entropy_D(col_list):
    col_probs = col_list.value_counts()/len(col_list)
    entropy = np.sum([-col_prob*log(col_prob) for col_prob in col_probs])
    return entropy

'''数据分片'''
def splitData(data,col_x):
    '''
    split data by col_x
    :param data:train data,dataframe
    :param col_x:
    :return: new data_split
    '''
    data_split={}
    col_x_values = set(data[col_x])
    for col_x_value in col_x_values:
        data_split[col_x_value] = data[data[col_x]==col_x_value]
    return data_split

'''计算条件熵'''
def entropy_D_A(data_split,col_D):
    sample_N=0
    entropy_d_a=0
    for key, values in data_split.items():
        sample_N += len(values)
    for key,values in data_split.items():
        entropy_d = entropy_D(values[col_D])
        p_A = len(values)/sample_N
        entropy_d_a += p_A*entropy_d
    return entropy_d_a

def choose_best_feature(data,col_y):
    best_feature = None
    max_info_gain=0
    columns = list(data.columns)
    columns.remove(col_y)
    entropy_d = entropy_D(data[col_y])

    for col_x in columns:
        splitdata = splitData(data,col_x)
        entropy_d_a = entropy_D_A(splitdata,col_y)
        info_gain = entropy_d-entropy_d_a
        if info_gain>max_info_gain:
            max_info_gain=info_gain
            best_feature=col_x
        if best_feature == None:
            best_feature = col_x
    return best_feature

def decision_tree(data,col_y):
    best_feature=choose_best_feature(data,col_y)
    feature_tree = {best_feature:{}}
    best_feature_unique_values = data[best_feature].unique()
    for value in best_feature_unique_values:
        new_data1=data[data[best_feature]==value]
        new_data2=new_data1.drop(best_feature,axis=1)
        if len(list(new_data2.columns))>1:
            feature_tree[best_feature][value]=decision_tree(new_data2,col_y)
        else:
            feature_tree[best_feature][value] = list(new_data2[col_y])[0]
            break
    return feature_tree
import pandas as pd
def createData():
    data={'X1': [1,1,1,0,0,0],
          'X2': [1,1,0,1,1,1],
          'X3': ['yes','yes','no','no','no','yes'],
          'X4': ['A', 'B', 'B', 'B', 'A','A'],
          'X5': ['M', 'FM', 'M', 'M', 'FM','M'],
          'target': ['Y','Y','Y','N','N','N']}
    return pd.DataFrame(data)

data=createData()
tree=decision_tree(data,'target')

import treePlotter
treePlotter.createPlot(tree)


