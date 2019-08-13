'''导入包'''
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 距离计算
def distance_compute(x_pred,X_train):
    train_num = X_train.shape[0]
    x_pred_tile = np.tile(x_pred,(train_num,1))
    sq_diff_X = (x_pred_tile-X_train)**2
    distance = np.sqrt(np.sum(sq_diff_X, axis=1))
    return distance

import operator
def kNN(x_pred,X_train,y_train,k):
    distance = distance_compute(x_pred,X_train)
    distance_index = distance.argsort()
    label_count={}
    for i in range(k):
        label = y_train[distance_index[i]][0]
        if label in label_count.keys():
            label_count[label]+=1
        else:
            label_count[label]=1
    label_count_sort = sorted(label_count.items(),
                              key=operator.itemgetter(1),#key=operator.itemgetter(1)表示按照第二个域来排序
                              # label_count={'label':count}的第二域为count
                              reverse=True)
    return label_count_sort[0][0]

from sklearn.datasets.samples_generator import make_classification
def sample_generator(samples=100,features=2,informative=2,repeated=0,redundant=0,classes=2):
    X, y = make_classification(n_samples=samples,n_features=features,n_informative=informative,n_repeated=repeated,
                        n_redundant=redundant,n_classes=classes)
    rng=np.random.RandomState(2)
    X+=2*rng.uniform(size=X.shape)

    unique_labels=set(y)
    colors=plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))
    for k,color in zip(unique_labels,colors):
        x_k=X[y==k]
        plt.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=color,
                 markeredgecolor="k",markersize=5)
    plt.title('data by make_classification()')
    plt.show()
    return X,y

#导入iris数据
def load_iris():
    data,target = datasets.load_iris(return_X_y=True)
    X,y = shuffle(data,target,random_state=2)
    return X,y

'''训练集与测试集的简单划分'''
def train_test_split(X, y,train_per=0.9):
    offset = int(X.shape[0] * train_per)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train=', X_train.shape)
    print('X_test=', X_test.shape)
    print('y_train=', y_train.shape)
    print('y_test=', y_test.shape)
    return X_train, y_train, X_test, y_test



'''主函数调用'''
if __name__ == '__main__':
    # 导入数据
    X,y = load_iris()
    #训练集与测试集分开
    X_train, y_train, X_test, y_test=train_test_split(X,y,train_per=0.7)

    accuracy_list=[]
    K = 10
    #分类预测
    for k_ in range(1,K,1):
        # 预测正确的数量
        correct_num = 0
        for i in range(len(X_test)):
            y_pred = kNN(x_pred=X_test[i],X_train=X_train,y_train=y_train,k=k_)
            y_t = y_test[i][0]
            if y_t==y_pred:
                correct_num+=1
        accuracy=(correct_num/len(y_test))*100
        accuracy_list.append(accuracy)
        print('predict accuracy is {0}%'.format(accuracy))
    plt.plot(range(1,K,1), accuracy_list)
    plt.xlabel('k_value')
    plt.ylabel('accuracy')
    plt.show()