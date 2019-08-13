'''导入包'''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
'''sigmoid函数'''
def sigmoid(x):
    z=1/(1+np.exp(-x))
    return z

'''logistics损失函数'''
def logistics_loss(X, y, w, b):
    """
    线性回归模型的损失函数
    :param X: 特征变量
    :param y: 回归值
    :param w: 权重值
    :param b: 截距值
    :return:
    """
    # 训练集的样本个数
    train_num = X.shape[0]
    # 特征数量
    feature_num = X.shape[1]
    # logistics回归方程
    sita = sigmoid(np.dot(X, w) + b)

    # 损失函数
    loss = -1/train_num*np.sum(y*np.log(sita)+(1-y)*np.log(1-sita))

    # 参数求偏导
    dw = np.dot(X.T,(sita-y)) / train_num
    db = np.sum((sita - y)) / train_num
    loss = np.squeeze(loss)
    return sita, loss, dw, db

"""参数初始化"""
def initialize_params(dims):
    w = np.zeros((dims, 1))
    b = 0
    return w, b

'''logistics模型训练'''
def logistics_train(X,y,learn_rate,epochs):
    feture_num = X.shape[1]
    loss_list=[]
    # 初始化参数
    w,b = initialize_params(feture_num)
    # 梯度下降法训练
    for i in range(1, epochs):
        sita, loss, dw, db = logistics_loss(X, y, w, b)
        w += -learn_rate*dw
        b += -learn_rate*db

        if i%100==0:
            loss_list.append(loss)
            print('epoch %d loss %f' % (i,loss))
    # 保存参数
    params={
        'w':w,
        'b':b
    }
    # 保存梯度
    grads = {
        'dw': dw,
        'db': db
    }
    return loss_list,params,grads

'''训练测试数据集划分'''
def train_test_split(X, y):
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train=', X_train.shape)
    print('X_test=', X_test.shape)
    print('y_train=', y_train.shape)
    print('y_test=', y_test.shape)
    return X_train, y_train, X_test, y_test

'''logistics预测'''
def logistics_predict(X_test,params,y_prob_threhold=0.5):
    w, b =params['w'],params['b']
    y_test_prob = sigmoid(np.dot(X_test, w) + b)
    y_test_class = np.zeros((y_test_prob.shape[0],1))
    true = (y_test_prob>y_prob_threhold).T
    true_index = np.argwhere(true[0]).reshape((1,-1))
    y_test_class = y_test_class.T
    y_test_class=y_test_class[0]
    y_test_class[true_index]=1
    return y_test_class

# 生成二分数据集
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
'''生成二分数据集'''
def sample_generator(samples=100,features=3,informative=3,repeated=0,redundant=0,classes=2):
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

'''训练集与测试集的简单划分'''
def train_test_split(X, y):
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    print('X_train=', X_train.shape)
    print('X_test=', X_test.shape)
    print('y_train=', y_train.shape)
    print('y_test=', y_test.shape)
    return X_train, y_train, X_test, y_test

'''计算准确率'''
def accuracy(y_test,y_pred):
    right_nums=0
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            if y_test[i]==y_pred[j] and i==j:
                right_nums+=1
    accuracy_ratio=right_nums/len(y_test)
    return accuracy_ratio

'''划分分类界限'''
def plot_logistic(X_train, y_train, params):
    n = X_train.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if y_train[i] == 1:
            xcord1.append(X_train[i][0])
            ycord1.append(X_train[i][1])
        else:
            xcord2.append(X_train[i][0])
            ycord2.append(X_train[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=32, c='red')
    ax.scatter(xcord2, ycord2, s=32, c='green')
    x = np.arange(-1.5, 3, 0.1)
    y = (-params['b'] - params['w'][0] * x) / params['w'][1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

X,y=sample_generator(samples=100,features=2,informative=2,repeated=0,redundant=0,classes=2)
X_train, y_train, X_test, y_test=train_test_split(X,y)
loss_list,params,grads=logistics_train(X_train,y_train,0.01,1000)
y_test_class = logistics_predict(X_test,params,y_prob_threhold=0.5)
accuracy_ratio = accuracy(y_test,y_test_class)
plot_logistic(X_train, y_train, params)
