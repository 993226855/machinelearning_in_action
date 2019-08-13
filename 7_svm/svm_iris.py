import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

from matplotlib.colors import ListedColormap

'''绘制决策边界'''
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 设定标注和色带
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 绘制决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    xx1_r = xx1.ravel()
    xx2_r = xx2.ravel()
    z = classifier.predict(np.array([xx1_r,xx2_r]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())
    # 绘制样本类别

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)
    # 高亮样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0,
                    linewidths=1, marker='o', s=55, label='test set')

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split

# iris是很有名的数据集，
iris=datasets.load_iris()
X=iris.data[:,[1,2]]
y=iris.target
X_trian,X_test,y_train,y_test = \
    train_test_split(X,y,test_size=0.3,random_state=0)
# 为了追求机器学习和优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_trian)
sc.mean_
sc.scale_
X_trian_std = sc.transform(X_trian)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_trian_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

# 导入SVC
from sklearn.svm import SVC

# 线性核函数
# svm1=SVC(kernel='linear',C=0.1,random_state=0)#线性核函数
# svm1.fit(X_trian_std,y_train)
#
# svm2=SVC(kernel='linear',C=10,random_state=0)#线性核函数
# svm2.fit(X_trian_std,y_train)
#
# fig=plt.figure(figsize=(10,6))
# ax1=fig.add_subplot(1,2,1)
#
# plot_decision_regions(X_combined_std,y_combined,classifier=svm1)
# plt.xlabel('petal length [standardized]')
# plt.xlabel('petal width [standardized]')
# plt.title('C=0.1')
#
# ax2=fig.add_subplot(1,2,2)
#
# plot_decision_regions(X_combined_std,y_combined,classifier=svm2)
# plt.xlabel('petal length [standardized]')
# plt.xlabel('petal width [standardized]')
# plt.title('C=10')
# plt.show()

# 径向基核函数
svm1=SVC(kernel='rbf',gamma=0.1,C=1,random_state=0)#线性核函数
svm1.fit(X_trian_std,y_train)
svm2=SVC(kernel='rbf',gamma=10,C=1,random_state=0)#线性核函数
svm2.fit(X_trian_std,y_train)

fig=plt.figure(figsize=(10,6))
ax1=fig.add_subplot(1,2,1)

plot_decision_regions(X_combined_std,y_combined,classifier=svm1)
plt.xlabel('petal length [standardized]')
plt.xlabel('petal width [standardized]')
plt.title('C=0.1')

ax2=fig.add_subplot(1,2,2)

plot_decision_regions(X_combined_std,y_combined,classifier=svm2)
plt.xlabel('petal length [standardized]')
plt.xlabel('petal width [standardized]')
plt.title('C=10')
plt.show()