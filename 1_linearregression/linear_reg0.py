'''导入包'''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


'''封装成类'''
class lr_model():
    def __init__(self):
        pass
    #替换为我自己写的函数
    """损失函数"""
    def linear_loss(self,X, y, w, b):
        """
        线性回归模型的损失函数
        :param X: 特征变量
        :param y: 回归值
        :param w: 权重值
        :param b: 截距值
        :return:
        """
        # 回归方程
        y_hat = np.dot(X, w) + b
        # 训练集的样本个数
        train_num = X.shape[0]
        # 训练集的特征个数
        feature_num = X.shape[1]
        # 损失函数
        loss = np.sum((y_hat - y) ** 2) / train_num
        # 参数求偏导
        dw = np.dot(X.T, (y_hat - y)) / train_num
        db = np.sum((y_hat - y)) / train_num
        return y_hat, loss, dw, db

    """参数初始化"""
    def initialize_params(self,dims):
        w = np.zeros((dims, 1))
        b = 0
        return w, b

    '''模型训练'''
    def linear_model_train(self,X, y, learn_rate, epochs):
        # 初始化参数
        w, b = self.initialize_params(X.shape[1])
        loss_list = []
        for i in range(1, epochs):
            y_hat, loss, dw, db = self.linear_loss(X, y, w, b)
            loss_list.append(loss)
            w += -(learn_rate * dw)
            b += -(learn_rate * db)
            # 打印迭代次数和损失
            if i % 10000 == 0:
                print('epochs = %d loss = %f' % (i, loss))

            # 保存参数
            params = {
                'w': w,
                'b': b
            }
            # 保存梯度
            grads = {
                'dw': dw,
                'db': db
            }
        return loss_list, loss, params, grads

    '''数据加载'''
    '''导入数据'''
    def load_data(self):
        diabetes = load_diabetes()
        data = diabetes.data
        target = diabetes.target
        # 打乱数据
        X, y = shuffle(data, target, random_state=13)
        X = X.astype(np.float32)
        return X, y

    '''训练集与测试集的简单划分'''
    def train_test_split(self,X, y):
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

    '''预测'''
    def linear_predict(self,X, params):
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b
        return y_pred

    '''交叉验证'''

    def linear_cross_validation(self,X, y, k_fold=5, randomize=True):
        sample_num = X.shape[0]
        offsets = [int((i / k_fold) * sample_num) for i in range(k_fold + 1)]

        for k in range(k_fold):
            offset1 = offsets[k]
            offset2 = offsets[k + 1]
            valid_index = np.arange(offset1, offset2, 1)
            train_index = np.array(list(set(np.arange(0, sample_num, 1)).difference(set(valid_index))), dtype=int)
            X_valid, y_valid = X[valid_index], y[valid_index]
            # 选出补集
            X_train, y_train = X[train_index], y[train_index]
            # (1,20)==>(20,1)
            y_train = y_train.reshape((-1, 1))
            y_valid = y_valid.reshape((-1, 1))

            loss_list, loss, params, grads = self.linear_model_train(X_train, y_train, 0.001, 100000)
            print('cross validatation is ', loss)
            y_pred = self.linear_predict(X_valid, params)
            valid_score = np.sum(((y_pred - y_valid) ** 2)) / len(X_valid)
            print('valid score is ', valid_score)


'''主函数调用'''
if __name__ == '__main__':
    # 调用类生成对象
    # lr = lr_model()
    ############第一种方案：不做交叉验证
    # X_train, y_train, X_test, y_test = train_test_split(X, y)
    # type(X_train), type(y_train)
    # # 模型训练，就该开始训练了
    # loss_list, loss, params, grads = linear_model_train(X_train, y_train, 0.001, 100000)
    # # 打印参数
    # print(params)
    # y_pred = linear_predict(X_test, params)
    # f = X_test.dot(params['w']) + params['b']
    #
    # # 预测完了之后，还得对预测结果和真值展示看看，效果如何
    # plt.scatter(range(X_test.shape[0]), y_test)
    # plt.plot(f, color='darkorange')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.show()
    #
    # # 训练过程中损失的下降：
    # plt.plot(loss_list, color='blue')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.show()

    ############第二种方案：交叉验证
    # X, y = load_data()
    # linear_cross_validation(X=X,y=y,k_fold=5)

    ############第三种方案：调用自己的类的交叉验证
    lr = lr_model()
    X, y = lr.load_data()
    lr.linear_cross_validation(X=X, y=y, k_fold=5)




