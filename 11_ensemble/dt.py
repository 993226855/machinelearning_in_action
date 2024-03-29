import numpy as np

class Node:
    def __init__(self, left, right, rule):
        '''
        node class,tree's node
        :param left:left tree
        :param right:right tree
        :param rule:[best feature,featureValue]
        '''
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]
class Leaf:
    def __init__(self, value):
        """
        `value` is an array of class probabilities if classifier is True, else
        the mean of the region：如果是分类模型那么叶节点是一系列分类概率，否则就是数据子集的均值
        """
        self.value = value
class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        n_feats=None,
        criterion="entropy",
        seed=None,
    ):
        """
        A decision tree model for regression or classification problems.

        Parameters
        ----------
        classifier : bool (default: True)
            Whether to treat target values as categorical (True) or
            continuous (False)
        max_depth: int (default: None)
            The depth at which to stop growing the tree. If None, grow the tree
            until all leaves are pure.
        n_feats : int (default: None) (select n_feats features from all feature,this is a randomly show)
            Specifies the number of features to sample on each split. If None,
            use all features on each split.
        criterion : str (default: 'entropy') 评判标准（信息增益、信息增益比、基尼系数）
            The error criterion to use when calculating splits. When
            `classifier` is False, valid entries are {'mse'}. When `classifier`
            is True, valid entries are {'entropy', 'gini'}.
        seed : int (default: None)
            Seed for the random number generator
        """
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None

        self.n_feats = n_feats
        self.criterion = criterion
        self.classifier = classifier
        self.max_depth = max_depth if max_depth else np.inf

        if not classifier and criterion in ["gini", "entropy"]:
            raise ValueError(
                "{} is a valid criterion only when classifier = True.".format(criterion)
            )
        if classifier and criterion == "mse":
            raise ValueError("`mse` is a valid criterion only when classifier = False.")

    def fit(self, X, Y):
        """
        Trains a binary decision tree classifier.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features
        Y : numpy array of shape (N,)
            An array of integer labels ranging between [0, n_classes-1] for
            each example in X if `self.classifier`=True else the set of target
            values for each example in X.
        """
        self.n_classes = max(Y) + 1 if self.classifier else None
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
        Use the trained decision tree to classify or predict the examples in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        preds : numpy array of shape (N,)
            The integer class labels predicted for each example in X if
            classifier = True, otherwise the predicted target values.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        """
        Use the trained decision tree to return the class probabilities for the
        examples in X.

        Parameters
        ----------
        X : numpy array of shape (N, M)
            The training data of N examples, each with M features

        Returns
        -------
        preds : numpy array of shape (N, n_classes)
            The class probabilities predicted for each example in X
        """
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        return np.array([self._traverse(x, self.root, prob=True) for x in X])

    def _grow(self, X, Y):
        # if all labels are the same, return a leaf
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])#如果是分类的问题返回分类概率，
            # 否则返回叶节点上的值

        # if we have reached max_depth, return a leaf
        if self.depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)

        N, M = X.shape
        self.depth += 1
        # select some feature randomly
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # greedily select the best split according to `criterion`
        feat, thresh = self._segment(X, Y, feat_idxs)#select feature and threshold
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        # grow the children that result from the split
        left = self._grow(X[l, :], Y[l])
        right = self._grow(X[r, :], Y[r])
        return Node(left, right, (feat, thresh))

    def _segment(self, X, Y, feat_idxs):
        """
        Find the optimal split rule (feature index and split threshold) for the
        data according to `self.criterion`.
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            vals = X[:, i]
            levels = np.unique(vals)
            thresholds = (levels[:-1] + levels[1:]) / 2 #beautiful skill,get average value
            gains = np.array([self._impurity_gain(Y, t, vals) for t in thresholds])

            if gains.max() > best_gain:
                split_idx = i
                best_gain = gains.max()
                split_thresh = thresholds[gains.argmax()]
        return split_idx, split_thresh

    def _impurity_gain(self, Y, split_thresh, feat_values):
        """
        Compute the impurity gain associated with a given split.
        IG(split) = loss(parent) - weighted_avg[loss(left_child), loss(right_child)]
        信息增益计算，这个算法有别于之前学习的，切记
        """
        if self.criterion == "entropy":
            loss = entropy
        elif self.criterion == "gini":
            loss = gini
        elif self.criterion == "mse":
            loss = mse

        parent_loss = loss(Y) #信息熵

        # generate split
        left = np.argwhere(feat_values <= split_thresh).flatten()
        right = np.argwhere(feat_values > split_thresh).flatten()

        if len(left) == 0 or len(right) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(Y)
        n_l, n_r = len(left), len(right)
        e_l, e_r = loss(Y[left]), loss(Y[right])
        child_loss = (n_l / n) * e_l + (n_r / n) * e_r

        # impurity gain is difference in loss before vs. after split
        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        '''
        输入待分类（预测）的样本，根据生成的树模型一个节点一个节点的去判断
        决策树本质就是一系列规则的集合
        :param X:待预测样本
        :param node:树模型
        :param prob:是否预测概率
        :return:返回预测类别
        '''
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            else:
                return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        else:
            return self._traverse(X, node.right, prob)

def mse(y):
    """
    Mean squared error for decision tree (ie., mean) predictions
    """
    return np.mean((y - np.mean(y)) ** 2)

def entropy(y):
    """
    Entropy of a label sequence
    """
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gini(y):
    """
    Gini impurity (local entropy) of a label sequence
    """
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])