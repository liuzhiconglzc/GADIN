import numpy as np
# TODO cart分类树

def gini(y, weight=None):
    """
    :param y:一维ndarray数组
    :param weight:一维ndarray数组，样本权重
    :return:基尼系数
    """
    if weight is None:
        weight = np.array([1.0] * len(y))
    weight_sum = np.sum(weight)
    prob = np.array([np.sum(weight[y == i]) / weight_sum for i in np.unique(y)])
    return np.sum([prob_i * (1 - prob_i) for prob_i in prob])


def con_gini(x, y, weight=None):
    """

    :param x:特征，一维ndarray数组
    :param y:标签，一维ndarray数组
    :param weight:样本权重，一维ndarray数组
    :return:条件基尼系数
    """
    if weight is None:
        weight = np.array([1.0] * len(y))
    weight_sum = np.sum(weight)
    condition_gini = 0
    for i in np.unique(x):
        index = np.where(x == i)
        y_new = y[index]
        weight_new = weight[index]
        condition_gini += gini(y_new, weight_new) * np.sum(weight_new) / weight_sum

    return condition_gini


def gini_gain(x, y, weight=None):
    """

    :param x: 特征，一维ndarray数组
    :param y: 标签，一维ndarray数组
    :param weight: 样本权重，一维ndarray数组
    :return: 基尼系数变化量
    """
    if weight is None:
        weight = np.array([1.0] * len(y))
    return gini(y, weight) - con_gini(x, y, weight)


class Node(object):
    def __init__(self
                 , n_sample=None
                 , gini=None
                 , left_child_node=None
                 , right_child_node=None
                 , class_prob=None
                 , feature_index=None
                 , feature_best_value=None):
        """
        :param n_sample: 节点样本量
        :param gini: 节点标签的基尼系数
        :param left_child_node: 节点的左侧子节点
        :param right_child_node: 节点的右侧字节带你
        :param class_prob: 节点标签的概率分布
        :param feature_index: 节点使用的特征
        :param feature_best_value: 特征的最佳分割点
        """
        self.n_sample = n_sample
        self.gini = gini
        self.left_child_node = left_child_node
        self.right_child_node = right_child_node
        self.class_prob = class_prob
        self.feature_index = feature_index
        self.feature_best_value = feature_best_value


class CartClassifier(object):
    def __init__(self
                 , criterion='gini'
                 , max_depth=None
                 , min_sample_split=2
                 , min_sample_leaf=1
                 , min_impurity_decrease=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_impurity_decrease = min_impurity_decrease
        if self.criterion == 'gini':
            self.func = gini_gain

    def __transform(self, x, y, weight):
        """
        对特征x进行二分
        :param x: 特征
        :param y: 标签
        :param weight:样本权重
        :return: 二分后的特征向量和最佳分割点
        """
        x_uni = np.unique(x)
        x_split = (x_uni[:-1] + x_uni[1:]) / 2
        best_split = None
        best_gain = 0
        for i in x_split:
            x_i = np.zeros_like(x)
            x_i[x >= i] = 1
            x_i[x < i] = 0
            curr_gain = gini_gain(x_i, y, weight)
            if curr_gain > best_gain:
                best_gain = curr_gain
                best_split = i

        x_new = np.zeros_like(x)
        x_new[x >= best_split] = 1
        x_new[x < best_split] = 0
        return x_new, best_split

    def fit(self, X, y, weight=None):
        if weight is None:
            weight = np.array([1.0] * len(y))
            # 首先定义根节点
        self.root_node = Node()
        # 递归建树
        self.__plant(X, y, self.root_node, 1, weight)

    def __plant(self, X, y, node, depth, weight):
        # 拿到特征矩阵的维度
        rows, cols = X.shape
        # 记录当前节点的样本量
        node.n_sample = rows
        # 记录当前节点的基尼系数
        node.gini = gini(y, weight)
        # 记录当前节点的概率分布
        node.class_prob = {key: np.sum(weight[y == key]) / np.sum(weight) for key in np.unique(y)}
        # 如果当前节点所含样本的标签只有一种取值，无需进行划分，跳出循环
        if len(np.unique(y)) == 1:
            return
        # 如果当前节点所含样本所有特征都只有一个取值，无法进行划分，跳出循环
        if np.sum([len(np.unique(X[:, i])) for i in range(cols)]) == cols:
            return
        if self.max_depth is not None and depth >= self.max_depth:
            return
        if rows < self.min_sample_split:
            return

        best_col = None
        best_gain = 0
        best_split = None
        # 遍历寻找最佳特征和特征的最佳分割点，此处缺失值采取了和C4.5算法类似的方法进行处理
        for col in range(cols):
            nonan_index = ~np.isnan(X[:, col])
            nonan_prob = np.sum(weight[nonan_index]) / np.sum(weight)
            if len(np.unique(X[:, col][nonan_index])) == 1:
                continue
            x_transform, curr_split = self.__transform(X[:, col][nonan_index], y[nonan_index], weight[nonan_index])
            curr_gain = self.func(x_transform, y[nonan_index], weight[nonan_index]) * nonan_prob
            if curr_gain > best_gain:
                best_gain = curr_gain
                best_col = col
                best_split = curr_split

        if self.min_impurity_decrease is not None and best_gain < self.min_impurity_decrease:
            return
        # 记录当前节点所用的特征
        node.feature_index = best_col
        # 记录当前节点特征的最佳分割点
        node.feature_best_value = best_split
        x_best = X[:, best_col]
        nan_index = np.isnan(X[:, best_col])
        # 如果做子节点或者右子节点的最小样本量小于最小叶子节点样本量，跳出循环
        if len(X[:, best_col][(nan_index + (x_best < best_split))]) <= self.min_sample_leaf or len(
                X[:, best_col][(nan_index + (x_best < best_split))]) <= self.min_sample_leaf:
            return
        # 同C4.5算法相同，缺失样本会被同时划分到左子节点
        left_index = nan_index + (x_best < best_split)
        left_x = X[left_index]
        left_y = y[left_index]
        left_weight = weight[left_index]
        left_weight[np.isnan(left_x[:, best_col])] = left_weight[np.isnan(left_x[:, best_col])] * np.sum(
            weight[x_best < best_split]) / np.sum(weight[~nan_index])
        # 记录左侧子节点
        node.left_child_node = Node()
        # 左侧子节点继续递归建树
        self.__plant(left_x, left_y, node.left_child_node, depth + 1, left_weight)

        # 缺失样本也会被划分到右侧子节点
        right_index = nan_index + (x_best >= best_split)
        right_x = X[right_index]
        right_y = y[right_index]
        right_weight = weight[right_index]
        right_weight[np.isnan(right_x[:, best_col])] = right_weight[np.isnan(right_x[:, best_col])] * np.sum(
            weight[x_best >= best_split]) / np.sum(weight[~nan_index])
        # 记录右侧子节点
        node.right_child_node = Node()
        # 右侧子节点递归建树
        self.__plant(right_x, right_y, node.right_child_node, depth + 1, right_weight)

    def predict(self, x):
        results = self.predict_prob(x)
        pred = []
        # 遍历每一行，拿到每一行数据最大概率对应的类别
        for i in range(len(results)):
            p = max(results[i], key=lambda k: results[i][k])
            pred.append(p)
        return pred

    def predict_prob(self, x):
        rows = x.shape[0]
        results = []
        # 遍历每一行，拿到每一行数据属于各类的概率
        for i in range(rows):
            result = self.__search_node(x[i], self.root_node)
            results.append(result)
        return results

    def __search_node(self, x, node):
        # 如果当前节点有左侧子节点或者右侧字节点，那说明还未到达叶子节点，所有继续往下搜索
        if node.left_child_node is not None or node.right_child_node is not None:
            if x[node.feature_index] >= node.feature_best_value:
                return self.__search_node(x, node.right_child_node)
            else:
                return self.__search_node(x, node.left_child_node)
        # 如果当前节点没有左侧子节点和右侧子节点，说明已经到达叶子节点，就可以拿出叶子节点中的概率分布
        else:
            result = {i: node.class_prob.get(i, 0) for i in self.root_node.class_prob.keys()}
            return result

    def prune(self, alpha):
        return self.__pruning_node(self.root_node, alpha)

    def __pruning_node(self, node, alpha):
        if node.left_child_node is None and node.right_child_node is None:
            return
        else:
            self.__pruning_node(node.left_child_node, alpha)
            self.__pruning_node(node.right_child_node, alpha)
        # 计算剪枝前的损失函数值
        pre_cost = alpha * 2 + node.left_child_node.gini + node.right_child_node.gini
        # 计算剪枝后的损失值
        post_cost = alpha + node.gini
        if post_cost < pre_cost:
            node.left_child_node = None
            node.right_child_node = None
            node.feature_index = None
            node.feature_best_value = None

# 下面是测试方法
if __name__ == '__main__':

    a = np.random.randn(15, 4)
    print(a)
    # a = np.array([[1, 0, 0, 1],
    #               [1, 0, 0, 2],
    #               [1, 1, 0, 2],
    #               [1, 1, 1, 1],
    #               [1, 0, 0, 1],
    #               [2, 0, 0, 1],
    #               [2, 0, 0, 2],
    #               [2, 1, 1, 2],
    #               [2, 0, 1, 3],
    #               [2, 0, 1, 3],
    #               [3, 0, 1, 3],
    #               [3, 0, 1, 2],
    #               [3, 1, 0, 2],
    #               [3, 1, 0, 3],
    #               [3, 0, 0, 1]])

    # 所属类别，这里只用一个属性表示
    b = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])

    # 训练决策树
    clf = CartClassifier()
    clf.fit(a, b)

    # 测试数据
    # c = np.random.randn(3, 4)
    # 可为空None----hmf
    c = np.array([[1, 0, 0, 1],
                  [1, 0, None, 2],
                  [1, 1, 0, 2]])

    # 三条样本属于两种类别的概率
    clf.predict_prob(c)

    print(clf.predict_prob(c))

    # 后剪枝函数
    clf.prune(0.01)
    clf.predict(c)
    print(clf.predict(c))


