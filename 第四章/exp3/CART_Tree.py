from math import log2


class CARTTree(object):
    def __init__(self):
        self.tree = {}  # CART Tree
        self.dataSet = []  # 数据集
        self.labels = []  # 标签集

    def getDataSet(self, dataset, labels):
        self.dataSet = dataset
        self.labels = labels

    def train(self):
        # labels = copy.deepcopy(self.labels)
        labels = self.labels[:]
        self.tree = self.buildTree(self.dataSet, labels)

    def buildTree(self, dataSet, labels):
        classList = [ds[-1] for ds in dataSet]  # 提取样本的类别
        if classList.count(classList[0]) == len(classList):  # 单一类别
            return classList[0]
        if len(dataSet[0]) == 1:  # 没有属性需要划分了
            return self.classify(classList)
        bestFeat, bestGroup = self.findBestSplit(dataSet)  # 选取最大增益的属性序号和分组(如果多于两个特征值)
        bestFeatLabel = labels[bestFeat]
        tree = {bestFeatLabel: {}}  # 构造一个新的树结点
        del (labels[bestFeat])  # 从总属性列表中去除最大增益属性
        featValues = list(bestGroup)  # 抽取最大增益属性的取值列表
        for value in featValues:  # 对于每一个属性类别
            subLabels = labels[:]
            subDataSet = self.splitDataSet(dataSet, bestFeat, value)  # 分裂结点
            subTree = self.buildTree(subDataSet, subLabels)  # 递归构造子树
            tree[bestFeatLabel][value] = subTree
        return tree

    # 计算出现次数最多的类别标签
    def classify(self, classList):
        items = dict([(classList.count(i), i) for i in classList])
        return items[max(items.keys())]

    # 判断元组中是否均为字符串
    def dete_str(self, data_Tuple):
        for elem in data_Tuple:
            if type(elem) != str:
                return False
        return True

    def splitGroup(self, vals):
        # 判断属性是否有多个特征值,如果小于等于2个则直接返回
        if len(vals) <= 2:
            return [vals]
        # 否则应当对特征进行分组
        vals = list(vals)
        retList = []
        ls_len = len(vals)
        if self.dete_str(vals):
            for i in range(ls_len):
                # 如果是字符串，说明特征为婚姻状况，定义分组列表
                retList.append((vals[i], tuple(elem for elem in vals if elem is not vals[i])))
        else:
            vals.sort()
            # 如果不是字符串，则是收入，定义收入的分组列表
            for i in range(ls_len - 1):
                retList.append((tuple(vals[j] for j in range(i + 1)), tuple(vals[k] for k in range(i + 1, ls_len))))
        # 返回分组的列表
        return retList

    # 计算最优特征
    def findBestSplit(self, dataset):
        numFeatures = len(dataset[0]) - 1
        baseGini = self.calcGini(dataset)  # 基础基尼系数
        num = len(dataset)  # 样本总数
        bestInfoGain = 0.0
        bestGroup = None
        bestFeat = -1  # 初始化最优特征向量轴
        # 遍历数据集各列，寻找最优特征轴
        for i in range(numFeatures):
            featValues = [ds[i] for ds in dataset]
            uniqueFeatValues = set(featValues)
            """
            最重要的地方：对于CART决策树来说，多个特征值需要分组，从而形成二叉树，因此需提前先把某个属性的多个特征分为两组
            """
            groups = self.splitGroup(uniqueFeatValues)
            # 按列和唯一值，计算基尼系数
            for group in groups:
                newGini = 0.
                for val in group:
                    subDataSet = self.splitDataSet(dataset, i, val)
                    prob = len(subDataSet) / float(num)  # 子集中的概率
                    newGini += prob * self.calcGini(subDataSet)
                infoGain = baseGini - newGini  # 信息增益
                if infoGain > bestInfoGain:  # 挑选最大值
                    bestInfoGain = baseGini - newGini
                    bestFeat = i
                    # 记录最佳的分组
                    bestGroup = group
                #    由于重新分组的缘故，需要把最佳的分组返回，便于生成二叉树
        return bestFeat, bestGroup  # 返回最佳分组

    # 从dataset数据集的feat特征中，选取值为value的数据
    def splitDataSet(self, dataset, feat, values):
        retDataSet = []
        for featVec in dataset:
            if type(values) is int:
                values = (values,)
            if featVec[feat] in values:
                reducedFeatVec = featVec[:feat]
                reducedFeatVec.extend(featVec[feat + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # 计算dataSet的基尼系数
    def calcGini(self, dataSet):
        num = len(dataSet)  # 样本集总数
        classList = [c[-1] for c in dataSet]  # 抽取分类信息
        labelCounts = {}
        for cs in set(classList):  # 对每个分类进行计数
            labelCounts[cs] = classList.count(cs)

        Gini = 1.
        for key in labelCounts:
            prob = labelCounts[key] / float(num)
            Gini -= prob ** 2
        return Gini

    # 预测。对输入对象进行Cart Tree分类
    def predict(self, tree, newObject):
        #    判断输入值是否为“dict”
        while type(tree).__name__ == 'dict':
            key = list(tree.keys())[0]
            # 对outcome进行一些操作
            if key == 'income':
                digi_str = list(tree[key].keys())[0]
                digit = float(digi_str[1:] if digi_str.startswith('<') else digi_str[2:])
                if newObject['income'] < digit:
                    newObject['income'] = '<' + str(digit)
                else:
                    newObject['income'] = '>=' + str(digit)
            for val in tree[key]:
                if type(val) is tuple:
                    newObject[key] = val
            tree = tree[key][newObject[key]]
        return tree
