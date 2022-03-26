# 导入数据集和模型所需的包
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split


def main():
    # 加载鸢尾花数据集的数据和标签
    X, y = load_iris(return_X_y=True)
    # 按照 训练集: 测试集 = 8: 2划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    # 定义模型
    model = KNN(n_neighbors=5)
    # 用训练集拟合模型
    model.fit(X_train,y_train)
    # 得到测试集的准确率
    test_acc = model.score(X_test,y_test)
    print("鸢尾花测试集中,模型Accuracy 为： %f" % (test_acc))
    # 加载预测的数据X1
    X1 = [[1.5, 3, 5.8, 2.2], [6.2, 2.9, 4.3, 1.3]]
    # 用模型进行预测并得到预测结果
    predict = model.predict(X1)
    # 获得X1中两个鸢尾花预测得到的索引所对应的类别
    X1_classes = [load_iris().target_names[idx] for idx in predict]
    print("两个鸢尾花的预测结果为：[1.5, 3, 5.8, 2.2] ：{0}, [6.2, 2.9, 4.3, 1.3] ：{1}".format(*X1_classes))


if __name__ == '__main__':
    main()
