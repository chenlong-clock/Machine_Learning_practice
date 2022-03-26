# 导入数据集和模型所需的包
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
# 定义每次迭代的训练函数
def train_one_epoch(X,y,model):
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=10).mean()
    return scores

def main():
    # 定义迭代次数和超参数n_neighbors
    epochs = 6
    n_neighbors = 5
    # 加载鸢尾花数据集的数据和标签
    X, y = load_iris(return_X_y=True)
    # 定义用于记录最佳准确率、最佳模型和最佳n_neighbors的参数
    best_score = 0
    best_model = None
    best_n = 5
    # 进行迭代
    for iter in range(epochs):
        # 定义当前的KNN模型
        cur_model = KNN(n_neighbors=n_neighbors)
        # 训练模型
        cur_score = train_one_epoch(X,y,cur_model)
        # 判断得分是否超过最佳得分， 如果是则替换最佳模型
        if cur_score > best_score:
            best_n = n_neighbors
            best_model = cur_model
            best_score = cur_score
        print("epoch {0}, 目前的 n_neighbors {1}, 10折交叉验证的平均准确率为{2}".format(iter+1, n_neighbors, cur_score))
        # 每次迭代n_neighbors += 1
        n_neighbors += 1
    print("迭代结束")
    print("最佳模型的n_neighbors 为 {0}, 10折交叉验证的平均准确率为 {1}".format(best_n,best_score))
    # 定义用于测试的数据
    X1 = [[1.5, 3, 5.8, 2.2], [6.2, 2.9, 4.3, 1.3]]
    # 用最佳模型进行预测并得到预测结果
    predict = best_model.predict(X1)
    # 获得X1中两个鸢尾花预测得到的索引所对应的类别
    X1_classes = [load_iris().target_names[idx] for idx in predict]
    print("两个鸢尾花的预测结果为：[1.5, 3, 5.8, 2.2] ：{0}, [6.2, 2.9, 4.3, 1.3] ：{1}".format(*X1_classes))


if __name__ == '__main__':
    main()
