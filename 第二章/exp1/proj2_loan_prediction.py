import matplotlib.pyplot as plt
import numpy as np

def build_dataset():
    # 创建用于训练的数据集 create dataset to be trained
    X = np.asmatrix([[4000, 8000, 5000, 7500, 12000],
                     [25, 30, 28, 33, 40]])
    y = np.asmatrix([20000, 70000, 35000, 50000, 85000])
    return X, y


def main():
    X, Y = build_dataset()
    # 为训练集添加一列1
    X_temp = np.ones((X.shape[1], 3))
    X_temp[:, :2] = X.transpose()
    X = np.asmatrix(X_temp)
    # 使用线性回归得到模型的权重 params -> [w1, w2, b] np.ndarray
    params = (((X.T * X).I) * X.T) * Y.T
    # 打印线性回归得到的模型
    print("The trained LinearRegression got function : Y = %f * x1 + %f * x2 + %f " % (params[0], params[1], params[2]))
    # 定义预测集
    X_predict = np.asarray([18000, 30, 1])
    # 用模型进行预测
    Y_predict = X_predict * params
    # 打印预测结果
    print("The input salary %s and age %s got predicted loan %s !" % (X_predict[0], X_predict[1], float(Y_predict)))


if __name__ == '__main__':
    main()