import matplotlib.pyplot as plt
import numpy as np

def build_dataset():
# 创建用于训练的数据集 create dataset to be trained
    X = np.asmatrix([10,15,20,30,50,60,60,70])
    y = np.asmatrix([0.8,1,1.8,2,3.2,3,3.1,3.5])
    return X,y

def main():
    X,Y = build_dataset()
    # 为训练集添加一列1
    X_temp = np.ones((X.shape[1],2))
    X_temp[:,0] = X
    X = np.asmatrix(X_temp)
    # 使用线性回归得到模型的权重 params -> [w,b] np.ndarray
    params = (((X.T * X).I) * X.T) * Y.T
    # 打印线性回归得到的模型
    print("The trained LinearRegression got function : Y = %f * x + %f " % (params[0], params[1]))
    plt.scatter(*build_dataset())
    plt.show()
    # 定义预测集
    X_predict = np.asarray([55,1])
    # 用模型进行预测
    Y_predict = X_predict * params
    # 打印预测结果
    print("The input square %s m^2 got predicted price %s !" % (X_predict[0],float(Y_predict)))
if __name__ == '__main__':
    main()