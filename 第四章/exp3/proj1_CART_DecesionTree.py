from CART_Tree import CARTTree
import treePlotter


def create_dataset():
    # 定义训练集
    dataset_columns = ['house', 'marriage', 'income', 'delinquency']
    dataset_samples = [[1, "single", 125, 'no'],
                       [0, 'married', 100, 'no'],
                       [0, 'single', 70, 'no'],
                       [1, 'married', 120, 'no'],
                       [0, 'divorced', 95, 'yes'],
                       [0, 'married', 60, 'no'],
                       [1, 'divorced', 220, 'no'],
                       [0, 'single', 85, 'yes'],
                       [0, 'married', 75, 'no'],
                       [0, 'single', 90, 'yes']]
    return dataset_samples, dataset_columns


import numpy as np


def divide_tree_group(node, key):
    # 对于income做一些操作，使得结果更加符合任务
    org_ls = [tp for tp in node[key]]
    group_ls = [np.asarray(tp) for tp in org_ls]
    if group_ls[0].min() > group_ls[1].min():
        div_num = (group_ls[0].min() + group_ls[1].max()) / 2
        new_dict = {'>=' + str(div_num): {node[key][org_ls[0]]}, '<' + str(div_num): {node[key][org_ls[1]]}}
    else:
        div_num = (group_ls[1].min() + group_ls[0].max()) / 2
        new_dict = {'<' + str(div_num): {node[key][org_ls[0]]}, '>=' + str(div_num): {node[key][org_ls[1]]}}
    node[key] = new_dict


def tree_postprocess(Tree):
    if type(Tree) is not dict:
        return
    for key in Tree.keys():
        # 递归深度优先遍历二叉树，找到'income'结点，并将其修改为中间值
        if key == 'income':
            divide_tree_group(Tree, key)
            break
        tree_postprocess(Tree[key])


def main():
    Cart_Tree = CARTTree()
    ds, labels = create_dataset()
    Cart_Tree.getDataSet(ds, labels)
    Cart_Tree.train()  # 训练CART决策树
    tree_postprocess(Cart_Tree.tree)  # 对于已生成的决策树进行一些后处理，使其符合题目要求
    print(Cart_Tree.tree)  # 输出CART决策树
    print(Cart_Tree.predict(Cart_Tree.tree, {'house': 0, 'marriage': 'single', 'income': 55}))
    treePlotter.createPlot(Cart_Tree.tree)


if __name__ == '__main__':
    main()
