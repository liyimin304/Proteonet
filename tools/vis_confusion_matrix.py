import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
print(matplotlib.get_cachedir())

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    # plt.figure(figsize=(5.8, 5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig("res_cnf.png", dpi=300)


def main():
    cnf_matrix0 = np.array([[10, 0, 0, 0],
                           [1, 9, 0, 0],
                           [0, 0, 15, 0],
                           [0, 0, 0, 7]])
    cnf_matrix1 = np.array([[9, 0, 0, 1],
                            [0, 9, 0, 1],
                            [0, 0, 15, 0],
                            [0, 1, 0, 6]])
    cnf_matrix2 = np.array([[10, 0, 0, 0],
                            [0, 10, 0, 0],
                            [0, 0, 14, 0],
                            [1, 0, 1, 5]])

    cnf_matrix3 = np.array([[10, 0, 0, 0],
                            [2, 8, 0, 0],
                            [0, 0, 14, 0],
                            [0, 0, 0, 7]])
    cnf_matrix4 = np.array([[9, 0, 0, 0],
                            [1, 9, 0, 0],
                            [0, 0, 14, 0],
                            [0, 1, 0, 6]])
    attack_types = ['MN', 'igAN', 'HC', 'DNK']

    cnf_matrix = cnf_matrix0+cnf_matrix1+cnf_matrix2+cnf_matrix3+cnf_matrix4

    # plot_confusion_matrix(cnf_matrix/5, classes=attack_types, normalize=True, title='Normalized confusion matrix')

    res_con_matrix = np.array([[9, 0.6, 0, 0.2],
                            [1.8, 7.8, 0, 0.4],
                            [0, 0, 14.4, 0],
                            [0.8, 0.6, 0, 5.6]])
    plot_confusion_matrix(res_con_matrix, classes=attack_types, normalize=True, title='Normalized confusion matrix')


if __name__ == '__main__':
    main()