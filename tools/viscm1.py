import itertools
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
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
    plt.savefig("F:\logs\png\cm/res_cnf.png", dpi=300)


def main():
    cnf_matrix0 = np.array([[6, 4, 0, 0],
                           [2, 8, 0, 0],
                           [0, 0, 15, 0],
                           [1, 1, 0, 5]])
    cnf_matrix1 = np.array([[8, 2, 0, 0],
                            [3, 7, 0, 0],
                            [0, 0, 15, 0],
                            [0, 1, 0, 6]])
    cnf_matrix2 = np.array([[6, 4, 0, 0],
                            [0, 9, 0, 1],
                            [0, 0, 14, 0],
                            [0, 2, 0, 5]])

    cnf_matrix3 = np.array([[5, 5, 0, 0],
                            [0, 10, 0, 0],
                            [0, 0, 14, 0],
                            [0, 0, 0, 7]])
    cnf_matrix4 = np.array([[6, 3, 0, 0],
                            [1, 8, 0, 1],
                            [0, 0, 14, 0],
                            [0, 4, 1, 2]])
    #attack_types = ['MN', 'igAN', 'HC', 'DNK']
    attack_types = ['HC', 'DN', 'MN', 'igAN']

    cnf_matrix = cnf_matrix0+cnf_matrix1+cnf_matrix2+cnf_matrix3+cnf_matrix4

    # plot_confusion_matrix(cnf_matrix/5, classes=attack_types, normalize=True, title='Normalized confusion matrix')

    res_con_matrix = np.array([[1, 0, 0, 0],
                            [0.03, 0.71, 0.03, 0.23],
                            [0, 0.0, 0.63, 0.37],
                            [0.0, 0.0, 0.12, 0.84]])
    plot_confusion_matrix(res_con_matrix , classes=attack_types, normalize=True, title='Normalized confusion matrix')


def plot_acc_loss(file1, file2,pic_dir):
    data1 = pd.read_csv(file1, header=None)
    data2 = pd.read_csv(file2, header=None)
    #data3 = pd.read_csv(file3, header=None)

    total_epoch1 = data1[0]
    train_loss1 = data1[1]
    Val_Acc1 = data1[2]

    total_epoch2 = data2[0]
    train_loss2 = data2[1]
    Val_Acc2 = data2[2]

    #total_epoch3 = data3[0]
    #train_loss3 = data3[1]
    #Val_Acc3 = data3[2]

    fig, ax1 = plt.subplots()
    # color = 'tab:black'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 2)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    lns11 = ax1.plot(total_epoch1, train_loss1, 'black', linewidth = 1, label='ProteoNet Train loss')
    lns12 = ax1.plot(total_epoch2, train_loss2, 'black', linewidth = 1, linestyle='--', label='ResNet-50 Train loss')
    #lns13 = ax1.plot(total_epoch3, train_loss3, 'black', linewidth = 1, linestyle='dotted', label='VisionTransformer Train loss')
    ax1.grid(False)

    ax2 = ax1.twinx()
    # color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(30, 100)
    lns21 = ax2.plot(total_epoch1, Val_Acc1, '#1e4fa7', linewidth = 1, label='ProteoNet Val acc')
    lns22 = ax2.plot(total_epoch2, Val_Acc2, '#1e4fa7', linewidth = 1, linestyle='--', label='ResNet-50 Val acc')
    #lns23 = ax2.plot(total_epoch3, Val_Acc3, '#1e4fa7', linewidth = 1, linestyle='dotted', label='VisionTransformer Val acc')
    lns = lns11 + lns12 +  lns21 + lns22
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)
    fig.tight_layout()
    plt.savefig(pic_dir, dpi=300)
    plt.close("all")

def plot_part_num():
    x = [3, 5, 7]
    y = [92.16, 95.14, 94.63]
    fig, ax1 = plt.subplots()
    # color = 'tab:black'
    ax1.set_xlabel('Number of Partition')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(90, 96)
    lns11 = ax1.plot(x, y, 'black', linewidth = 1)
    ax1.grid(False)
    ticks = ax1.set_xticks(x)

    fig.tight_layout()
    plt.savefig("part_num", dpi=300)
    plt.close("all")


def plot_MBV2():
    MB = [91.716, 90.7, 89.946, 90.046]
    PMB = [92.216, 92.054, 90.716, 91.024]
    bar_colors = ['#1e4fa7', '#8e99d0']
    species = ["Accuracy", "Mean Percision", "Mean Recall", "Mean F1 Score"]
    penguin_means = {
        'MobileNetV2': MB,
        'ProteoNet-MobileNetV2': PMB,
    }
    width = 0.25  # the width of the bars
    multiplier = 0
    x = np.arange(len(species))  # the label locations
    fig, ax = plt.subplots(layout='constrained')
    for i, (attribute, measurement) in enumerate(penguin_means.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=bar_colors[i])
        # ax.bar_label(rects, padding=3)
        multiplier += 1


    # color = 'tab:black'
    ax.set_ylabel('Percent (%)')
    ax.set_ylim(80, 95)
    ticks = ax.set_xticks(x + width, species)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig("differ_backbone", dpi=300)
    plt.close("all")

if __name__ == '__main__':
    main()
'''
    plot_acc_loss(
        #'/home/huangjinze/code/MassSpectrumClsV1/logs/WeightedResPartNet_FL/2023-12-09-22-48-08fold0/metrics_outputs.csv',
        #'/home/huangjinze/code/MassSpectrumClsV1/logs/ResPartNet/2023-12-10-09-22-36fold0/metrics_outputs.csv',
        'F:\logs\ProteoNet\liver/tra/fold4/metrics_outputs.csv',
        'F:\logs\Resnet50\liver/tra/fold0/metrics_outputs.csv',
       # 'F:\logs/vision-transformer\liver/tra/fold0/metrics_outputs.csv',
        pic_dir="F:\logs\png\liver/"
    )
'''
    #plot_part_num()
    #plot_confusion_matrix()
    #plot_MBV2()