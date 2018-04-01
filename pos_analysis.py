import pickle
from collections import defaultdict

import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt
import numpy as np

# plt.rcdefaults()


def main():
    pos_mistake_dict = defaultdict(lambda: 0)
    y_true = []
    y_pred = []
    with open("result/test_pos_bilstm_cnn9000.txt", "r") as f:

        for line in f.readlines():
            if (len(line) > 5):
                pos_mistake_dict["->".join(line.strip().split(" ")[1:])] += 1
                y_true += [line.strip().split(" ")[2]]
                y_pred += [line.strip().split(" ")[1]]

    # labels=list(set(y_pred))

    # precision_dict=dict(zip(labels,precision_score(y_true, y_pred,labels,average=None)))
    # for (key, value) in sorted(precision_dict.items(), key=operator.itemgetter(1), reverse=True):
    # print("%s & %s     \\\\ \\hline" % (key, value))

    labels = []
    sizes = []
    sorted_pos_mistake_dict = sorted(pos_mistake_dict.items(), key=operator.itemgetter(1))
    for tuple in sorted_pos_mistake_dict:

        if tuple[0] != "" and tuple[0].split("->")[0] != tuple[0].split("->")[1] and pos_mistake_dict[tuple[0]] > 250:
            print(str(tuple) + " " + str(tuple[0].split("->")[0]) + " " + str(tuple[0].split("->")[1]))
            labels += [tuple[0]]
            sizes += [pos_mistake_dict[tuple[0]]]
            print(str(tuple[0]) + "," + str(pos_mistake_dict[tuple[0]]))
    #
    # plt.pie(sizes, labels=labels,
    #         autopct='%1.1f%%', shadow=True, startangle=140)

    # plt.axis('equal')
    # plt.title("POS tagging mistake: (predicted tag -> ground truth tag)")
    # plt.savefig("result/pie_chart.png")
    # plt.clf()

    labels = []
    performance = []
    precision_dict = dict(zip(labels, precision_score(y_true, y_pred, labels, average=None)))
    for (key, value) in sorted(precision_dict.items(), key=operator.itemgetter(1), reverse=True):
        labels += [key]
        performance += [value]

    y_pos = np.arange(len(labels))

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('score')
    plt.title('Precision score by POS tag')
    plt.savefig("result/precision.png")
    # plt.show()




if __name__ == '__main__':
    main()
