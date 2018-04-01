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
    mode = "NER"

    if mode == "NER":
        filename = "result/test_ner_bilstm_cnn4277.txt"
    else:
        filename = "result/test_pos_bilstm_cnn9000.txt"
    pos_mistake_dict = defaultdict(lambda: 0)
    y_true = []
    y_pred = []
    with open(filename, "r") as f:

        for line in f.readlines():
            if (len(line) > 5):
                pos_mistake_dict["->".join(line.strip().split(" ")[1:])] += 1
                y_true += [line.strip().split(" ")[2]]
                y_pred += [line.strip().split(" ")[1]]

    labels = []
    sizes = []
    sorted_pos_mistake_dict = sorted(pos_mistake_dict.items(), key=operator.itemgetter(1))
    for tuple in sorted_pos_mistake_dict:

        if tuple[0] != "" and tuple[0].split("->")[0] != tuple[0].split("->")[1] and pos_mistake_dict[tuple[0]] > (
                23 if mode == "NER" else 250):
            print(str(tuple) + " " + str(tuple[0].split("->")[0]) + " " + str(tuple[0].split("->")[1]))
            labels += [tuple[0]]
            sizes += [pos_mistake_dict[tuple[0]]]
            print(str(tuple[0]) + "," + str(pos_mistake_dict[tuple[0]]))

    plt.pie(sizes, labels=labels,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title("NER tagging mistake: (predicted tag -> ground truth tag)")
    plt.savefig("result/" + mode + "_pie_chart.png")
    plt.clf()


if __name__ == '__main__':
    main()
