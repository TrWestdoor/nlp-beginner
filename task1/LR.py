import numpy as np


def load_data():
    dataMat = []
    labelMat = []
    fr = open('train.tsv')
    for line in fr.readlines()[1:]:
        line = line.strip().split()
        dataMat.append([int(line[0]), int(line[1]), ' '.join(line[2:-1])])
        labelMat.append(int(line[-1]))
    return dataMat, labelMat

