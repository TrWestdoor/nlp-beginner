import numpy as np
test


def load_data():
    dataMat = []
    labelMat = []
    fr = open('train.tsv')
    for line in fr.readlines()[1:]:
        line = line.strip().split()
        dataMat.append([int(line[0]), int(line[1]), ' '.join(line[2:-1])])
        labelMat.append(int(line[-1]))
    return dataMat, labelMat


def create_dic(dataMat):
    Dict = []
    for line in dataMat:
        line = line[2].split()
        Dict += line
    Dict = set(Dict)
    return Dict


dataMat, labelMat = load_data()
Dict = create_dic(dataMat)
print(Dict)
