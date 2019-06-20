import numpy as np
import os


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
        for element in line:
            if element not in Dict and element not in [',', '.']:
                Dict.append(element)
    return Dict


def create_vector(dict_, dataMat):
    m = len(dataMat)
    n = len(dict_)
    dataMatrix = np.zeros((m, n))
    for i in range(m):
        line = dataMat[i][2].strip().split()
        for word in line:
            if word not in [',', '.']:
                dataMatrix[i][dict_.index(word)] = 1
    return dataMatrix


if __name__ == '__main__':
    dataMat, labelMat = load_data()
    if os.path.exists('dict.txt'):
        fr = open('dict.txt', 'r')
        Dict = fr.readline().strip().split()
    else:
        Dict = create_dic(dataMat)
        fw = open('dict.txt', 'w')
        fw.write(' '.join(Dict))

    dataMatrix = create_vector(Dict, dataMat)
    print(np.shape(dataMatrix))
