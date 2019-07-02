import numpy as np
import os


def load_data(file_name):
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines()[1:]:
        line = line.strip().split()
        dataMat.append([line[0], line[1], ' '.join(line[2:-1])])
        labelMat.append(line[-1])
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
            if word not in dict_:
                continue
            if word not in [',', '.']:
                dataMatrix[i][dict_.index(word)] = 1
    return dataMatrix


def stand_regresssion(xMat, yMat):
    yMat = yMat.T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def softmax(x_vector):
    exps = np.exp(x_vector)
    return exps / np.sum(exps)


def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    p = softmax(X)
    #log_likelihood = -np.log(p[range(m),y])
    loss = p - y
    print(loss.shape)
    #loss = np.sum(log_likelihood) / m
    return loss


def gradAscent(dataMat, classLabels):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(classLabels)
    m, n = np.shape(dataMatrix)
    # print(n, m)
    alpha = 0.01
    maxCycles = 2000
    weights = np.ones((n, 5))

    for i in range(maxCycles):
        error = cross_entropy(dataMatrix*weights, labelMatrix)
        weights = weights - alpha * dataMatrix.transpose() * error
        print(dataMatrix * error)
        print(np.dot(dataMatrix, error))


if __name__ == '__main__':
    import time
    dataMat, labelMat = load_data('train.tsv')
    labelMat = [int(x) for x in labelMat]
    y = np.zeros((len(dataMat), 5))
    for i in range(len(dataMat)):
        y[i][labelMat[i]-1] = 1
    testMat, _ = load_data('test.tsv')
    if os.path.exists('dict.txt'):
        fr = open('dict.txt', 'r')
        Dict = fr.readline().strip().split()
    else:
        Dict = create_dic(dataMat)
        fw = open('dict.txt', 'w')
        fw.write(' '.join(Dict))

    dataMatrix = create_vector(Dict, dataMat)
    testMatrix = create_vector(Dict, testMat)
    gradAscent(dataMatrix, y)



'''
    ws = stand_regresssion(np.mat(dataMatrix), np.mat(labelMat))
    y_pre = testMatrix * ws
    print(y_pre)
    with open('submission.csv', 'w') as f:
        i = 0
        f.write('PhraseId,Sentiment\n')
        for num in [_[0] for _ in testMat]:
            f.write(str(num) + ',' + str(y_pre[i][0]) + '\n')
            i += 1
'''


