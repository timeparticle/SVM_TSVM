import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = np.loadtxt(fileName)
    dataMat = fr[:,0:-1]
    labelMat = fr[:,-1]
    print(fr.shape)
    return dataMat, labelMat

def loadDataSet1(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    print(dataMat)
    return dataMat, labelMat

# def show_accuracy(y_hat,y_train,str):
#     pass
# def iris_type(s):
#     it = {b'Iris-setosa': 1, b'Iris-versicolor': -1, b'Iris-virginica': 0}
#     return it[s]
# path = 'iris.data'  # 数据文件路径
# data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})

data = np.loadtxt('iris12.txt')
data = np.array([[x[0], x[3], x[4]] for x in data])
np.savetxt('testdata1.txt',data[:,:],fmt='%.2f',delimiter='\t', newline='\n', header='')
# dataMat, labelMat = loadDataSet('trdata.txt')
# print(dataMat,np.numarray(dataMat))
# dataMat1, labelMat1 = loadDataSet('iris12.txt')
# print(dataMat1.shape)