"""
NAT - Assignment2
Luca G.McArthur s14422321
Gabriel Hoogervorst s1505156

This script defines the benchmark functions that will serve as fitness functions
for WOA.

"""
from sklearn.metrics import accuracy_score
import sys
sys.path.append(r'C:\Users\A\Desktop\故障诊断\所有程序')
import lssvm
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
dataSetfile2 = r"C:\Users\A\Desktop\初稿\特征1\gyTZ2.mat"
dataSe = sio.loadmat(dataSetfile2)
dataSet = dataSe['newTZ']
     
X = dataSet[:,1:]
y = dataSet[:,0]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

def get_function_details(func_name):

    if func_name == 'F1':
        lower_b = -100
        upper_b = 100
        dim = 30
        bench_f = Bench_Function('F1')

    elif func_name == 'F3':
        lower_b = -100
        upper_b = 100
        dim = 30
        bench_f = Bench_Function('F3')

    elif func_name == 'F10':
        lower_b = -32
        upper_b = 32
        dim = 30
        bench_f = Bench_Function('F10')

    elif func_name == 'F22':
        lower_b = 0
        upper_b = 10
        dim = 4
        bench_f = Bench_Function('F22')
    elif func_name == 'func':
        lower_b = 0.1
        upper_b = 15
        dim = 2
        bench_f = Bench_Function('func')

    return lower_b, upper_b, dim, bench_f


def get_function_details_2D(func_name):

    if func_name == 'F1':
        lower_b = -100
        upper_b = 100
        dim = 2
        bench_f = Bench_Function('F1')

    elif func_name == 'F3':
        lower_b = -100
        upper_b = 100
        dim = 2
        bench_f = Bench_Function('F3')

    elif func_name == 'F10':
        lower_b = -32
        upper_b = 32
        dim = 2
        bench_f = Bench_Function('F10')

    elif func_name == 'F22':
        lower_b = -32
        upper_b = 32
        dim = 2
        bench_f = Bench_Function('F22')
    elif func_name == 'func':
        lower_b = 0.1
        upper_b = 15
        dim = 2
        bench_f = Bench_Function('func')

    return lower_b, upper_b, dim, bench_f

class Bench_Function:

    def __init__(self, func_name):
        self.func_name = func_name

    def get_fitness(self, x):
        if self.func_name == 'F1':
            fitness = np.sum(x**2, 0)

        elif self.func_name == 'F3':
            dim = x.shape[0]
            fitness = 0
            for i in range(dim+1):
                fitness += np.sum(np.transpose(x).ravel()[:i])**2

        elif self.func_name == 'F10':
            dim = x.shape[0]
            fitness = -20*np.exp(-0.2*np.sqrt(np.sum(x**2, 0)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x), 0)/dim) + 20 + np.exp(1)
        elif self.func_name == 'func':
            clf = lssvm.LSSVM(gamma =x[0], kernel='rbf', sigma = x[1])
            clf.fit(X_train, y_train) 
            pred = clf.predict(X)
            acc = np.sum(pred == y)/len(y)
            #acc = accuracy_score(y_true = y, y_pred = pred)
            fitness = 1-acc

        elif self.func_name == 'F22':
            a_sh = np.array([[4, 4, 4, 4],
                             [1, 1, 1, 1],
                             [8, 8, 8, 8],
                             [6, 6, 6, 6],
                             [3, 7, 3, 7],
                             [2, 9, 2, 9],
                             [5, 5, 3, 3],
                             [8, 1, 8, 1],
                             [6, 2, 6, 2],
                             [7, 3.6, 7, 3.6]])
            c_sh = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            fitness = 0
            for i in range(7):
                fitness -= (np.dot((x-a_sh[i,:]), np.transpose((x-a_sh[i,:]))) + c_sh[i])**(-1)


        return fitness
