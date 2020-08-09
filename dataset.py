import numpy as np

def function(x1,x2):
    "functional(x1,x2)"
    y = 1.335*(1.5*(1-x1))+np.exp(2*x1-1)*np.sin(3*np.pi*(x1-0.6)**2)+np.exp(3*(x2-0.5))*np.sin(4*np.pi*(x2-0.9)**2)
    
    return y


def get_functional_test_data():
    "get functional test data"
    X=[]
    Y=[]
    with open("../data/functional/test_data.txt",'r') as f:
        data = f.readlines()

    for line in data:
        x = []
        item = line.split()
        #print(item)
        for index in range(len(item)-1):
            num = float(item[index])
            x.append(num)
        X.append(x)
        y = float(item[-1])
        Y.append(y)

    X_test = np.array(X)
    Y_test = np.array(Y)
    return X_test, Y_test


def get_functional_train_data():
    "get functional train data"
    X=[]
    Y=[]
    with open("../data/functional/train_data.txt",'r') as f:
        data = f.readlines()

    for line in data:
        print(line)
        x = []
        item = line.split()
        #print(item)
        for index in range(len(item)-1):
            num = float(item[index])
            x.append(num)
        X.append(x)
        y = float(item[-1])
        Y.append(y)

    X_train = np.array(X)
    Y_train = np.array(Y)
    return X_train,Y_train

def get_dataset(n_instance=1000, scenario="functional", seed=1):
    if scenario == "functional":
        X_train, Y_train = get_functional_train_data()
        X_test, Y_test = get_functional_test_data()
    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, Y_train, X_test, Y_test