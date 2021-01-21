import numpy as np
import copy

def function(x1,x2):
    "functional(x1,x2)"
    y = 1.335*(1.5*(1-x1))+np.exp(2*x1-1)*np.sin(3*np.pi*(x1-0.6)**2)+np.exp(3*(x2-0.5))*np.sin(4*np.pi*(x2-0.9)**2)
    
    return y


def function_3d(x1, x2, x3):

    y = 4 * (x1 - 0.5) * (x2 - 0.5) * np.sin(2 * np.pi * ((x2 ** 2 + x3 ** 2)**0.5))

    return y


def get_functional_test_data(file_name):
    "get functional test data"
    X=[]
    Y=[]
    with open(file_name,'r') as f:
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


def get_functional_train_data(file_name):
    "get functional train data"
    X = []
    Y = []
    with open(file_name,'r') as f:
        data = f.readlines()

    for line in data:
        x = []
        item = line.split()
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

def add_train(train, vir):
    t = []
    for i in range(len(train)):
        t.append(train[i])
    for i in range(len(vir)):
        t.append(vir[i])
    return t


def get_true_x_give_y(given_y=0.5, tolerance=0.01, num_realizations=10000000, *, X=None, y=None):
    if (X is None) or (y is None):
        X, y = gen_data_magical_sinus(num_realizations)

    data_points = np.concatenate((X, y), axis=1)
    true_x_give_y = data_points[((y.squeeze() > given_y - tolerance / 2) *
                                 (y.squeeze() < given_y + tolerance / 2)), :-1]
    return true_x_give_y


def gen_data_magical_sinus(n_instance):
    """
    Generate n_instance of samples from a modified sinus function, noted by
    mdf_sinus here.
    """

    def _randrange(n, vmin, vmax):
        """
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        """
        return (vmax - vmin) * np.random.rand(n) + vmin

    noisey = 0
    x_1 = _randrange(n_instance, 0, 1)
    x_2 = _randrange(n_instance, 0, 1)

    ex_1 = np.random.normal(0, 0.1, n_instance)
    ex_2 = np.random.normal(0, 1, n_instance)

    X = np.column_stack((x_1, x_2))
    y = _magical_sinus(x_1, x_2) + noisey * np.sin(ex_1 + ex_2)
    y = y.reshape((n_instance, 1))
    return X, y


def _magical_sinus(x, y):
    """
    Create a noise-free single-valued benchmarking function:
                 z = f(x, y)
    derived from sinus function. It feeds two variables and
    returns a single value for each given pair of inputs(x, y).
    """
    z =1.335*(1.6*(1-x))+np.exp(2*x-1)*np.sin(4*np.pi*(x-0.6)**2)+np.exp(3*(y-0.5))*np.sin(3*np.pi*(y-0.9)**2)
#    z = (1.3356 * (1.5 * (1 - x))
#         + (np.exp(2 * x - 1) * np.sin(3 * np.pi * (x - 0.6) ** 2))
#         + (np.exp(3 * (y - 0.5)) * np.sin(4 * np.pi * (y - 0.9) ** 2)))
    return z


def get_true_x_give_y_real(given_y=0.5, tolerance=0.01, num_realizations=10000000, *, X=None, y=None):
    if (X is None) or (y is None):
        data_points = np.genfromtxt(f"./data/hdpe/hdpe.data", delimiter=' ')
        X = data_points[0:135, 0:15]
        y = data_points[0:135, 15]

    #data_points = np.concatenate((X, y), axis=1)
    true_x_give_y = data_points[((y.squeeze() > given_y - tolerance / 2) *
                             (y.squeeze() < given_y + tolerance / 2)), :-1]
    return true_x_give_y

