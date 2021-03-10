# =============神经网络用于回归=============
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
#---------------------------- Generate Data ----------------------------#
X_R, y_R = make_regression(n_samples=100, n_features=1, n_informative=1,
bias=150.0, noise=30, random_state = 0 )
X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_R[0::5],
y_R[0::5],random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(11,8), dpi=100)
#---------------------------- MLP Regression ---------------------------#
for dummy, MYactivation in zip(subaxes,['tanh', 'relu']):
    mlp_reg = MLPRegressor(hidden_layer_sizes=[100,100], activation=MYactivation, solver='lbfgs').fit(X_train, y_train)
    y_predict_output = mlp_reg.predict(X_predict_input)
    dummy.plot(X_predict_input, y_predict_output, '^', markersize=10)
    dummy.plot(X_train, y_train, 'o')
    dummy.set_title('MLP regression\n activation={})'.format(MYactivation))
    dummy.set_xlabel('Input feature')
    dummy.set_ylabel('Target value')
    dummy.legend(('Predicted', 'Actual'), loc='upper left', shadow=True)
plt.show()