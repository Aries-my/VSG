import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

noise = 1
length = 1

run_hyperopt_search = True

kernel = GPy.kern.RBF(input_dim=1, variance=noise, lengthscale=length)
gpr = GPy.models.GPRegression(X_train, y_train, kernel)
if run_hyperopt_search:
    gpr.optimize(messages=True)


# In[ ]:


ypred_gp_test, cov_test = gpr.predict(X_test)