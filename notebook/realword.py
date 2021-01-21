# %%

import numpy as np
import importlib
import dataset
import config, plotting, sample, SampleCharacter, XLim, QRselection
import QrModels, Point
import copy
import RelativeImportance
import Measure
import Generation
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor

importlib.reload(config)
importlib.reload(dataset)
importlib.reload(plotting)
importlib.reload(sample)
importlib.reload(SampleCharacter)
importlib.reload(XLim)
importlib.reload(QRselection)
importlib.reload(QrModels)
importlib.reload(Point)
importlib.reload(RelativeImportance)
importlib.reload(Measure)
importlib.reload(Generation)

# %%

import os

dataset_config = config.DatasetConfig(scenario="MLCC", n_instance=1000)

assert (dataset_config.scenario == "PG"
        or dataset_config.scenario == "PT"
        or dataset_config.scenario == "MLCC"
        )
fig_dir = f"../figures/{dataset_config.scenario}"

try:
    os.mkdir(fig_dir)
    print(f"Directory {fig_dir} created ")
except FileExistsError:
    print(f"Directory {fig_dir} already exists replacing files in this notebook")

# %%

file_name_test = "../data/" + dataset_config.scenario + "/test_data.txt"
X_test, Y_test = dataset.get_functional_test_data(file_name_test)

# %%

file_name_train = "../data/" + dataset_config.scenario + "/train_data.txt"
X_train, Y_train = dataset.get_functional_train_data(file_name_train)

# %%

random_seed = 1985
if dataset_config.scenario == "PG":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.0001, lr_disc=0.001, dec_gen=0, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

elif dataset_config.scenario == "PT":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.0001, lr_disc=0.0005, dec_gen=0, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )
elif dataset_config.scenario == "MLCC":
    exp_config = config.Config(
        model=config.ModelConfig(activation="elu", lr_gen=0.0001, lr_disc=0.0005, dec_gen=0, dec_disc=0,
                                 optim_gen="Adam", optim_disc="Adam", z_input_size=1, random_seed=random_seed),
        training=config.TrainingConfig(n_epochs=500, batch_size=100, n_samples=50),
        dataset=dataset_config,
        run=config.RunConfig(save_fig=1)
    )

# %%

coef = sample.get_sta_reg_cov(X_train, Y_train)

# %%

if dataset_config.scenario == "PG":
    imp = RelativeImportance.relativeImp_PG()
elif dataset_config.scenario == "PT":
    imp = RelativeImportance.relativeImp_PT()
# elif dataset_config.scenario == "MLCC":
# imp = RelativeImportance.relativeImp_MLCC()
imp = [0.005807103724392776, 0.08056379800289575, 0.027708178275372558, 0.029279516948187895, 0.06921072266530037,
       0.1222271387246593, 0.11319916571268794, 0.04157214482892202, 0.12625181040975142, 0.06501790211941245,
       0.3118271249994784, 0.0073353935889390945]
# imp = list(imp.values())
print("The importance for every dimension:")
print(imp)

# %%

length, max_dist = sample.get_sample_length(X_train, imp)

# %%

x_min = np.amin(X_train, axis=0)
x_max = np.amax(X_train, axis=0)
L = sample.get_x_len(x_min, x_max)

# %%

n_sample, length = sample.divide_sample(length, L, 180)

# %%

dim = len(X_train[0])
gen_x = sample.gen_x_center(dim, length, n_sample, x_min)

# %%

gen_sample_point = sample.gen_product2(gen_x)
print("样方中心点：")
gen_sample_point = np.array(gen_sample_point)
# print(gen_sample_point)

# %%

xlimit = []
for index in range(len(n_sample)):
    l = []
    for i in range(n_sample[index]):
        x = x_min[index] + i * length[index]
        l.append(x)
    l.append(x_min[index] + n_sample[index] * length[index])
    xlimit.append(l)
print("over")

# %%

sample_list = []
xlim_list = []

# %%

x_value = []
for index in range(dim):
    xl = []
    for i in range(len(X_train)):
        x = X_train[i][index]
        r = 0
        for xi in xl:
            if xi == x:
                r = 1
        if r == 0:
            xl.append(x)
    x_value.append(xl)

x_value_ori = copy.deepcopy(x_value)
print("over")

# %%

XLim.con_s(gen_sample_point, sample_list, dim, xlimit)
print("over")

# %%

XLim.con_sample(xlim_list, length, x_min, dim, n_sample)
print("over")

# %%

XLim.sample_feature(xlim_list, sample_list, x_value)
print("over")

# %%

XLim.add_xvalue(xlim_list)
# os.exist(0)

# %%

x_com, x_add = Generation.get_x_com(X_train, dim, x_value, x_value_ori)

# %%

E_dist = Generation.E_dist(x_com)

# %%

gen_x = []
discard_list = []
Generation.gen_x_sample(sample_list, X_train, gen_x, max_dist, dim, discard_list, x_com, x_max, x_min, E_dist)
# gen_x_cross = sample.gen_product(x_value)
print("over")

# %%

# f_list = []
# sample.cross_point_del(gen_x_cross, X_train)
# sample.point_filiter(gen_x_cross, X_train, max_dist, x_value, x_value_ori, dim, f_list)
# print("over")

# %%

plot_xlim = copy.deepcopy(xlimit)
for index in range(dim):
    i = len(plot_xlim[index])
    plot_xlim[index][i - 1] = x_max[index]
print("over")

# %%

gen_x_checked = []
XLim.check2(sample_list, xlim_list, gen_x_checked, discard_list, gen_x)
XLim.sample_attri(sample_list, X_train, gen_x_checked, Y_train)
XLim.xl_attri(xlim_list, X_train, gen_x_checked)
gen_x = np.array(gen_x)
# plotting.plot_genx(X_train, np.array(gen_x_checked), length, n_sample, exp_config, fig_dir, plot_xlim, "gen_x_checked.png")


# %%

import GPy

noise = 0.01
length_ = 0.1

run_hyperopt_search = True

kernel = GPy.kern.RBF(input_dim=3, variance=noise, lengthscale=length_)
gpr = GPy.models.GPRegression(X_train, Y_train.reshape(-1, 1), kernel)
if run_hyperopt_search:
    gpr.optimize(messages=True)
print("over")

# %%

gen_y_cross, cov_train_cross = gpr.predict(np.array(gen_x_checked))
print("over")

# %%

XLim.add_y(sample_list, gpr)
print("over")

# %%

point_list = []
Point.con_point(gen_x_checked, gen_y_cross, point_list)
print("over")

# %%

for sample in sample_list:
    for index in range(len(sample.gen_xlist)):
        for point in point_list:
            r = -1
            for i in range(dim):
                if point.x[i] != sample.gen_xlist[index][i]:
                    r = 0
                    break
            if r == -1:
                sample.points.append(point)
print("over")

# %%

import statsmodels.api as sm

qrX = X_train
qrX = sm.add_constant(qrX[0:])
qr = sm.QuantReg(Y_train.reshape(-1, 1), qrX)
res = qr.fit(q=.2)
print(res.summary())

# %%

quantiles = np.arange(.05, .96, .1)
quantiles = np.around(quantiles, decimals=3)


def fit_model(q):
    res = qr.fit(q=q)
    return q, np.around(res.params, decimals=4)


models = []

for x in quantiles:
    q, param = fit_model(x)
    model = QrModels.QrModels(q, param[0], param[1:])
    models.append(model)

for model in models:
    print(str(model.q) + '\t' + str(model.a) + '\t' + str(model.param))

ols = sm.OLS(Y_train.reshape(-1, 1), qrX).fit()

for ol in ols.params:
    print(str(ol))
print("over")

# %%

if dataset_config.scenario == "MLCC":
    y_quantile = [15676.750, 16559.600, 16657.000, 17030.200, 17143.350, 17286.050, 17342.350, 17843.250, 18193.950,
                  18403.250]

# %%

vir_xpoint = []
vir_ypoint = []

for xv in x_value:
    xv.sort()
for xv in x_value_ori:
    xv.sort()
print("over")

# %%

QRselection.qr_selection(xlim_list, models, vir_xpoint, vir_ypoint, y_quantile, ols, x_value_ori,
                         n_sample, X_train, Y_train, sample_list, point_list, x_value)
print("over")

# %%

i = 0
for point in point_list:
    if point.checked == 1:
        i += 1
print("over")

# %%

from hpelm import ELM

elm = ELM(X_train, Y_train.reshape(-1, 1))
elm.add_neurons(20, "sigm")
elm.add_neurons(10, "rbf_l2")
elm.train(X_train, Y_train, "LOO")
y_predict_output = elm.predict(X_test)

test_list = []
for i in range(len(X_test)):
    point = Point.Point(X_test[i], y_predict_output[i])
    point.true = Y_test[i]
    point.erro = Y_test[i] - point.y
print("over")

# %%

plotting.plot_erro(test_list, exp_config, fig_dir, "erro")

# %%

mae = Measure.mae1(test_list)
mse = Measure.mse1(test_list)
mape = Measure.mape1(test_list)
print(" all test point")
print(mae)
print(mse)
print(mape)

# %%


np.savetxt("../figures/MLCC/vx_data.txt", np.array(vir_xpoint), fmt='%.8f', delimiter=' ')

# %%

np.savetxt("../figures/MLCC/vy_data.txt", np.array(vir_ypoint), fmt='%.8f', delimiter=' ')




