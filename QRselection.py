import statsmodels.formula.api as smf
import statsmodels.api as sm
#import statsmodels as sm
qr = sm.QuantReg()


quantiles = np.arange(.05, .96, .1)
def fit_model(q):
    res = qr.fit(q=q)
    return [q, res.params[0], res.params[1] + res.conf_int().loc[0]]

models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b1', 'b2', 'lb1', 'ub1'])

qr_ci = qr.conf_int().loc['y'].tolist()
qr = dict(a = qr.params[0],
           b1 = qr.params[1],
           b2 = qr.params[2],
           lb = qr_ci[0],
           ub = qr_ci[1])

print(models)
print(qr)


