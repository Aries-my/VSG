function error = sumup(target, output)
%SUMUP Compute and summary errors such as MSE, RMSE, and so on.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


assert(isequal(size(target, 2), size(output, 2)), 'Dimension of target and output dismatch')

[M, N] = size(target);

err = target - output;
mu = sum(sum(err)) / (M*N);
SSE = sum(sum((err) .^ 2));
MSE = SSE / (M*N);
RMSE = sqrt(MSE);
MAE = sum(sum(abs(err))) / (M*N);
MAPE = (100 * sum(sum(abs(err ./ target)))) / (M*N);
STD = sqrt(sum(sum(err - mu) .^ 2) / (M*N));

error.SSE = SSE;
error.MSE = MSE;
error.RMSE = RMSE;
error.MAE = MAE;
error.MAPE = MAPE;
error.STD = STD;



