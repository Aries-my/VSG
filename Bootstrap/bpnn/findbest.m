function [best, pos]= findbest(model, option, trainPerf, valPerf, testPerf)
%FINDBEST Find best model from all models.
%   [best, bestloss, bestpos] = FINDBEST(model, option, trainerr, ...
%           valerr, testerr) Return a model best performance. 

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

[~, pos] = min(trainPerf);
if option.validation
    [~, pos] = min(valPerf);
end
if option.testing
    [~, pos] = min(testPerf);
end
best = model{pos};
















