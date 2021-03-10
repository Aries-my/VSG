function [error, output] = nneval(net, input, target)
%NNEVAL Compute loss on a given dataset.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS

% normalization for input and target.
normalizationFcn = net.normalizationFcn;
input = preprocess(normalizationFcn, input, PS);
target = preprocess(normalizationFcn, target, TS);

% Calculate normalization output.
[loss, outLayer] = feedforward(net, input, target);
Nl = net.numLayer;
output = outLayer{Nl};

% Calculate error and loss using reversed normalization output.
target = postprocess(normalizationFcn, target, TS);
output = postprocess(normalizationFcn, output, TS);

error = sumup(target, output);