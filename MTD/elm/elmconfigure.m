function [net, data]= elmconfigure(net, p, t)
%ELMCONFIGURE  Configure paramaters for ELM.

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


if ~isequal(size(p, 2), size(t, 2))
    error('ELM:elmconfigure:misMatch','Inputs data and targets data mismatch.');
end

net.numInput = size(p, 1);
net.numOutput = size(t, 1);
net.layer{end}.size = net.numOutput;

% Initialize weights and biases.
net = elminit(net);

% normalize inputs and targets.
normalizationFcn = net.normalizationFcn;
normalizationParam = net.normalizationParam;

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        [pn, ps] = mapminmax(p, normalizationParam);
        [tn, ts] = mapminmax(t, normalizationParam);
    case {'mapstd', 'std'}
        [pn, ps] = mapstd(p, normalizationParam);
        [tn, ts] = mapstd(t, normalizationParam);
end

data.p = p;
data.t = t;
data.P = pn;
data.T = tn;
data.PS = ps;
data.TS = ts;

N = size(p, 2);
% Dataset division. Return index of dataset for training, validation and testing.
[trainInd, valInd, testInd] = divideindex(net.divideFcn, N, net.divideParam);

data.trainInd = trainInd;
data.valInd = valInd;
data.testInd = testInd;


function  [trainInd, valInd, testInd] = divideindex(divideFcn, N, divideParam)
%   Return index of training set, validation set and testing set.  

switch lower(divideFcn)
    case 'dividerand'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideblock'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideint'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideind' % FIXME
        assert(~isa(struct2array(divideParam(:)), 'integer'), ' Error in specific index.');
        [trainInd,valInd,testInd] = divideind(N, divideParam);
end


