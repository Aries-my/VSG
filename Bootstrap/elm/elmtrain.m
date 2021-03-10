function [net, tr] = elmtrain(net, input, target)
%ELMTRAIN Train a single-hidden layer feedforward neural network using
%           extreme learning algorithm (ELM).

%   Ref. G.B. Huang, Q.Y. Zhu, C.K. Siew,
%       Extreme learning machine: Theory and applications,
%       Neurocomputing, 70 (2006) 489-501
%

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS

[net, data] = elmconfigure(net, input, target);
[net, tr] = elmprepare(net);

trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

% Training samples.
trainP  = data.P(:, trainInd);
trainT = data.T(:, trainInd);

% Validation samples.
valP    = data.P(:, valInd);
valT   = data.T(:, valInd);

% Testing samples.
testP   = data.P(:, testInd);
testT  = data.T(:,testInd);

validation = true;
if isempty(valInd)
    validation = false;
end

testing = true;
if isempty(testInd)
    testing = false;
end

PS = data.PS;
TS = data.TS;

trainPerf = [];
valPerf = [];
testPerf = [];
trainTime = [];
valTime = [];
testTime = [];

net = elmcore(net, trainP, trainT);
% Training Phase.
tic
trainPerf = evaluate(net, trainP, trainT);
trainTime = toc;
% Validation Phase
if  validation
    tic
    valPerf = evaluate(net, valP, valT);
    valTime = toc;
end

% Testing Phase.
if  testing
    tic
    testPerf = evaluate(net, testP, testT);
    testTime = toc;
end

tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;
tr.cpuTime.train = trainTime;
tr.cpuTime.val = valTime;
tr.cpuTime.test = testTime;
tr.bestPerf.train = trainPerf;
tr.bestPerf.val = valPerf;
tr.bestPerf.test = testPerf;
save('elmmodel.mat', 'net', 'PS', 'TS');

function perf = evaluate(net, input, target)
%ELMEVAL Evaluate peformance of ELM using a give datasets and return
%   reversed normalization output and error index.

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global TS

output = elmfeedforward(net, input);

% Calculate error and reversed normalization output.
normalizationFcn = net.normalizationFcn;
target = postprocess(normalizationFcn, target, TS);
output = postprocess(normalizationFcn, output, TS);

performFcn = net.performFcn;
switch lower(performFcn)
    case {'mse'}
        perf = mse(target - output);
    case {'mae'}
        perf = mae(target - output);
    case {'sae'}
        perf = sae(target - output);
    case {'sse'}
        perf = sse(target - output);
end