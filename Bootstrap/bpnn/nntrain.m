function [net, tr] = nntrain(net, input, target)
%NNTRAIN Train a network.
%   [net, tr] = TRAINNET(net, inputs, targets) Train network using
%           training set and test the trained BPNN using testing set. All
%           parameters, such as learn ratio, momentum, activation function,
%   Inputs:
%           net - A neural network.
%           inputs - Inputs of network.
%           targets - Targets assosited with inputs.
%   output:
%           net - The trained neural network.
%   Example:
%
%           [net, tr] = trainnetwork(net, input, target)
%

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS

[net, data] = nnconfigure(net, input, target);
[net, tr, param.pdW, param.pdb, options] = nnprepare(net);

PS = data.PS;
TS = data.TS;

trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

trainP  = data.P(:, trainInd); % Training samples.
trainT = data.T(:, trainInd);
valP    = data.P(:, valInd);   % Validation samples.
valT   = data.T(:, valInd);
testP   = data.P(:, testInd);  % Testing samples.
testT  = data.T(:,testInd);

validation = true;
testing = true;
if isempty(valInd)
    validation = false;
end
if isempty(testInd)
    testing = false;
end
options.validation = validation;
options.testing = testing;

efig = []; % Plot errors.
if net.showWindow
    efig = figure();
end

sfig = []; % Plot gradients and number of validation error increases
if net.showState
    sfig = figure();
end

n = size(trainP, 2); % Number of training samples.
if n > 1
    batch = fix(log2(n));
else
    batch = 1;
end
batchSize = fix(n / batch);
assert(rem(batch, 1) == 0, 'Size of batch must be an integer');
epochs = net.epoch;

trainPerf = [];
valPerf = [];
testPerf = [];

k = 1;
for i = 1 : epochs
    tic;
    ind = randperm(n);
    for j = 1 : batch
        P = trainP(:, ind((j - 1) * batchSize + 1 : j * batchSize));
        T = trainT(:, ind((j - 1) * batchSize + 1 : j * batchSize));
        
        [loss, outLayer, E]= feedforward(net, P, T);
        [net, gradient, param] = backpropagation (net, P, E, outLayer, param);
        
        los(k) = loss; % Loss for a single sample at each epoch.
        grad(k) = gradient; % Gradient for a single sample at each epoch.
        
        k = k + 1;
    end %  for j = 1 : batch
    
    cpuTime(i) = toc; % Time interval between epochs.
    avgLoss(i) = mean(los((k - batch) : (k - 1))); % Average loss of mini-batch at each epoch
    avgGradient(i) = mean(grad((k - batch) : (k - 1)));  % Average gradient of mini-batch at each epoch.
    
    trainPerf(i) = evaluate(net, trainP, trainT);
    perfStr = sprintf('; Full-batch training MSE = %3.6f', trainPerf(i));
    if validation
        valPerf(i) = evaluate(net, valP, valT);
        perfStr = sprintf([perfStr, ', validation MSE = %3.6f'],  valPerf(i));
    end
    if testing
        testPerf(i) = evaluate(net, testP, testT);
        perfStr = sprintf([perfStr, ', testing MSE = %3.6f'], testPerf(i));
    end
    
    % Stop criteria
    stop = false; 
    flag = 'maxiteration';
    fail = [];
    if validation
        [stop, flag, fail(i)] = stopcriteria(net, i, options.criteria, avgGradient, trainPerf, valPerf);
    end
    
    % Disp run information at each epoch.
    if net.showCommandLine
        showMessage = ['epoch %d / %d. Took %3.6f seconds. Mini-batch average gradient = %3.6f, Mini-batch average loss = %3.6f' perfStr '\n'];
        fprintf(1, showMessage, i, epochs, cpuTime(i), avgGradient(i), avgLoss(i));
    end
    
    if net.showWindow
        efig = updatefigure(net, i, efig, options, trainPerf, valPerf, testPerf);
    end
    
    if net.showState
        sfig = updatefigure(net, i, sfig, options, trainPerf, valPerf, testPerf , avgGradient, fail);
    end
    
    param.pdW = param.dW;
    param.pdb = param.db;
    
    % Store all models.
    model{i} = net;
    
    % Trigger stop criteria
    if stop
        break;
    end
end %  i = 1 : epochs

% Find best model with best performance.;
[net, pos]= findbest(model, options, trainPerf, valPerf, testPerf);


tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;

tr.cpuTime = cpuTime;
tr.allModel = model;
tr.stopFlag = flag;

tr.epoch = i;
tr.bestEpoch = pos;
tr.fail = fail;
tr.gradient = avgGradient;
tr.batch = batch;
tr.batchSize = batchSize;
tr.perf.train = trainPerf;
tr.bestPerf.train = trainPerf(pos);
if options.validation
    tr.perf.val = valPerf;
    tr.bestPerf.val = valPerf(pos);
end
if options.testing
    tr.perf.test = testPerf;
    tr.bestPerf.test = testPerf(pos);
end
save('bpmodel.mat', 'net', 'PS', 'TS');


function perf = evaluate(net, input, target)
%EVALUATE Evaluate net's performance.
%   inputs:
%       net - the trained neural network.
%       target - normalization target.
%       output - normalization output.
%   outputs:
%       perf - network's performance.

global TS

[~, outLayer] = feedforward(net, input, target);
Nl = net.numLayer;
output = outLayer{Nl};

% Reversed normalization for target and output.
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

