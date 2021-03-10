function [net, tr, pdW, pdb, options] = nnprepare(net)
%NNPREPARE Prepare varibles pdW, pdB and loss to trainning process.
%   [pdW, pdB] = NNPREPARE(net) create pdW and pdB, whose size are same as
%           weights and biases.              

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

% Initilize previous delta of weights and biases.
Nl = net.numLayer;
for i = 1 : Nl
    pdW{i} = zeros(size(net.weight{i}));    % previous dW. 
    pdb{i} = zeros(size(net.bias{i}));      % previous db.
end


tr.trainInd = [];
tr.valInd = [];
tr.testInd = [];

tr.cpuTime = [];                % Time cost between two successive epochs.
tr.allModel =[];                % All models at each epoch.
tr.stopFlag = [];               % Stop criteria message.

tr.epoch = [];                  % Actual epoch.
tr.bestEpoch = [];              % Best epoch.
tr.fail = [];                   % Number of successive iteration of validation perfermance fails to increace.
tr.gradient = [];               % Gradient of output layer (if transFcn is logsig) or gradient of last hidden layer (if transFcn is purelin).
tr.batch = [];                  % Number of batchs;
tr.batchSize = [];              % Size of each batch.
tr.perf.train = [];             % Trainning performance 
tr.perf.val = [];               % Validation performance
tr.perf.test = [];              % Testing performance
tr.bestPerf.train = [];         % Trainning performance at best epoch.
tr.bestPerf.val = [];           % Validation performance at best epoch.
tr.bestPerf.test = [];          % Testing performance at best epoch.


% Stop criteria.
options.criteria.maxiteration = true;
options.criteria.goal = true;
options.criteria.mingrad = true;
options.criteria.maxfail = true;

% Status of validation and testing.
options.validation = [];
options.testing = [];











