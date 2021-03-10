function [net, tr] = experiment(modelName, modelParam, data)
%EXPERIMENT Conduct a expected model with particular model parameters on 
%       datasets.

%   Date: December 22, 2017
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)

traindata = data.train;
testdata = data.test;
alldata = [traindata; testdata];
input = alldata(:, 1 : end - 1)';
target = alldata(:, end)';
numTrain = size(traindata, 1);
numTest = size(testdata, 1);
trainInd = 1 : numTrain;
testInd = numTrain + (1 : numTest);

switch lower(modelName)
    case {'bp', 'bpnn'}
        
        hiddenLayerSize = modelParam.hiddenLayerSize;
        mc = modelParam.momentCoefficient;
        lr = modelParam.learnRatio;
        epoch = modelParam.maxIteration;
        goal = modelParam.goal;
        
        net = nncreate(hiddenLayerSize);
        net.lr = lr; % Training parameters
        net.mc = mc;
        net.divideFcn = 'divideind';  % Dataset division
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = [];
        net.divideParam.testInd = testInd;
        net.epoch = epoch;  % Stop criteria
        net.goal = goal;
        net.showWindow = false; % Display performance
        net.showState = false;
        net.showCommandLine = false;
        
        [net, tr] = nntrain(net, input, target);
        
    case {'elm', 'elmnn'}
        
        hiddenLayerSize = modelParam.hiddenLayerSize;
        net = elmcreate(hiddenLayerSize);
        net.divideFcn = 'divideind';  % Dataset division
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = [];
        net.divideParam.testInd = testInd;
        
        [net, tr] = elmtrain(net, input, target);
        
    case {'m5p', 'model trees'}

        M = modelParam.interval;
        net = m5pcreate(M);
        net.divideFcn = 'divideind';  % Dataset division
        net.divideParam.trainInd = trainInd;
        net.divideParam.valInd = [];
        net.divideParam.testInd = testInd;
        [net, tr] = m5ptrain(net, input, target);
end

trainBestPerf = tr.bestPerf.train;
valBestPerf = tr.bestPerf.val;
testBestPerf = tr.bestPerf.test;
trainInd = tr.trainInd;
valInd = tr.valInd;
testInd = tr.testInd;
trainInp = input(:, trainInd);
trainTarg = target(:, trainInd);
valInp = input(:, valInd);
valTarg = target(:, valInd);
testInp = input(:, testInd);
testTarg = target(:, testInd);

switch lower(modelName)
    case {'bp', 'bpnn'}
        trainOut = nnpredict(net, trainInp);
        valOut = nnpredict(net, valInp);
        testOut = nnpredict(net, testInp);
    case {'elm', 'elmnn'}
        trainOut = elmpredict(net, trainInp);
        valOut = elmpredict(net, valInp);
        testOut = elmpredict(net, testInp);
    case {'m5p', 'm5pnn'}
        trainOut = tr.output.train; 
        valOut = tr.output.val; 
        testOut = tr.output.test; 
end

display = true;
if display
    figure
    index = 1 : length(trainInd);
    plot(index, trainTarg, 'b--o', index, trainOut, 'b-.+')
    xlabel('Index of training samples')
    ylabel('Output')
    legend('Actual value', 'Predictive value')
    title(['Training perforance = ' num2str(trainBestPerf)])
    
    if ~isempty(valInd)
        figure
        index = 1 : length(valInd);
        plot(index, valTarg, 'r:*', index, valOut, 'r-.s')
        xlabel('Index of testing samples')
        ylabel('Output')
        legend('Actual value', 'Predict value')
        title(['Testing perforance = ' num2str(valBestPerf)])
    end
    
    if ~isempty(testInd)
        figure
        index = 1 : length(testInd);
        plot(index, testTarg, 'r:*', index, testOut, 'r-.s')
        xlabel('Index of testing samples')
        ylabel('Output')
        legend('Actual value', 'Predict value')
        title(['Testing perforance = ' num2str(testBestPerf)])
    end
end