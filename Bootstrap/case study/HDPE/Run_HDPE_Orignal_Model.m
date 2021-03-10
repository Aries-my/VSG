traindata = load('..\..\datasets\HDPE_train');
testdata = load('..\..\datasets\HDPE_test');

%%%%%%%%%%%%%%%%%%%%%% Setting Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nh = 18;
mc = 0.90;
lr = 0.65;
epoch = 2000;
goal = 1e-6;
interval = 18;

modelName = 'elmnn';
switch lower(modelName)
    case {'bp', 'bpnn'}
        modelParam.hiddenLayerSize = Nh;
        modelParam.momentCoefficient = mc;
        modelParam.learnRatio = lr;
        modelParam.maxIteration = epoch;
        modelParam.goal = goal;
    case {'elm', 'elmnn'}
        modelParam.hiddenLayerSize = Nh;
    case {'m5p', 'model trees'}
        modelParam.interval = interval;
end

%%%%%%%%%%%%%%%%%%% Trainning and Testing Constructed Model %%%%%%%%%%%%%%%
data.train = traindata;
data.test = testdata;

[net, tr] = experiment(modelName, modelParam, data);


trainInp = traindata(:, 1 : end - 1)';
trainTarg = traindata(:, end)';
testInp = testdata(:, 1 : end - 1)';
testTarg = testdata(:, end)';
if strcmp(modelName, 'bpnn')
    trainOut = nnpredict(net, trainInp);
    trainPerf = nneval(net, trainInp, trainTarg);
    
    testOut = nnpredict(net, testInp);
    testPerf = nneval(net, testInp, testTarg);
end
if strcmp(modelName, 'elmnn')
    trainOut = elmpredict(net, trainInp);
    trainPerf = elmeval(net, trainInp, trainTarg);
    
    testOut = elmpredict(net, testInp);
    testPerf = elmeval(net, testInp, testTarg);
end
error = [testPerf.MSE; testPerf.RMSE; testPerf.MAE];