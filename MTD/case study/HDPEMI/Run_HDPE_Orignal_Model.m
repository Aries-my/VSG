traindata = load('hdpemi_train');
testdata = load('hdpemi_test');

%%%%%%%%%%%%%%%%%%%%%% Setting Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nh = 14;
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

if strcmp(modelName, 'bpnn')
    trainInp = traindata(:, 1 : end - 1)';
    trainTarg = traindata(:, end)';
    trainOut = nnpredict(net, trainInp);
    trainPerf = nneval(net, trainInp, trainTarg);
    
    testInp = testdata(:, 1 : end - 1)';
    testTarg = testdata(:, end)';
    testOut = nnpredict(net, testInp);
    testPerf = nneval(net, testInp, testTarg);
end