traindata = load('..\..\datasets\hdpemi_train');
testdata = load('..\..\datasets\hdpemi_test');
virtualdata = load('..\..\datasets\hdpemi_virtual');
%%%%%%%%%%%%%%%%%%%%%% Setting Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nh = 14;
mc = 0.90;
lr = 0.65;
epoch = 2000;
goal = 1e-6;

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
end

%%%%%%%%%%%%%%%%%%% Trainning and Testing Constructed Model %%%%%%%%%%%%%%%
combinedata = [traindata; virtualdata];
data.train = combinedata;
data.test = testdata;

[net, tr] = experiment(modelName, modelParam, data);
trainInp = data.train(:, 1 : end - 1)';
trainTarg = data.train(:, end)';
testInp = data.test(:, 1 : end - 1)';
testTarg = data.test(:, end)';
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