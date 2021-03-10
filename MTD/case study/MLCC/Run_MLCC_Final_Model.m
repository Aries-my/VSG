traindata = load('mlcc_train');
testdata = load('mlcc_test');
virtualdata = load('mlcc_virtual');
%%%%%%%%%%%%%%%%%%%%%% Setting Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nh = 14;
mc = 0.95;
lr = 0.45;
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
if size(traindata, 2) ~=  size(virtualdata, 2) % for mtdvsg
    alldata = [traindata; testdata];
    x = alldata(:, 1 : end - 1)';
    y = alldata(:, end)';
    ex = extendattribute(x);
    alldata = [ex; y]';
    
    numTrain = size(traindata, 1);
    numTest =  size(testdata, 1);
    trainInd = 1 : numTrain;
    testInd = numTrain + (1 : numTest);
    traindata = alldata(trainInd, :);
    testdata = alldata(testInd, :);
end

combinedata = [traindata; virtualdata];
data.train = combinedata;
data.test = testdata;

[net, tr] = experiment(modelName, modelParam, data);

if strcmp(modelName, 'bpnn')
    trainInp = data.train(:, 1 : end - 1)';
    trainTarg = data.train(:, end)';
    trainOut = nnpredict(net, trainInp);
    trainPerf = nneval(net, trainInp, trainTarg);
    
    testInp = data.test(:, 1 : end - 1)';
    testTarg = data.test(:, end)';
    testOut = nnpredict(net, testInp);
    testPerf = nneval(net, testInp, testTarg);
end