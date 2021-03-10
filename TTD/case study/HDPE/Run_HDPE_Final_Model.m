traindata = load('HDPE_train');
testdata = load('HDPE_test');
virtualdata = load('HDPE_virtual');
%%%%%%%%%%%%%%%%%%%%%% Setting Model Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
interval = 150;
modelName = 'm5p';
modelParam.interval = interval;

%%%%%%%%%%%%%%%%%%% Trainning and Testing Constructed Model %%%%%%%%%%%%%%%
combinedata = [traindata; virtualdata];
data.train = virtualdata;
data.test = testdata;

[net, tr] = experiment(modelName, modelParam, data);
trainInp = data.train(:, 1 : end - 1);
trainTarg = data.train(:, end);
trainOut = tr.output.train;
testInp = data.test(:, 1 : end - 1);
testTarg = data.test(:, end);
testOut = tr.output.test;
trainPerf = sumup(trainTarg, trainOut);
testPerf = sumup(testTarg, testOut);