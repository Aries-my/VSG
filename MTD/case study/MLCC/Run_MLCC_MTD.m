traindata = load('..\..\datasets\mlcc_train');
testdata = load('..\..\datasets\mlcc_test');
trainInp = traindata(:, 1 : end - 1)';
trainTarg = traindata(:, end)';
testInp = testdata(:, 1:end - 1)';
testTarg = testdata(:, end)';

options.hiddenLayerSize = 12;
options.momentCoefficient = 0.90;
options.learnRatio = 0.65;
options.maxIteration = 2000;
options.goal = 1e-6;
options.virtualSampleSize = 184;

[virtualInp, vitualTarg, info] = mtdvsg(trainInp, trainTarg, options);
virtualdata = [virtualInp; vitualTarg]';
savedataset('..\..\datasets\mlcc_virtual', virtualdata)