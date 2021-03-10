traindata = load('HDPE_train');
testdata = load('HDPE_test');
trainInp = traindata(:, 1 : end - 1)';
trainTarg = traindata(:, end)';
testInp = testdata(:, 1:end - 1)';
testTarg = testdata(:, end)';

virtualSampleSize = 184;
option.virtualSampleSize = virtualSampleSize;
[virtualInp, vitualTarg, info] = ttdvsg(trainInp, trainTarg, option);
virtualdata = [virtualInp; vitualTarg]';
savedataset('..\..\datasets\HDPE_virtual', virtualdata)