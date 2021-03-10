traindata = load('mlcc_train');
testdata = load('mlcc_test');
trainInp = traindata(:, 1 : end - 1)';
trainTarg = traindata(:, end)';
testInp = testdata(:, 1:end - 1)';
testTarg = testdata(:, end)';

virtualSampleSize = 184;
option.virtualSampleSize = virtualSampleSize;
[virtualInp, vitualTarg, info] = ttdvsg(trainInp, trainTarg, option);
virtualdata = [virtualInp; vitualTarg]';
savedataset('..\..\datasets\hdpemi_virtual', virtualdata)