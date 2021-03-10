traindata = load('hdpemi_train');
testdata = load('hdpemi_test');
trainInp = traindata(:, 1 : end - 1);
trainTarg = traindata(:, end);
testInp = testdata(:, 1:end - 1);
testTarg = testdata(:, end);


options.virtualSampleSize = 184;

[virtualInp, vitualTarg, info] = bootstrapvsg(trainInp, trainTarg, options);
virtualdata = [virtualInp, vitualTarg];
savedataset('..\..\datasets\hdpemi_virtual', virtualdata)