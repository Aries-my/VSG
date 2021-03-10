traindata = load('HDPE_train');
testdata = load('HDPE_test');
trainInp = traindata(:, 1 : end - 1);
trainTarg = traindata(:, end);
testInp = testdata(:, 1:end - 1);
testTarg = testdata(:, end);


options.virtualSampleSize = 184;

[virtualInp, vitualTarg, info] = bootstrapvsg(trainInp, trainTarg, options);
virtualdata = [virtualInp, vitualTarg];
save('..\..\datasets\HDPE_virtual', 'virtualdata');