function [net, tr] = m5ptrain(net, input, target)

[net, data] = m5pconfigure(net, input, target);

trainInd = data.trainInd;
valInd = data.valInd;
testInd = data.testInd;

validation = true;
if isempty(valInd)
    validation = false;
end

testing = true;
if isempty(testInd)
    testing = false;
end

alldata = [input; target];
traindata = alldata(:, trainInd);
validdata = alldata(:, valInd);
testdata = alldata(:, testInd);

tr.bestPerf.train = [];
tr.bestPerf.val = [];
tr.bestPerf.test = [];

tr.output.train = [];
tr.output.val = [];
tr.output.test = [];

class = 3;
Param = net.interval;
Rt = RunWeka(traindata', traindata', class, Param);
tr.bestPerf.train = Rt.Error.MSE;
tr.output.train = Rt.Y(:, 2);

if validation
    Rv = TestWeka(validdata', class, 'M5P');
    tr.bestPerf.val = Rv.Error.MSE;
    tr.output.val = Rv.Y(:, 2);
end

if testing
    Rw = TestWeka(testdata', class, 'M5P');
    tr.bestPerf.test = Rw.Error.MSE;
    tr.output.test = Rw.Y(:, 2);
end
net.data.train = traindata;
net.data.val = validdata;
net.data.test = testdata;
tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;










