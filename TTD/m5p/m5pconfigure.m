function [net, data] = m5pconfigure(net, input, target)

net.numInput = size(input, 1);
net.numOutput = size(target, 1);
N = size(input, 2);
[trainInd, valInd, testInd] = divideindex(net.divideFcn, N, net.divideParam);

data.trainInd = trainInd;
data.valInd = valInd;
data.testInd = testInd;

function  [trainInd, valInd, testInd] = divideindex(divideFcn, N, divideParam)
%   Return index of training set, validation set and testing set.  

switch lower(divideFcn)
    case 'dividerand'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideblock'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideint'
        [trainInd,valInd,testInd] = dividerand(N, divideParam);
    case 'divideind' % FIXME
        assert(~isa(struct2array(divideParam(:)), 'integer'), ' Error in specific index.');
        [trainInd,valInd,testInd] = divideind(N, divideParam);
end