function [traindata, testdata, trainInd, testInd] = dividedataset(dataset, numTrain)
%DIVIDEDATASET Divide dataset into two parts, trainning set and 
% testing set. Each row is arranged in a format of [x, y].

%   Date: December 31, 2016
%   Author: Zhongsheng chen (E-mail:zhongsheng.chen@outlook.com)


numTotal = size(dataset, 1);
numTest = numTotal - numTrain;

ind = randperm(numTotal);
trainInd = sort(ind(1 : numTrain));
testInd = sort(ind(numTrain + (1 : numTest)));

traindata = dataset(trainInd, :);          
testdata = dataset(testInd, :);
end

