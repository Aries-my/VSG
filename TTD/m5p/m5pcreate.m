function net = m5pcreate(M)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here


if nargin < 1
    M = 20;
end

net.interval = M;

% Data set division
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.75;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Performance function
net.performFcn = 'mse';



