function net = elminit(net)
%ELMINIT Return a single-hidden layer feedforward neural networks with 
%   random weights (W) and biases (B). weights{1} is weights connecting 
%   the first hidden layer to input layer (IW).
    
%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

lb = -1;
ub =  1;

% initialize input weights, biases, and output weights.
net.inputWeight = (ub - lb) .* rand(net.layer{1}.size, net.numInput) + lb;
net.bias = (ub - lb) .* rand(net.layer{1}.size, 1) + lb;
net.outputWeight = zeros(net.numOutput, net.layer{1}.size);


