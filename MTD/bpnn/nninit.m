function  net = nninit(net)
%INITNET Return a neural network with random weights (W)
%   and biases (B). weights{1} is weights connecting the first hidden layer
%   to input layer (IW).
    
%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

lb = -1;
ub =  1;
% Number of layers. 
Nl = net.numLayer;

% initialize weights and biases.
for i =  1 : Nl
    if i == 1
        net.weight{i} = (ub - lb) .* rand(net.layer{1}.size, net.numInput) + lb;
    else
        net.weight{i} = (ub - lb) .* rand(net.layer{i}.size, net.layer{i - 1}.size) + lb;
    end
    net.bias{i} = (ub - lb) .* rand(net.layer{i}.size, 1) + lb;
end
