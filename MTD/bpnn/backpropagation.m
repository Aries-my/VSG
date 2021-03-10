function [net, gradient, param] = backpropagation(net, input, error, outlayer, param)
%ERRORBACKPROPAGATION Adjust weights and biases of the net according the
%       BP learning algorithm. Return gradient of output layer if output
%       transferFcn is logsig. Otherwise return gradient of last hidden
%       layer if output transferFcn is purelin.
%   Inputs:
%           net - A input training neural network.
%           input - Inputs of network.
%           error - errors between targets and outputs of network.
%           param - A struct for specifying pdW, pdb and d.
%               pdW - Prevoius delta of weights (W).
%               pdb - Prevoius delta of biases (b).
%   output:
%           net - The updated network
%           gradient - gradient of the network. When transfor function on
%   output layer is purelin, the average gradient of nerons of last hidden
%   layer is returned. Otherwise, the average gradient of nerons of output
%   layer is returned.
%   Example:
%           [net, gradient] = backpropagation(net, input, error, outlayer)
%       peroform a standard BP algorithm without moment iterm.
%           [net, gradient, param] = backpropagation(net, input, error, ...
%                                       outlayer, param)
%           peroform a standard BP algorithm with moment iterm.


%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


% the number of batch samples (batch size).
Q = size(input, 2);

Nl =  net.numLayer;
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        gradient = outlayer{Nl} .* (1 - outlayer{Nl});
        d{Nl} =  gradient .* error;
    case {'purelin', 'softmax'}
        gradient = outlayer{Nl - 1} .* (1 - outlayer{Nl - 1});
        d{Nl} = error;
end

for i = Nl - 1 : -1 : 1
    switch net.layer{i}.transferFcn
        case {'logsig'}
            grad = outlayer{i} .* (1 - outlayer{i});
        case {'tansig'}
            grad = 1 - outlayer(i) .^ 2;
        case {'tansigopt'}
            grad = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * outlayer{i} .^ 2);
    end
    Wb = [net.weight{i + 1}, net.bias{i + 1}];
    if i + 1 == Nl
        d{i} = grad .* (Wb' * d{i + 1});
    else
        d{i} = grad .* (Wb' * d{i + 1}(1 : end - 1, :));
    end
end

% Comput dW and db.
for i = Nl : -1 : 1
    if i == Nl
        dWB{i} = (d{i} * outlayer{i - 1}') / Q;
    elseif i == 1
        dWB{i} = (d{i}(1 : end - 1, :) * [input; ones(1, Q)]') / Q;
    else
        dWB{i} = (d{i}(1 : end - 1, :) * outlayer{i - 1}') / Q;
    end
end

% update weights and biases.
mc = net.mc;
for i = Nl : -1 : 1
    lr = net.lr;
    dWb = lr .* dWB{i};
    if mc > 0
        pdWb = [param.pdW{i}, param.pdb{i}];
        pdWb = mc * pdWb + (1 - mc) * dWb;
        
        dWb = pdWb;
    end
    Wb = [net.weight{i}, net.bias{i}];
    Wb = Wb + dWb;
    
    net.weight{i} = Wb(:, 1 : end - 1); net.bias{i} = Wb(:, end);
    param.pdW{i} = pdWb(:, 1 : end - 1); param.pdb{i} = pdWb(:, end);
    param.dW{i} = dWb(:, 1 : end - 1); param.db{i} = dWb(:, end); 
end

% Average gradient.
gradient = sum(sum(gradient)) / (size(gradient, 1) * size(gradient, 2));




