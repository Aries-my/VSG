function  [loss, outLayer, error] = feedforward(net, input, target)
%FEEDFORWARD Return output of layers according to weights and
%                   biases in forward propagation phrase and error (loss).
%   net = FEEDFORWARD(net, input, target) Feed inputs (in) to
%                           neural network (net), and calculate each
%                           outputs of each layers in turn.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

% the number of batch samples (batch size).
Q = size(input, 2); 

% number of layers (numHiddenLayers + numOutputs).
Nl = net.numLayer;

% the output of input layer of the network is same as the input of input layer.
out = input;

% calculate output of hidden layer of the network.
for i = 1 : Nl - 1
    Wb = [net.weight{i}, net.bias{i}];
    X = [out; ones(1, Q)];
    switch net.layer{i}.transferFcn
        case {'logsig'}
            out = logsig(Wb*X);
        case {'tansig'}
            out = tansig(Wb*X);
        case {'tansigopt'}
            out = tansigopt(Wb*X);
    end
    outLayer{i} = [out; ones(1, Q)];
end

% calculate output of output layer of the network.
Wb = [net.weight{Nl}, net.bias{Nl}];
X = [outLayer{Nl -1}];
switch net.layer{Nl}.transferFcn
    case {'logsig'}
        outLayer{Nl} = logsig(Wb*X);
    case {'purelin'}
        outLayer{Nl} = purelin(Wb*X);
    case {'softmax'}
        outLayer{Nl} = softmax(Wb*X);
end

output = outLayer{Nl};
error = target - output;

switch net.layer{Nl}.transferFcn
    case {'logsig', 'purelin'}
        loss = 1/2 * sum(sum(error .^ 2)) / Q;
    case {'softmax'}
        loss = -sum(sum(target .* log(output))) / Q;
end