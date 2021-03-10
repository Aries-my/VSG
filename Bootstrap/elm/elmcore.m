function net = elmcore(net, input, target)
%ELMCORE Computer hidden neuron output matrix H and output weights matrix 
%       beta.

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


% the number of samples.
Q = size(input, 2);

% calculate output of hidden layer of the network.
% load('Wb.mat')
Wb = [net.inputWeight, net.bias];
X = [input; ones(1, Q)];

% calculate hidden neuron output matrix
switch net.layer{1}.transferFcn
    case {'logsig'}
        H = logsig(Wb*X);
    case {'tansig'}
        H = tansig(Wb*X);
    case {'sin'}
        H = sin(Wb*X);
    case {'radbas'}
        H = radbas(Wb*X);           % Radial basis function
    case {'tribas'}
        H = tribas(tempH);          % Triangular basis function
    case {'hardlim'}
        H = double(hardlim(tempH)); % Hard Limit
end

% output weights matrix
beta = pinv(H') * target';
net.outputWeight = beta;

