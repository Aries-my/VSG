function output = nnpredict(varargin)
%NNPREDICT Compute output for a given input. 
%   output = NNPREDICT(input) feed a given R by Q matrix, input, into  
%       the trained network and return the output (U by Q matrix) of the 
%       network. The trained network is load from a local file.
%   output = NNPREDICT(net, input) return the output (U by Q matrix) 
%       of  the specified network. input is formated matrix, where row (R)
%       indicates the number of input, column (Q) indicates the number of 
%       samples.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS
minargs = 1;  maxargs = 2;
narginchk(minargs, maxargs);

if nargin == 1 && ismatrix(varargin)
    load('bpmodel.mat', 'net' ,'PS', 'TS');
    % net load from local file (bpmodel.mat).
    input = varargin{:};
end

if nargin == 2 && isstruct(varargin{1})
    net = varargin{1};
    input = varargin{2};
    
    assert(isstruct(net), 'net must be a struct.')
end
 
if ~isequal(size(input, 1), net.numInput)
    error('NN:nnpredict:misMatch','Query inputs and expected inputs does not match.')
end

normalizationFcn = net.normalizationFcn;
input = preprocess(normalizationFcn, input, PS);

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

% calculate error and loss using reversed normalization output.
output = postprocess(normalizationFcn, output, TS);



