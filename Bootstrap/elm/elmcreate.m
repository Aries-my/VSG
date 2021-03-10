function net = elmcreate(hiddenLayer)
%ELMCREATE Create a basic ELM. Since ELM is a single-hidden layer  
%   feedforward neural network (SLFN), ELM only have one hidden layer. 
%   In order to expand ELM to deep learning field, I keep a additive layers
%   in this version of ELM implemention.
%   net = elmcreate(hiddenLayer) create a SLFN with hiddenLayer notes. In 
%   basic ELM, hiddenLayer ia a scalar.But, in the future, hiddenLayer can 
%   be a vector whose element is the number of neurons on layers in its 
%   potential deep learning applications.

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


net.numInput = 0;
net.numOutput = 0;

if nargin < 1
    hiddenLayer = 10;
end

hiddenLayerSize = [hiddenLayer, net.numOutput];
Nl = size(hiddenLayer, 2) + 1;

for i = 1 : Nl
    if i == Nl
        net.layer{i}.transferFcn = 'purelin';
    else
        net.layer{i}.transferFcn = 'logsig';
    end
     net.layer{i}.size = hiddenLayerSize(i);
end
net. numLayer = Nl;

% Normalization
net.normalizationFcn = 'mapminmax';
net.normalizationParam.ymax = 1;
net.normalizationParam.ymin = -1;

% Data set division
net.divideFcn = 'dividerand';
net.performFcn = 'mse';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;


