function output = elmpredict(varargin)
%ELMPREDICT Compute output for a given input using ELM. 
%   output = ELMPREDICT(input) feed a given R by Q matrix, input, into  
%       the trained network and return the output (U by Q matrix) of the 
%       network. The trained network is load from a local file.
%   output = ELMPREDICT(net, input) return the output (U by Q matrix) 
%       of  the specified network. input is formated matrix, where row (R)
%       indicates the number of input, column (Q) indicates the number of 
%       samples. Normalization is performed on inputs and reversed 
%       normalization is carried on outputs.


%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS
minargs = 1;  maxargs = 2;
narginchk(minargs, maxargs);

if nargin == 1 && ismatrix(varargin)
    load('elmmodel.mat', 'net' ,'PS', 'TS');
    % net load from local file (elmmodel.mat).
    input = varargin{:};
end

if nargin == 2 && isstruct(varargin{1})
    net = varargin{1};
    input = varargin{2};
    
    assert(isstruct(net), 'net must be a struct.')
end

if ~isequal(size(input, 1), net.numInput)
    error('ELM:calcOutput:misMatch','Query inputs and expected inputs does not match.')
end

normalizationFcn = net.normalizationFcn;
input = preprocess(normalizationFcn, input, PS);

output = elmfeedforward(net, input);

% Calculate error and reversed normalization output.
output = postprocess(normalizationFcn, output, TS);




