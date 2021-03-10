function perf = elmeval(net, input, target)
%ELMEVAL Evaluate peformance of ELM using a give datasets and return 
%   reversed normalization output and error index.

%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

global PS TS

normalizationFcn = net.normalizationFcn;

input = preprocess(normalizationFcn, input, PS);
output = elmfeedforward(net, input);

% Calculate error and reversed normalization output.
output = postprocess(normalizationFcn, output, TS);
perf = sumup(target, output);






