function T = postprocess(normalizationFcn, TN, TS)
%POSTPROCESS reverse outputs of the network.
%   outputs = POSTPROCESS(net, outputs, TS) reverse outputs of the network 
%       according to target normalization setting (TS);

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        T = mapminmax('reverse', TN, TS);
    case {'mapstd', 'std'}
        T = mapstd('reverse', TN, TS);
end



