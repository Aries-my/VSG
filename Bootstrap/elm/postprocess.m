function T = postprocess(normalizationFcn, TN, TS)
%POSTPROCESS reverse outputs of the network.
%   outputs = POSTPROCESS(net, outputs, TS) reverse outputs of the network 
%       according to target normalization setting (TS);

%   Date: June 25, 2016
%   Author: kalvin chern (E-mail:1456597761@qq.com)

switch lower(normalizationFcn)
    case {'mapminmax', 'minmax'}
        T = mapminmax('reverse', TN, TS);
    case {'mapstd', 'std'}
        T = mapstd('reverse', TN, TS);
end



