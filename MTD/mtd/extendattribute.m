function extension = extendattribute(x)
% EXTENDATTRIBUTE Extend attributes using membership function value (MV)
%   inputs:
%        x - is matrix, along with each row vector represent an attribute.
%   outputs:
%        extension - a extension verion of x, which combine x and its
%        membership fuction value.
% Usages:
%     extension = extendattribute(x)

%   Date: January 2, 2017
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


[numAttribute] = size(x, 1);

extension = [];
for i = 1 : numAttribute
    [L, U, CL] = getacptrange(x(i, :));
    MV = membership(x(i, :), L, U, CL);
    extension = [extension; x(i, :); MV];
end


