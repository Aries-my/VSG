function [LB, UB, CL, minimum, maximum] = getacptrange(attribute)
%GETACPTRANGE Calculate the acceptable range of the given observations.
%   inputs:
%         x - a row vector of observations.
%   outputs:
%         LB - the lower boundary.
%         UB - the upper boundary.
%         minimum - the minimum of observations.
%         maximum - the maximum of observations.
%         CL      - the central location of observations.
% Usages:
% e.g. x = rand(100,1);
%     [LB, UB, CL, minimum, maximum] = getacptrange(x);



%   Date: December 27, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

minimum = min(attribute);
maximum = max(attribute);
Sx = var(attribute);                    % variance. Sx equal to sx^2
CL = (minimum + maximum) ./ 2;          % central location
NL = sum(attribute < CL);               % the number of data smaller than CL.
NU = sum(attribute >= CL);              % the number of data greater than CL.
SkewL = NL ./ (NL + NU );               % the left skewness.
SkewU = NU ./ (NL + NU );               % the right skewness
tempL = CL - SkewL .* sqrt((-2 .* Sx .* log(10e-20)) ./ NL);
tempU = CL + SkewU .* sqrt((-2 .* Sx .* log(10e-20)) ./ NU);

if tempL <= minimum
    LB = tempL;
else
    LB = minimum;
end

if tempU >= maximum
    UB = tempU;
else
    UB = maximum;
end

