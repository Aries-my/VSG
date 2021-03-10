function [stop, flag, fail] = stopcriteria(net, i, criteria, avgGradient, trainPerf, validPerf)
%STOPCRITERION Trigger a stop criterion and Return stop category.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)

persistent count; % count for validation error violation.
if isempty(count)
    count = 0;
end

if criteria.maxiteration && i == net.epoch
    stop = true;
    flag = 'maxiteration';
    fail = count;
    return;
end

if criteria.goal && trainPerf(i) < net.goal
    stop = true;
    flag = 'goal';
    fail = count;
    return;
end

if criteria.mingrad && avgGradient(i) < net.mingrad
    stop = true;
    flag = 'mingrad';
    fail = count;
    return;
end

if i > 1 && validPerf(i) - validPerf(i - 1) > 0
    count = count + 1;
else
    count = 0;
end

fail = count;
if  criteria.maxfail && fail == net.maxfail
    stop = true;
    flag = 'maxfail';
    return;
end
stop = false;
flag = 'normal';

