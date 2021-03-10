function [xnew, ynew, info] = bootstrapvsg(x, y, options)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

sample = [x, y];
k = options.virtualSampleSize;
boot = bootstrap(sample, k);
xnew = boot(:, 1: end-1);
ynew = boot(:, end);
info = 'bootstrap';
end


function boot = bootstrap(sample, k)

boot = datasample(sample, k);

end

