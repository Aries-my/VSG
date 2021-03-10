function [xnew, ynew, info] = ttdvsg(x, y, options)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


data = [x; y];
newdata = [];
N = size(data, 1);
NV = options.virtualSampleSize;
for i = 1 : N
    
    TV = [];
    [LB, UB, CL] = getacptrange(data(i, :));
    
    for j = 1 : NV
        
        while (true)
            tv =  LB + (UB - LB) * rand;
            pt = membership(tv, LB, UB, CL);
            s = rand;
            
            if s < pt
                TV = [TV, tv];
                break;
            end
        end
        
    end
    
    newdata = [newdata; TV];
end
xnew = newdata(1:end - 1, :);
ynew = newdata(end, :);
info = [];


