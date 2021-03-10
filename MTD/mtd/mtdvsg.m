function [xnew, ynew, info] = mtdvsg(x, y, options)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

ex = extendattribute(x);

hiddenLayerSize = options.hiddenLayerSize;
mc = options.momentCoefficient;
lr = options.learnRatio;
epoch = options.maxIteration;
goal = options.goal;

net = nncreate(hiddenLayerSize);
net.lr = lr; % Training parameters
net.mc = mc;
net.epoch = epoch;  % Stop criteria
net.goal = goal;
net.divideFcn = 'dividerand';  % Dataset division
net.divideParam.trainRatio = 1.00;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

[net, tr] = nntrain(net, ex, y);

info.tr = tr;
xnew = createvirtualinput(x, options);
ynew = createvirtualoutput(net, xnew);


function xnew = createvirtualinput(x, options)

xnew = [];
NV = options.virtualSampleSize;
numAtrribute = size(x, 1);
for i = 1 : numAtrribute
    
    TV = [];
    MV = [];
    [LB, UB, CL] = getacptrange(x(i, :));
    
    for j = 1 : NV
        
        strategy = 'non-heuristic';
        switch lower(strategy)
            case {'heuristic'}
                while (true)
                    tv =  LB + (UB - LB) * rand;
                    pt = membership(tv, LB, UB, CL);
                    s = rand;
                    
                    if s < pt
                        MV = [MV, pt];
                        TV = [TV, tv];
                        break;
                    end
                end
                
            case {'non-heuristic'}
                tv = LB + (UB - LB) * rand;
                pt = membership(tv, LB, UB, CL);
                MV = [MV, pt];
                TV = [TV, tv];
            otherwise
                tv = LB + (UB - LB) * rand;
                pt = membership(tv, LB, UB, CL);
                MV = [MV, pt];
                TV = [TV, tv];
        end
        
    end
    
    xnew = [xnew; TV; MV];
    
end

function ynew = createvirtualoutput(net, xnew)

ynew = nnpredict(net, xnew);

