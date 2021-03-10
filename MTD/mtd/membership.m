function MV = membership(attribute, LB, UB, CL)
% MEMBERSHIP Return membership fuction value for given attribute.
% attribute - must be a row vector.
% MV - has same dimension with attribute

MV = zeros(size(attribute));
numObservation = size(attribute, 2);
for i = 1 : numObservation % i-th sample of attribute
    if attribute(i) < CL
        MV(i) = (attribute(i) - LB) ./ (CL - LB);
    else
        MV(i) =  (UB - attribute(i)) ./ (UB - CL);
    end
end
