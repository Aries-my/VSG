function Error = Error2(P,T)
%   Error=Error2(P,T)
%                                ANNN1
%                       
%           NEURAL NETWORK SCRIPT LINK FOR HYDROLOGICAL PURPOSES
%
%                             Version 2.0
%                         
%                       Department of Hydroinformatics
%                                 Delft
%                        Gerald A. Corzo Perez
%                               UNESCO-IHE
%                               www.hi.ihe.nl
%                                --OO--
%
%
%DEFAULT PARAMETERS FOR TRAINING NETWORKS:if isempty(Par)
%     P= Predicted values (vector)
%     T= Target values  (vector)
%
%OTHER FILES IN DIRECTORY
%Error1, Error2, Error7, Error6 in adition works only with Matlab version
% %above 7 and should have the NN toolbox
% Error.RMSE=RMSE;
% Error.NSC=NSC;
% Error.Cor=Cor;
% Error.NRMSE=NRMSE;
% Error.MAE=MAE;
% Error.StdT=StdT;
% Error.StdP=StdP;
% Error.MuT=MuT;
% Error.MuP=MuP;
% Error.PERS=PERS;
% Error.SSE=SSE;
% Error.SSEN=SSEN;  %Sum Squared Error Naive
% Error.RMSEN=RMSEN; %RMSE Naive
% Error.NRMSEN=NRMSEN;%
% Error.MARE=MARE -> Mean Absolute relative error
% Author: Gerald Corzo
% May /2005

S = size(P);
if S(1) < S(2)
    P=P';
end

S=size(T);
if S(1) < S(2)
    T = T';
end

n = size(P, 1);
%Traditional Measures of error
SSE = sum((P-T).^2);
MSE = SSE/size(P, 1);
RMSE = sqrt(SSE/size(P, 1));
StdT = std(T, 1);
StdP = std(P, 1);
NRMSE = 100*RMSE / StdT;%sqrt(SSE/sum((T-mean(T)).^2));
NSC = 1 - SSE/sum((T - mean(T)).^2);
Cor = sum((P-mean(P)).*(T-mean(T)))/(sqrt(sum((P-mean(P)).^2))*sqrt(sum((T-mean(T)).^2)));
MAE = sum(abs(P-T))/size(P,1);
MARE = sum(abs((T-P)./T)) / n;

StdT = std(T,1);
StdP = std(P,1);
MuT = mean(T);
MuP = mean(P);
RMAE = MAE/StdT;

%Calculating PERS
P2 = T(2:end, :);
T2=T(1:end-1, :);
SSEN = sum((P2-T2).^2);
PERS = 1-(SSE/SSEN);
RMSEN = sqrt(SSEN/(n-1));
NRMSEN = 100*RMSEN/std(T2, 1);

Error.MAPE = 100*sum(abs((T-P)./T))./n;
Error.RMSE = RMSE;
Error.MSE = MSE;
Error.NSC = NSC;
Error.Cor = Cor;
Error.NRMSE = NRMSE;
Error.MAE = MAE;
Error.StdT = StdT;
Error.StdP = StdP;
Error.MuT = MuT;
Error.MuP = MuP;
Error.PERS = PERS;
Error.SSE = SSE;
Error.SSEN = SSEN;  %Sum Squared Error Naive
Error.RMSEN = RMSEN; %RMSE Naive
Error.NRMSEN = NRMSEN; %NRMSE Naive
Error.MARE = MARE;

Error.Er = T-P;
Io = find(Error.Er <= 0);
Iu = find(Error.Er > 0);

S1 = size(Io, 1);
S2 = size(Iu, 1);

Error.Po = S1/size(Error.Er, 1);
Error.Pu = S2/size(Error.Er, 1);