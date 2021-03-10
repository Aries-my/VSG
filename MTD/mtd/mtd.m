function [L,U,MF] = mtd(data,n,varargin)
    %MTD Mega-trend-defussion
    % %% input arguments
    % data         is a N-by-1 vecteor.
    % n            is the number of vitual sample of a contain field.
    
    % %% output arguments
    % L            is lower bound of a contain field.
    % U            is upper bound of a contain field.
    % MF           is a vecteor which have a same size scale of data, indicating member function value of vitual sample x.
    %
    %Ref. Der-Chiang Li, Liang-Sian Lin.Generating information for small data sets with a multi-modal distribution[J].
    %                                                                 Decision Support Systems,2014,(66),71:81
    %     Der-Chiang Li, Chien-Chih Chen, Che-Jung Chang, Wu-Kuo Lin. A tree-based-trend-diffusion prediction procedure for small sample sets in the early stages of manufacturing systems[J].
    %                                                                  Expert Systems with Applications,2012,(39),1575£º1581
    %
    %
    %
    
    p = inputParser;
    addRequired(p,'data',@(x)validateattributes(x,{'numeric'},{'vector'}))
    addRequired(p,'n',@(x)validateattributes(x,{'numeric'},{'scalar','positive'}))
    addOptional(p,'Method','heuristic',@(x)validateattributes(x,{'char'},{'nonempty'}))
    
    parse(p,data,n,varargin{:})
    
    
    [L,U,CL] = fieldRange(data);
    
    if nargin == 2
        i=1;                              % i-th MF value
        while(n ~= 0)
            tv = L+(U-L)*rand;            % random nunber between L and U, tv
            if tv < CL
                MF(i) = (tv-L)./(CL-L);
            else
                MF(i) = (U-tv)./(U-CL);
            end
            
            i = i+1;
            n = n-1;
        end
    elseif nargin == 3 && strcmp(varargin{1},'heuristic')
        i = 1;
        while(n ~= 0)
            
            while(1)
                tv = L+(U-L)*rand;            % random nunber between L and U, tv
                if tv < CL
                    MF(i) = (tv-L)./(CL-L);
                else
                    MF(i) = (U-tv)./(U-CL);
                end
                
                Pt = MF(i);
                s = rand;
                if s < Pt
                    break                     % if s is smaller than Pt, tv can be keep, otherwise, tv should be droped.
                end 
            end
            
            i = i+1;
            n = n-1;
        end           
    end
end

function [L,U,CL] = fieldRange(data)
    
    p=inputParser;
    addRequired(p,'data',@(x)validateattributes(x,{'double'},{'column'}))
    parse(p,data)
    
    coef=1;
    vector = p.Results.data;
    maxVal = max(vector);
    minVal = min(vector);
    
    CL  =  (maxVal+minVal)./2;
    Sx  =  var(vector); % Sx equal to sx^2
    NU  =  sum(vector>CL);
    NL  =  sum(vector<CL);
    
    SkewL = NL./(NL+NU);
    SkewU = NU./(NL+NU);
    
    tempL = CL-SkewL*sqrt((-2*Sx*log(10e-20))./NL);
    tempU = CL+SkewU*sqrt((-2*Sx*log(10e-20))./NU);
    
    
    if tempL <= minVal
        L = tempL;
    elseif  minVal >= 0
        L = minVal./coef;
    elseif minVal < 0
        L = minVal.*coef;
    end
    
    if tempU >= maxVal
        U = tempU;
    elseif maxVal >= 0
        U = maxVal.*coef;
    elseif maxVal < 0
        U = maxVal./coef;
    end
    
end

