function status = savedataset( varargin )
%SAVEDATASET Save the specified variables to local files.
%   savedataset('filename1.txt', variable1, ..., 'filenamen.txt', variablen)

%   Date: December 31, 2016
%   Author: Zhongsheng Chen (E-mail:zhongsheng.chen@outlook.com)


assert(rem(nargin, 2) == 0, 'Number of input must be even.')

for i = 1 : nargin / 2
    nameprovided{i} = varargin{2 * i - 1};
    requiredvalue{i} = varargin{2 * i};
end

for i = 1 : nargin / 2
    filename = [nameprovided{i}];
    fid = fopen(filename, 'w');
    if fid < 0
        error('Can not open file: %s', filename);
        status = false;
        break;
    else
        [row, col] = size(requiredvalue{i});
        format = [];
        for j = 1 : col
            format = [format, '%10.6f\t'];
        end
        format = [format, '\n'];
        
        for j = 1 : row
            fprintf(fid, format, requiredvalue{i}(j, :));
        end
        ok = fclose(fid);
    end
end
if ok
    status = true;
    ebd
end

