function K = kernelmatrix(ker, parameter, testX, trainX)
% KERNELMATRIX calculate the kernel matrix of instance vectors in testX and trainX.
%
%	Description
%   K = KERNELMATRIX(KER, X, trainX, PARAMETER) calculate the kernel 
%   matrix of the instance vectors in testX and trainX.
%   The function we writed is based on the original kernelmatrix function 
%   writed by Gustavo Camps-Valls(2006(c)) and updated by Jordi(jordi@uv.es,2007-11).
%
%   Inputs,
%       KER:          type of kernel function ('lin', 'poly', 'rbf', 'sam')
%       PARAMETER:    parameters of kernel function
%       TESTFEATURE:  data matrix with training samples in rows and features in columns (N1 x D)
%       TRAINFEATURE: data matrix with test samples in rows and features in in columns (N2 x D)
%
%   Outputs,
%       K:  kernel matrix, each element corresponds to the kernel of two feature vectors (N1 x N2)
%
%   Extended description of input/ouput variables
%       PARAMETER:
%           SIGMA:  width of the RBF and sam kernel
%           BIAS:   bias in the linear and polinomial kernel
%           DEGREE: degree in the polynomial kernel
%
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
switch ker
    case 'lin'
        if exist('trainX','var')
            K = testX * trainX' + parameter;
        else
            K = testX * testX' + parameter;
        end

    case 'poly'
        if exist('trainX','var')
            K = (testX * trainX' + 1).^parameter;
        else
            K = (testX * testX' + 1).^parameter;
        end
        
    %To speed up the computation of the RBF kernel matrix, 
    %we exploit a decomposition of the Euclidean distance (norm).
    case 'rbf'  
        n1sq = sum(testX'.^2,1); %compute x^2
        n1 = size(testX',2);
        if isempty(trainX);
            %||x-y||^2 = x^2 + y^2 - 2*x'*y 
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*testX*testX';
        else
            n2sq = sum(trainX'.^2,1);
            n2 = size(trainX',2);
            %||x-y||^2 = x^2 + y^2 - 2*x'*y 
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*testX*trainX'; 
        end;
        K = exp(-D/(2*parameter^2));

    case 'sam'
        if exist('trainX','var');
            D = testX*trainX';
        else
            D = testX*testX';
        end
        K = exp(-acos(D).^2/(2*parameter^2));

    otherwise
        error(['Unsupported kernel ' ker])
end
