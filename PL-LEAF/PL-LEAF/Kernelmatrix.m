function K = Kernelmatrix(ker,X,X2,parameter)
switch ker
    case 'lin'
        if exist('X2','var')
            K = X' * X2 + parameter;
        else
            K = X' * X + parameter;
        end

    case 'poly'
        if exist('X2','var')
            K = (X' * X2 + 1).^parameter;
        else
            K = (X' * X + 1).^parameter;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end;
        K = exp(-D/(2*parameter^2));

    case 'sam'
        if exist('X2','var');
            D = X'*X2;
        else
            D = X'*X;
        end
        K = exp(-acos(D).^2/(2*parameter^2));

    otherwise
        error(['Unsupported kernel ' ker])
end