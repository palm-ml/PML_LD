function [W,enhancement,ZPool] = AMTrain(logicalLabel, features,param)
    [d,n] = size(features);
    [l,~] = size(logicalLabel);
    k = param.k;
    k2 = param.k2;    
    perf = 0;

    tic;
    global   trainFeature;
    global   trainLabel;
    global   G;
    global   lambda;
    global   FGF;
    % K-NN similarity matrix A
    Idx = knnsearch(features,features,'K',param.K);
    GraphConnect = zeros(size(features,1),size(features,1));
    for i = 1:size(features,1)
        GraphConnect(i,Idx(i,:)) = 1;
    end
    GraphConnect = GraphConnect + GraphConnect';
    GraphConnect(GraphConnect > 0) = 1;
    sigma = 10;
    A =  exp(-(L2_distance(features', features').^2) / (2 * sigma ^ 2));
    A = A .* GraphConnect;
    A = A - diag(diag(A));
    A_hat = diag(sum(A,2));
    G = A_hat - A;
    
    % linear model,kernel method
    para.ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')
    para.par  = 1*mean(pdist(features)); %parameter of kernel function
    H = kernelmatrix(para.ker, para.par, features, features);% build the kernel matrix on the labeled samples (N x N)
    UnitMatrix = ones(size(features,1),1);
    trainFeature = [H,UnitMatrix];

    FGF = trainFeature'*G*trainFeature;
    % Group 
    [T, ~] = kmeans(features,param.g,'emptyaction','drop');
    [XPool, YPool, param ] = InitGroup( logicalLabel, trainFeature,T, param );

    trainLabel = logicalLabel;
    lambda = param.lambda1;
    tic;
    W = InitW(  trainFeature, logicalLabel, G,param);
    
    fprintf('Training time of BFGS-LLD: %8.7f \n', toc);
    
    g = param.g;
    l = size(logicalLabel,2);
    k2 = size(logicalLabel,2)
    ZPool = cell(g,1);
    for i = 1:g
        ZPool{i} = rand(l,k2);
    end
    param.maxIter = 1;
    param.tooloptions.maxiter = 5;
    param.tooloptions.gradnorm = 1e-3;
    
    obj_old = [];
    last = 0;
    
    init_time = toc;
    
    tic;
    for i=1:param.maxIter
        disp(i);
        for gr=1:g
            X = XPool{gr};
            [Lg] = UpdateS(ZPool{gr},W,X',param);
            ZPool{gr} = Lg;
        end
        [ W] = UpdateW(W, trainFeature', logicalLabel', XPool,ZPool, G,param);
        
        enhancement = trainFeature*W';
        obj = object( W,  trainFeature', logicalLabel', XPool,ZPool, G,param );
        display(obj);
        last = last + 1;
        obj_old = [obj_old;obj];
        
        
        if last < 5
            continue;
        end
        stopnow = 1;
        for ii=1:3
            stopnow = stopnow & (abs(obj-obj_old(last-1-ii)) < 1e-6);
        end
        if stopnow
            break;
        end
    end
    
    end
    function stopnow = mystopfun(problem, x, info, last)
    if last < 5
        stopnow = 0;
        return;
    end
    flag = 1;
    for i = 1:3
        flag = flag & abs(info(last-i).cost-info(last-i-1).cost) < 1e-5;
    end
    stopnow = flag;
    end
    