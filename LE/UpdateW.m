function [ W ] = UpdateW(W, X, Y, XPool,ZPool, G,param )

[d,k] = size(W);

manifold = euclideanfactory(d, k);
problem.M = manifold;

XGX = X*G*X';
ZZPool = cell(size(ZPool));
for i=1:length(ZPool)
    ZZPool{i} = ZPool{i}*(ZPool{i})';
end
XXPool = cell(size(XPool));
for i=1:length(XPool)
    XXPool{i} = XPool{i}'*(XPool{i});
end
XX = X*X';
LX = Y*X';
XGX1 = X*G'*X';
XGX2 = X*G*X';
problem.cost = @(w) Wcost(w, X, Y, XGX, ZZPool, XPool,param);
problem.grad = @(w) Wgrad(w, XX, LX, XGX1, XGX2,  ZZPool,XXPool, param);
%checkgradient(problem);

options = param.tooloptions;
%[x xcost info] = trustregions(problem,W,options);
[x xcost info] = steepestdescent(problem,W,options);
W = x;

end

function cost = Wcost(w, X, Y, XGX, ZZPool,XPool, param)

lambda1 = param.lambda1;
lambda2 = param.lambda2;
WX = w*X;
cost = trace((WX - Y)'*(WX - Y))+ lambda1* trace(w*XGX*w') ;
for i=1:length(ZZPool)
    ZZ = ZZPool{i};
    Xi = XPool{i}';
    cost = cost + lambda2*trace(Xi'*w'*ZZ*w*Xi);
end
end

function grad = Wgrad(w,  XX, LX, XGX1, XGX2,  ZZPool,XXPool, param)

lambda1 = param.lambda1;
lambda2 = param.lambda2;


grad = 2*w*XX - 2*LX + lambda1*w*XGX1 + lambda1*w*XGX2;
for i=1:length(ZZPool)
    ZZ = ZZPool{i};
    XiXi = XXPool{i};
    grad = grad + 2*lambda2*ZZ*w*XiXi;
end
end
