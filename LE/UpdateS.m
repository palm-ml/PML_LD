function [ L,S] = UpdateS(L,W,X,param)
   [l,k] = size(L);
  
   manifold = elliptopefactory(l,k);
   problem.M = manifold;
   WX = W*X;
   WXXW = W*X*X'*W';
    % Define the problem cost function and its gradient.
    problem.cost = @(x) LCost(x, WX,param);
    problem.grad = @(x) LGrad(x,WXXW,param);

    % Numerically check gradient consistency.
    %checkgradient(problem);

    options = param.tooloptions;
   % [x xcost info] = trustregions(problem,L,options);
     [x xcost info] = steepestdescent(problem,L,options);
    L = x;
    S = L*L';
%    USU = U'*S*U;
end

function cost = LCost(L,WX,param)
  
    cost = trace(WX'*L*L'*WX);
end
function grad = LGrad(L,WXXW,param)
    grad = 2*WXXW*L;
end
