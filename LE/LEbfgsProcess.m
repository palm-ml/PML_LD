function [target,gradient] = LEbfgsProcess(weights)
%BFGSPROCESS	Provide the target function and the gradient.
%
%	Description
%   [TARGET,GRADIENT] = LEbfgsProcess(WEIGHTS) provides the target function and the gradient.
%   They will be used in the optimization of leBFGS-- LEBFGSTRAIN.
%   
%   Inputs,
%       WEIGHTS: the weights which will be optimized in LEBFGSTRAIN
%   Outputs,
%       TARGET:  the target function which will be used in LEBFGSTRAIN
%       GRADIENT: the gradient which will be used in LEBFGSTRAIN
% 
%	See also
%	LEBFGSTRAIN,LEPREDICT, FMINLBFGS 
%	
%   Copyright: Ning Xu (xning@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

% Load the data set.
global   trainFeature;
global   trainLabel;
global   lambda;
global   FGF;
%load dt.mat;
[size_sam,size_X]=size(trainFeature);

modProb =  trainFeature * weights;  % size_sam * size_Y

% %Target function.
% target = sum(sum((trainLabel - modProb).^2));
%  
% % The gradient.
% gradient = trainFeature'*(modProb - trainLabel);


L = sum(sum((modProb - trainLabel).^2)); 
R =  trace(weights'* FGF * weights);
gradL = 2*trainFeature' * (modProb - trainLabel);
gradR =  FGF*weights + ( FGF)'*weights;


target =( L + lambda * R );
gradient = (gradL +  lambda * gradR);

end


