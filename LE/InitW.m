function [ W ] = InitW(trainFeature,trainLabel,G,param)


lambda = param.lambda1;
item=rand(size(trainFeature,2),size(trainLabel,2));
%save dt.mat trainFeature trainLabel G lambda
[W,fval] = fminlbfgsGLLE(@LEbfgsProcess,item);
W = W';

end

