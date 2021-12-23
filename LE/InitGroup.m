function [XPool, YPool, param ] = InitGroup( Y, X, T,param )

[l,n] = size(Y);


gp = unique(T);
g = length(gp);
param.g = g;
XPool = cell(g,1);
YPool = cell(g,1);
for i=1:g
    ii = T==gp(i);
    Xg = X(ii,:);
    XPool{i,1}=Xg;
    Yg = Y(ii,:);
    YPool{i,1}=Yg;
end
end