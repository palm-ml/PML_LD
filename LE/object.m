function [ cost ] = object( w, X, Y, XPool,ZPool, G,param )
%OBJECT �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
lambda1 = param.lambda1;
lambda2 = param.lambda2;
XGX = X*G*X';
WX = w*X;
cost = trace((WX - Y)'*(WX - Y))+ lambda1* trace(w*XGX*w') ;
for i=1:length(ZPool)
    ZZ = ZPool{i}*ZPool{i}';
    Xi = XPool{i}';
    cost = cost + lambda2*trace(Xi'*w'*ZZ*w*Xi);
end

end

