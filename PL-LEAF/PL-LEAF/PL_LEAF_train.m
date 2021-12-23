function [Beta,b]= PL_LEAF_train(train_data,train_p_target,k,ker,C1,C2,epsi,par,tol)
%PL_LEAF_train is the training phase of PL-LEAF[1] 
%    Syntax
%
%       [Beta,b]= PL_LEAF_train(train_data,train_p_target,k,ker,C1,C2,epsi,par,tol)
%
%    Description
%      
%      parameters,
%           train_data     - An PxD array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target - An PxQ array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%           k              - Number of neighbors,here we set k=10
%           ker            - Type of kernel function,here we use rbf kernel[2]
%           C1             - Penalty parameter,here we set C1=10
%           C2             - Penalty parameter,here we set C2=1
%           epsi           - Instances whose distance computed is more than epsi should be penalized[2]
%           par            - Parameters of kernel function[2]
%           tol            - Tolerance during the iteration[2]
%      and returns,
%           Beta           - An PxQ array ,coeficient matrix of trainFeature's linear combination
%           b              - An 1xQ array ,intercept matrix
%   [1]Min-Ling Zhang,Bin-Bin Zhou,Xu-Ying Liu. Partial Label Learning via Feature-Aware Disambiguation,In: Proceedings of the 22th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'16), San Francisco,2016.
%   [2]Sanchez-Fernandez M, De-Prado-Cumplido M, Arenas-Garcia J, et al. SVM multiregression for nonlinear channel estimation in multiple-input multiple-output systems[J]. IEEE Transactions on Signal Processing, 2004, 52(8):2298-2307. 
y=build_label_manifold(train_data,train_p_target,k);
%second phase of training
[Beta,b] = plmsvr(train_data,y,train_p_target,ker,C1,C2,epsi,par,tol);
end


