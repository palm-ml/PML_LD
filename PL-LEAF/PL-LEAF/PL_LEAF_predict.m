function [predict_LD]= PL_LEAF_predict(train_data,test_data,ker,Beta,b,par)
%PL_LEAF_predict is the testing phase of PL-LEAF[1] 
%    Syntax
%
%       [accuracy,predict_label]=PL_LEAF_predict(Beta,b,test_data,test_target)
%
%    Description
%      
%      parameters,
%           train_data    - An PxD array, the ith instance of training instance is stored in train_data(i,:)
%           test_data     - An TxD array, the ith instance of training instance is stored in train_data(i,:)
%           test_target   - An TxQ array, if the jth class label is one of the ground-tuth labels for the ith test instance, then train_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
%           ker           - Type of kernel function,here we use rbf kernel[2]
%           Beta          - An PxQ array ,coeficient matrix of trainFeature's linear combination
%           b             - An 1xQ array ,intercept matrix
%           par           - Parameters of kernel function[2]
%      and returns,
%           accuracy      - Predictive accuracy on the test set
%           predict_label - An TxQ array, if the ith test instance is predicted to have the jth class label, then predict_label(i,j) is 1, otherwise predict_label(i,j) is 0
%   [1]Min-Ling Zhang,Bin-Bin Zhou,Xu-Ying Liu. Partial Label Learning via Feature-Aware Disambiguation,In: Proceedings of the 22th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'16), San Francisco,2016.    
%   [2]Sanchez-Fernandez M, De-Prado-Cumplido M, Arenas-Garcia J, et al. SVM multiregression for nonlinear channel estimation in multiple-input multiple-output systems[J]. IEEE Transactions on Signal Processing, 2004, 52(8):2298-2307. 
num_test=size(test_data,1); %the number of testing instance


Ktest = Kernelmatrix(ker,test_data',train_data',par);
Ypredtest =Ktest*Beta+repmat(b,num_test,1);
distribution  = softmax(Ypredtest');
predict_LD = distribution';
end


