function [Beta,b] = plmsvr(train_data,y,train_p_target,ker,C1,C2,epsi,par,tol)
%plmsvr is the second phase of PL-LEAF training[1] 
%    Syntax
%
%       [Beta,b] = plmsvr(train_data,y,train_p_target,ker,C1,C2,epsi,par,tol)
%
%    Description
%      
%      parameters,
%           train_data     - An PxD array, the ith instance of training instance is stored in train_data(i,:)
%           y              - The return value of function build_label_manifold ,the first phase of PL-LEAF[1]
%           train_p_target - An PxQ array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(i,j) equals +1, otherwise train_p_target(i,j) equals 0
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
n_m=size(train_data,1); %number of instance
n_k=size(y,2); %number of label
Z=train_p_target*n_k-repmat(sum(train_p_target==0,2),1,n_k);
Z=1./Z;
%build the kernel matrix on the labeled samples
H = Kernelmatrix(ker,train_data',train_data',par);
%create matrix for regression parameters
Beta=zeros(n_m,n_k);
b=zeros(1,n_k);
%E = prediction error per output (n_m x n_k)
f=H*Beta+repmat(b,n_m,1);
E=y-f;
u=sqrt(sum(E.^2,2));
%RMSE
RMSE(1,1) = sqrt(mean(u.^2));
%points for which prediction error is larger than epsilon
i1=find(u>=epsi);
%set initial values of alphas
a=2*C1*(u-epsi)./u;
L=zeros(size(u));
% we modify only entries for which  u > epsi. with the sq slack
L(i1)=u(i1).^2-2*epsi*u(i1)+epsi^2;
L2=sum(sum(Z.*f));
%Lp is the loss function to minimize
Lp(1,1)=sum(diag(Beta'*H*Beta))/2+C1*sum(L)/2-C2*L2;
eta=1;
k=2;
hacer=1;
val=1;
while(hacer)
    Beta_a=Beta;
    b_a=b;
    E_a=E;
    u_a=u;
    i1_a=i1;
    M1=[H(i1,i1)+diag(1./a(i1))];
    M1=[M1 ones(size(M1,1),1)];
    temp=[a(i1)'*H(i1,i1) sum(a(i1))];
    M1=[M1;temp];
    M1=M1+1e-11*eye(size(M1,1));
    %compute betas
    sal1=inv(M1)*[y(i1,:)+C2*Z(i1,:)./repmat(a(i1), 1, n_k);(a(i1)'*y(i1,:))+C2*sum(Z)];
    b_sal1=sal1(end,:);
    sal1=sal1(1:end-1,:);
    eta=1;
    %Beta=zeros(size(Beta));
    Beta(i1,:)=sal1;
    b=b_sal1;  
    %error
    f = H*Beta+repmat(b,n_m,1);
    E=y-f;
    %RSE
    u=sqrt(sum(E.^2,2));
    i1=find(u>=epsi);
    L=zeros(size(u));
    L(i1)=u(i1).^2-2*epsi*u(i1)+epsi^2;
    %recompute the loss function
    L2=sum(sum(Z.*f));
    Lp(k,1)=sum(diag(Beta'*H*Beta))/2+C1*sum(L)/2-C2*L2 ;
    %Loop where we keep alphas and modify betas
    while(Lp(k,1)>Lp(k-1,1))
        eta=eta/10;
        i1=i1_a;       
        %the new betas are a combination of the current (sal1) and of the
        %previous iteration (Beta_a)
        Beta(i1,:)=eta*sal1+(1-eta)*Beta_a(i1,:);
        b=eta*b_sal1+(1-eta)*b_a;
        f=H*Beta+repmat(b,n_m,1);
        E=y-f;
        u=sqrt(sum(E.^2,2));
        i1=find(u>=epsi);
        L=zeros(size(u));
        L(i1)=u(i1).^2-2*epsi*u(i1)+epsi^2;
        L2=sum(sum(Z.*f));
        Lp(k,1)=sum(diag(Beta'*H*Beta))/2+C1*sum(L)/2-C2*L2;
        %stopping criterion #1
        if(eta<10^-16)
            'stop 0';
            Lp(k,1)=Lp(k-1,1)-10^-15;
            Beta=Beta_a;
            b=b_a;
            u = u_a;
            i1 = i1_a;
            hacer=0;  
        end      
    end  
    %here we modify the alphas and keep betas.
    a_a=a;
    a=2*C1*(u-epsi)./u;
    RMSE(k,1) = sqrt(mean(u.^2));
    %stopping criterion #2
    if((Lp(k-1,1)-Lp(k,1))/Lp(k-1,1)<tol)
        hacer=0;    
    end
    k=k+1; 
    %stopping criterion #3 - algorithm does not converge. (val = -1)
    if(length(i1)==0)
        hacer=0;
        Beta=zeros(size(Beta));
        val=-1;
        'stop 2';
    end
end