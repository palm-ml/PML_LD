clc
clear
addpath(genpath(pwd));

datasets = {'emotions3'};
dst_folder = "results";
n_fold = 5;

for dataN = 1:1
    % load data
    Dataset = datasets{dataN};
    load(Dataset);
    % preprocess
    n_sample = size(data, 1);
    n_test = round(n_sample / n_fold);
    data = zscore(data);
    % create save_folder
    save_folder = fullfile(dst_folder,Dataset);
    if exist(save_folder,'dir')==0
        mkdir(save_folder);
    end
    results = zeros(n_fold, 5); % save evaluation results
    % n_fold validation and evaluation
    for i = 1:n_fold
        fprintf('Data processing, Cross validation: %d\n', i);
        % split data
        start_idx = (i-1)*n_test + 1;
        if i == n_fold
            test_idx = start_idx : n_sample;
        else
            test_idx = start_idx:start_idx + n_test - 1;
        end
        II = 1:n_sample;
        train_idx = setdiff(II, test_idx);
        train_data = data(train_idx, :);
        train_p_target = partial_labels(:, train_idx);
        test_data = data(test_idx, :);
        test_target = target(:, test_idx);
        % setup parameters for LE model
        param = importdata('arts_param.mat');
        param.tooloptions.maxiter = 50;
        param.tooloptions.gradnorm = 1e-3;
        param.tooloptions.stopfun = @mystopfun;
        param.lambda1 = 0.01;
        param.lambda2 = 0.01;
        param.g = 100;
        param.K = 10;
        % LE process
        [W,numerical] = AMTrain(train_p_target', train_data,param);
        % save LE results
        save_path = fullfile(save_folder,strcat('res',num2str(i)));
        save(save_path, 'numerical');
        % ---------------------------------------------------------------------------------------
        numerical = (softmax(numerical'))';
        % setup parameters for classifier model
        tol  = 1e-10;   %Tolerance during the iteration
        epsi =0.1;      %Instances whose distance computed is more than epsi should be penalized
        ker  = 'rbf';   %Type of kernel function
        beta1=1;        %Penalty parameter
        beta2=50;       %Penalty parameter
        par = 1*mean(pdist(train_data));
        % training classifier model
        [Beta,b] = plmsvr(train_data,numerical,train_p_target',ker,beta1,beta2,epsi,par,tol);
        Pre_LD = PL_LEAF_predict(train_data,test_data,ker,Beta,b,par);
        % eavluation
        results(i,1) = Ranking_loss(Pre_LD',test_target);       % Ranking Loss
        results(i,2) = Average_precision(Pre_LD',test_target);  % Average Precision
        results(i,3) = One_error(Pre_LD',test_target);          % One Error
        results(i,4) = coverage(Pre_LD',test_target);           % Coverage
        bin_Pre_LD = binaryzation(softmax(Pre_LD')',0.1);
        bin_test_target = binaryzation(softmax(test_target)',0.1);
        results(i,5) =  Hamming_loss(bin_Pre_LD',bin_test_target');
    end
    % save results
    save_path = fullfile(save_folder,'evaluations');
    save(save_path, 'results');
end
