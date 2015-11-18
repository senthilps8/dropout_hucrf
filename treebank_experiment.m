function treebank_experiment(type, no_hidden, rho)
%TREEBANK_EXPERIMENT Runs a POS tagging experiment on the Penn Treebank
%
%   treebank_experiment(type, no_hidden, rho)
%
% The function runs an experiment on the Penn Treebank with the specified 
% CRF type. The optional variables are the number of hidden units no_hidden
% (only applies to 'hidden_' types; default = 100) and the variable rho, 
% which indicates the margin variable.
% The function returns the per-word tagging error of the experiment in err.
%
% NOTE: This function only implements perceptron training!
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Use MEX-implementations
	warning('Please do not run this code in the Matlab GUI, but in the console. The Matlab GUI suppresses progress indicators of this experiment.');
    mexall    
    
    % Default values
    if ~exist('no_hidden', 'var') || isempty(no_hidden)
        no_hidden = 100;
    end
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0;
    end
    base_eta = 500000;
    max_iter = 100;
    burnin_iter = 10;
        
    % Load data
    load(['data' filesep 'treebank_pos.mat']);
    
    % Create training and test set
    ind = randperm(length(X));                          % random 0.9-0.1 division
    train_X = X(ind(1:round(.9 * length(X))));
    train_T = T(ind(1:round(.9 * length(X))));
    test_X  = X(ind(1+round(.9 * length(X)):end));
    test_T  = T(ind(1+round(.9 * length(X)):end));
    clear X T
    tic
    
    % Perform predictions
    switch type
        
        case 'herding'
            pred_T = crf_herding(train_X, train_T, test_X, test_T, 'discrete', false, base_eta, rho, max_iter, burnin_iter);
        
        case 'hidden_herding'
            pred_T = hidden_crf_herding(train_X, train_T, test_X, test_T, 'drbm_discrete', no_hidden, false, base_eta, rho, max_iter, burnin_iter);
        
        case 'herding_2nd_order'
            pred_T = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'discrete', false, base_eta, rho, max_iter, burnin_iter); 
        
        case 'hidden_herding_2nd_order'
            pred_T = hidden_crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'drbm_discrete', no_hidden, false, base_eta, rho, max_iter, burnin_iter);
        
        case 'perceptron'
            pred_T = crf_herding(train_X, train_T, test_X, test_T, 'discrete', true, base_eta, rho, max_iter, burnin_iter);
            
        case 'perceptron_2nd_order'
            pred_T = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'discrete', true, base_eta, rho, max_iter, burnin_iter);        
            
        case 'hidden_perceptron'
            pred_T = hidden_crf_herding(train_X, train_T, test_X, test_T, 'drbm_discrete', no_hidden, true, base_eta, rho, max_iter, burnin_iter);
        
        case 'hidden_perceptron_2nd_order'
            pred_T = hidden_crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'drbm_discrete', no_hidden, true, base_eta, rho, max_iter, burnin_iter);
        
        case {'linear_crf', 'hidden_crf', 'linear_crf_sgd', 'hidden_crf_sgd'}

            % Train Conditional Random Field
            switch type
                case 'linear_crf'
                    model = train_crf(train_X, train_T, 'discrete', lambda, 250, [], [], optimizer);
                case 'hidden_crf'
                    model = train_crf(train_X, train_T, 'drbm_discrete', lambda, 250, no_hidden, 'random', optimizer);
                case 'linear_crf_sgd'
                    max_iter = 100;
                    model = train_crf_sgd(train_X, train_T, test_X, test_T, 'discrete', lambda, max_iter, .05, 100);                   
                case 'hidden_crf_sgd'
                    max_iter = 100;
                    model = train_crf_sgd(train_X, train_T, test_X, test_T, 'drbm_discrete', lambda, max_iter, .05, 100, no_hidden);
            end
        
            % Perform prediction on test set
            pred_T = cell(length(test_X), 1);
            for i=1:length(test_X)
                pred_T{i} = viterbi_crf(test_X{i}, model);
            end
        otherwise
            error('Unknown emission potential.');
    end
        
    % Measure per-word tagging error (on test set)    
    err = 0; tot = 0;
    for i=1:length(test_T)
        tot = tot + length(test_T{i});
        err = err + sum(pred_T{i} ~= test_T{i});
    end
    disp(['Per-word tagging error (test set): ' num2str(err / tot)]);
    toc    