function err = cb513_experiment(type, no_hidden, lambda, rho, window)
%CB513_EXPERIMENT Runs experiment on CB513 data set
%
%    err = cb513_experiment(type, no_hidden, lambda, rho, window)
%
% The function runs an experiment on the CB513 data set with the specified 
% CRF type. The optional variables are the number of hidden units no_hidden
% (only applies to 'hidden_' types; default = 100) and the regularization
% parameter lambda (only applies to '_crf' types; default = 0). The
% variable rho indicates the margin variable (only 'perceptron' types).
% The function returns the per-frame error of the experiment in err.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Use Matlab-implementation of gradient
    delete crf_grad.mex* forward_backward_crf.mex* viterbi_crf.mex* crf_herding.mex* crf_herding_2nd_order.mex* hidden_crf_herding.mex* hidden_crf_herding_2nd_order.mex* viterbi_crf.mex* viterbi_crf_2nd_order.mex* viterbi_hidden_crf.mex* viterbi_hidden_crf_2nd_order.mex*

    % Process inputs
    if ~exist('no_hidden', 'var') || isempty(no_hidden)
        no_hidden = 100;
    end
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0;
    end
    if ~exist('window', 'var') || isempty(window)
        window = 13;
    end
    base_eta = 100 * max(1, no_hidden / 100); 
    
    % Load data (PSI-BLAST features)
    X = cell(0);
    load(['data' filesep 'cb513_psiblast.mat']);
    for i=1:length(X)
        X{i} = X{i}(21:end,:);                      % we don't really need residue indicators        
        X{i} = (2 * X{i}) - 1;                      % normalize between -1 and 1
    end
    
    % Implement the windowing of frames
    if window > 1
        D = size(X{1}, 1);
        for i=1:length(X)
            N = size(X{i}, 2);
            cur_X = zeros(window * D, N);
            for h=-fix(window / 2):fix(window / 2)
                cur_X((h + fix(window / 2)) * D + 1:(h + 1 + fix(window / 2)) * D, max(1 - h, 1):min(N - h, N)) = X{i}(:,max(1 + h, 1):min(N + h, N));
            end
            X{i} = cur_X;
        end
    end
    
    % For L-BFGS training, determine lambda based on cross-validation
    if any(strcmpi(type, {'linear_crf', 'quadr_crf', 'hidden_crf'})) && isempty(lambda)
        disp('Determining lambda using cross-validation...');
        perm = randperm(length(X));
        train_ind = perm(1:round(.9 * length(X)));
        test_ind  = perm(1+round(.9 * length(X)):end); 
        train_X = X(train_ind); train_T = T(train_ind);
        test_X  = X(test_ind);  test_T  = T(test_ind);   
        lambda = [0 .001 .002 .005 .01 .02 .05 .1 .2 .5 1 2 5 10 20 50];
        lambda_err = zeros(length(lambda), 1);
        for i=1:length(lambda)
             
            % Train up model
            disp(['Cross-validation test ' num2str(i) ' of ' num2str(length(lambda)) ' (lambda = ' num2str(lambda(i)) ')...']);
            switch type
                case 'linear_crf'
                    model = train_crf(train_X, train_T, 'continuous', lambda(i));
                    
                case 'quadr_crf'
                    model = train_crf(train_X, train_T, 'quadr_continuous', lambda(i));
                    
                case 'hidden_crf'
                    model = train_crf(train_X, train_T, 'drbm_continuous', lambda(i), 150, no_hidden, 'random');
            end
            
            % Perform testing
            tot = 0;
            for j=1:length(test_X)
                if max(cellfun(@max, test_T)) > 3 && exist('label_map', 'var')
                    lambda_err(i) = lambda_err(i) + sum(label_map(viterbi_crf(test_X{j}, model)) ~= label_map(test_T{j}));
                else
                    lambda_err(i) = lambda_err(i) + sum(viterbi_crf(test_X{j}, model) ~= test_T{j});
                end
                tot = tot + length(test_T{j});
            end
            disp(['Test error: ' num2str(lambda_err(i) / tot)]);
        end
        
        % Set optimal lambda
        [~, min_ind] = min(lambda_err);
        lambda = lambda(min_ind);
        disp(['Optimal lambda is: ' num2str(lambda)]);
    end

    % Perform 10-fold cross-validation
    no_folds = 10;
    err = ones(no_folds, 1);
    perm = randperm(length(X)); ind = 1;
    fold_size = floor(length(X) ./ no_folds);
    for fold=1:no_folds
    
        % Split into training and test set
        train_ind = perm([1:ind - 1 ind + fold_size:end]);
        test_ind = perm(ind:ind + fold_size - 1);        
        train_X = X(train_ind);
        train_T = T(train_ind);
        test_X  = X(test_ind);
        test_T  = T(test_ind);
        ind = ind + fold_size;

        % Perform predictions using model
        switch type
            
            case 'perceptron'
                [pred_T, model] = crf_herding(train_X, train_T, test_X, test_T, 'continuous', true, base_eta, rho);                 
            
            case 'perceptron_2nd_order'
                pred_T = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'continuous', true, base_eta, rho);                 
                
            case 'quadr_perceptron'
                [pred_T, model] = crf_herding(train_X, train_T, test_X, test_T, 'quadr_continuous', true, base_eta, rho);  
            
            case 'quadr_perceptron_2nd_order'
                pred_T = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'quadr_continuous', true, base_eta, rho);                 
            
            case 'hidden_perceptron'
                [pred_T, model] = hidden_crf_herding(train_X, train_T, test_X, test_T, 'drbm_continuous', no_hidden, true, base_eta, rho, 200, 20);  
            
            case 'hidden_perceptron_2nd_order'
                [pred_T, model] = hidden_crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'drbm_continuous', no_hidden, true, base_eta, rho);  
                
            case 'herding'
                pred_T = crf_herding(train_X, train_T, test_X, test_T, 'continuous', false, base_eta, rho);  
            
            case 'herding_2nd_order'
                [pred_T, model] = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'continuous', false, base_eta, rho);                 
            
            case 'quadr_herding'
                pred_T = crf_herding(train_X, train_T, test_X, test_T, 'quadr_continuous', false, base_eta, rho);  
              
            case 'quadr_herding_2nd_order'
                pred_T = crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'quadr_continuous', false, base_eta, rho);                 
            
            case 'hidden_herding'
                pred_T = hidden_crf_herding(train_X, train_T, test_X, test_T, 'drbm_continuous', no_hidden, false, base_eta, rho);
            
            case 'hidden_herding_2nd_order'
                pred_T = hidden_crf_herding_2nd_order(train_X, train_T, test_X, test_T, 'drbm_continuous', no_hidden, false, base_eta, rho);
            
            case 'linear_crf'        
                model = train_crf(train_X, train_T, 'continuous', lambda, 150);
                
            case 'quadr_crf'
                model = train_crf(train_X, train_T, 'quadr_continuous', lambda);
                
            case 'hidden_crf'
                model = train_crf(train_X, train_T, 'drbm_continuous', lambda, 250, no_hidden, 'random');
                
            case 'linear_crf_sgd'        
                model = train_crf_sgd(train_X, train_T, 'continuous', lambda, 75, 1e-5, 1);
                
            case 'quadr_crf_sgd'
                model = train_crf_sgd(train_X, train_T, 'quadr_continuous', lambda, 75, 1e-5, 1);
                
            case 'hidden_crf_sgd'
                model = train_crf_sgd(train_X, train_T, 'drbm_continuous', lambda, 50, 2e-3, 1, no_hidden); % was 4e-4
                
            otherwise
                error('Unknown model.');
        end
        
        % Perform prediction on test set for CRFs
        if any(strcmpi(type, {'linear_crf', 'quadr_crf', 'hidden_crf', 'linear_crf_sgd', 'quadr_crf_sgd', 'hidden_crf_sgd'}))
            pred_T = cell(length(test_X), 1);
            for i=1:length(test_X)
                if strcmpi(type, 'hidden_perceptron')
                    pred_T{i} = viterbi_hidden_crf(test_X{i}, model);
                else
                    pred_T{i} = viterbi_crf(test_X{i}, model);
                end
            end
        end
    
        % Measure per-character tagging error
        err(fold) = 0; tot = 0;
        for i=1:length(pred_T)
            tot = tot + length(test_T{i});
            if max(cellfun(@max, T)) > 3 && exist('label_map', 'var')
                err(fold) = err(fold) + sum(label_map(pred_T{i}) ~= label_map(test_T{i}));
            else
                err(fold) = err(fold) + sum(pred_T{i} ~= test_T{i});
            end
        end
        err(fold) = err(fold) / tot;
        disp(['Per-frame error on test set: ' num2str(err(fold))]);
    end
    disp(['Mean error over ' num2str(no_folds) ' folds (lambda = ' num2str(lambda) '): ' num2str(mean(err)) ' (std. dev. ' num2str(std(err)) ')']);
    