function [pred_T, model] = crf_herding_2nd_order(train_X, train_T, test_X, test_T, type, average_models, base_eta, rho, max_iter, burnin_iter)
%CRF_HERDING Performs herding in a second-order chain CRF
%
%   [pred_T, model] = crf_herding_2nd_order(train_X, train_T, test_X, test_T, type, average_models, base_eta, rho, max_iter, burnin_iter)
%   [pred_T, model] = crf_herding_2nd_order(train_X, train_T, test_X, test_T, model, average_models, base_eta, rho, max_iter, burnin_iter)
%
% Performs herding in a second-order chain CRF on the data in the cell-array
% train_X, and the corresponding targets train_T. The function performs
% target prediction on the time series in test_X. The variable type
% describes the type of emission potential: 'discrete', 'continuous', or 
% 'quadr_continuous'. The variable average_models can be set to true for
% basic perceptron-training, and to false for herding (default = false).
% The variable base_eta is the base step size (default = 1). The variable
% max_iter indicates the number of iterations (default = 100). The variable
% burnin_iter specifies the burn-in time (default = 10).
% The predictions for the test data are returned in pred_T. The used
% predictor is returned in model. 
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('average_models', 'var') || isempty(average_models)
        average_models = false;
    end
    if ~exist('base_eta', 'var') || isempty(base_eta)
        base_eta = 1;
    end
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0;
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 100;
    end
    if ~exist('burnin_iter', 'var') || isempty(burnin_iter)
        burnin_iter = 10;
    end

    % Initialize some variables
    if isstruct(type)
        model = type;
        type = model.type;
    end
    n = length(train_T);
    m = length(test_T);
    total_length = 0;
    for i=1:n
        total_length = total_length + length(train_T{i});
    end
    pred_interval = min(2000, n / 10);
    
    % Compute number of features / dimensionality and number of labels
    if strcmpi(type, 'discrete')
        D = 0;
        for i=1:n
            for j=1:length(train_X{i})
                D = max(D, max(train_X{i}{j}));
            end
        end
        for i=1:m
            for j=1:length(test_X{i})
                D = max(D, max(test_X{i}{j}));
            end
        end
    elseif any(strcmpi(type, {'continuous', 'quadr_continuous'}))
        D = size(train_X{1}, 1);
    else
        error('Data type should be discrete or continuous.');
    end
    K = 0;
    for i=1:n
        K = max(K, max(train_T{i}));
    end
    
    % Initialize model
    if ~exist('model', 'var')
        model.type = type;
        model.A      = zeros(K, K, K);
        model.E      = randn(D, K) * .0001;
        model.E_bias = zeros(1, K);
        model.pi     = zeros(K, 1);
        model.pi2    = zeros(K, K);
        model.tau    = zeros(K, 1);
        model.tau2   = zeros(K, K);
        if strcmpi(type, 'quadr_continuous')
            model.invS = repmat(eye(D), [1 1 K]);
        end
    end
    
    % Initialize mean model, or training and test predictions
    if average_models
        mean_model = model;
        ii = 0;
    else
        pred_trn_T = cell(length(train_X), 1);
        pred_tst_T = cell(length(test_X),  1);
        for i=1:length(train_X)
            pred_trn_T{i} = zeros(K, size(train_X{i}, 2));
        end    
        for i=1:length(test_X)
            pred_tst_T{i} = zeros(K, size(test_X{i}, 2));
        end    
    end
    
    % Compute step sizes
    eta_P  = base_eta / (total_length * numel(model.pi));
    eta_P2 = base_eta / (total_length * numel(model.pi2));
    eta_T  = base_eta / (total_length * numel(model.tau));
    eta_T2 = base_eta / (total_length * numel(model.tau2));
    eta_A  = base_eta / (total_length * K * K);
    if strcmpi(type, 'discrete')
        eta_E  = base_eta / (total_length * numel(model.A));
    else
        eta_E  = base_eta / (total_length * numel(model.E));
    end
    if strcmpi(type, 'quadr_continuous')
        eta_invS = base_eta / (total_length * numel(model.invS));
    end

    % Perform sweeps through training data
    for iter=1:max_iter
        
        % Print out progress
        disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) '...']);
        old_P = model.pi; old_PP = model.pi2; old_R = model.tau; old_RR = model.tau2; old_A = model.A; old_E = model.E;
        ind = randperm(n);
        train_X = train_X(ind);
        train_T = train_T(ind);
        
        % Sweep through all training time series
        for i=1:n            
                        
            % Run Viterbi decoder
            cur_T = viterbi_crf_2nd_order(train_X{i}, model, train_T{i}, rho);
            
            % Update co-occurring state parameters (positive phase)
            model.pi(train_T{i}(1)) = model.pi(train_T{i}(1)) + eta_P;
            if length(train_T{i}) > 1
                model.pi2(train_T{i}(1), train_T{i}(2)) = model.pi2(train_T{i}(1), train_T{i}(2)) + eta_P2;
            end
            model.tau(train_T{i}(end)) = model.tau(train_T{i}(end)) + eta_T;
            if length(train_T{i}) > 1
                model.tau2(train_T{i}(end - 1), train_T{i}(end)) = model.tau2(train_T{i}(end - 1), train_T{i}(end)) + eta_T2;
            end            
            for j=3:length(train_T{i})
                model.A(train_T{i}(j - 2), train_T{i}(j - 1), train_T{i}(j)) = ...
                model.A(train_T{i}(j - 2), train_T{i}(j - 1), train_T{i}(j)) + eta_A;
            end
            
            % Update state-data parameters (positive phase)
            if strcmpi(type, 'discrete')
                for j=1:length(train_T{i})
                    model.E(train_X{i}{j}, train_T{i}(j)) = model.E(train_X{i}{j}, train_T{i}(j)) + eta_E;
                    model.E_bias(train_T{i}(j)) = model.E_bias(train_T{i}(j)) + eta_E;
                end
            elseif strcmpi(type, 'continuous')
                for k=1:K
                    ind = (train_T{i} == k) & (cur_T ~= k);
                    model.E(:,k) = model.E(:,k) + eta_E * sum(train_X{i}(:,ind), 2);
                    model.E_bias(k) = model.E_bias(k) + eta_E * length(ind);
                end
            elseif strcmpi(type, 'quadr_continuous')                
                for k=1:K
                    ind = (train_T{i} == k) & (cur_T ~= k);
                    zero_mean_X = bsxfun(@minus, train_X{i}(:,ind), model.E(:,k));
                    model.E(:,k) = model.E(:,k) - eta_E * 2 * sum(model.invS(:,:,k) * zero_mean_X, 2);
                    model.invS(:,:,k) = model.invS(:,:,k) + eta_invS * (zero_mean_X * zero_mean_X');
                    model.E_bias(k) = model.E_bias(k) + eta_E * length(ind);
                end
            end
            
            % Update co-occurring state parameters (negative phase)
            model.pi(cur_T(1)) = model.pi(cur_T(1)) - eta_P;
            if length(cur_T) > 1
                model.pi2(cur_T(1), cur_T(2)) = model.pi2(cur_T(1), cur_T(2)) - eta_P2;
            end
            model.tau(cur_T(end)) = model.tau(cur_T(end)) - eta_T;
            if length(cur_T) > 1
                model.tau2(cur_T(end - 1), cur_T(end)) = model.tau2(cur_T(end - 1), cur_T(end)) - eta_T2;
            end
            for j=3:length(cur_T)
                model.A(cur_T(j - 2), cur_T(j - 1), cur_T(j)) = model.A(cur_T(j - 2), cur_T(j - 1), cur_T(j)) - eta_A;
            end
            
            % Update state-data parameters (negative phase)             
            if strcmpi(type, 'discrete')
                for j=1:length(cur_T)
                    model.E(train_X{i}{j}, cur_T(j)) = model.E(train_X{i}{j}, cur_T(j)) - eta_E;
                    model.E_bias(cur_T(j)) = model.E_bias(cur_T(j)) - eta_E;
                end
            elseif strcmpi(type, 'continuous')                
                for k=1:K
                    ind = (cur_T == k) & (train_T{i} ~= k);
                    model.E(:,k) = model.E(:,k) - eta_E * sum(train_X{i}(:,ind), 2);
                    model.E_bias(k) = model.E_bias(k) - eta_E * length(ind);
                end
            elseif strcmpi(type, 'quadr_continuous')
                for k=1:K
                    ind = (cur_T == k) & (train_T{i} ~= k);
                    zero_mean_X = bsxfun(@minus, train_X{i}(:,ind), model.E(:,k));
                    model.E(:,k) = model.E(:,k) + eta_E * 2 * sum(model.invS(:,:,k) * zero_mean_X, 2);
                    model.invS(:,:,k) = model.invS(:,:,k) - eta_invS * (zero_mean_X * zero_mean_X');
                    model.E_bias(k) = model.E_bias(k) - eta_E * length(ind);
                end
            end
            
            % Make test predictions
            if iter >= burnin_iter && ~average_models && ~rem(i, pred_interval)
                for j=1:m
                    sequence = viterbi_crf_2nd_order(test_X{j}, model);
                    pred = zeros(K, length(sequence));
                    pred(sub2ind(size(pred), sequence, 1:length(sequence))) = 1;
                    pred_tst_T{j} = pred_tst_T{j} + pred;
                end
            end
        end
        
        % Print out parameter change
        change = sum(abs(old_P - model.pi)) + sum(abs(old_PP(:) - model.pi2(:))) + sum(abs(old_R - model.tau)) + sum(abs(old_RR(:) - model.tau2(:))) + sum(abs(old_A(:) - model.A(:))) + sum(abs(old_E(:) - model.E(:)));
        disp(['Cumulative parameter change: ' num2str(change)]);
        
        % Only if we already have predictions or a mean model
        if iter >= burnin_iter
            pred_T = cell(m, 1);
            
            % Average models and perform prediction
            if average_models
                ii = ii + 1;
                mean_model.pi   = ((ii - 1) / ii) .* mean_model.pi   + (1 / ii) .* model.pi;
                mean_model.pi2  = ((ii - 1) / ii) .* mean_model.pi2  + (1 / ii) .* model.pi2;
                mean_model.tau  = ((ii - 1) / ii) .* mean_model.tau  + (1 / ii) .* model.tau;
                mean_model.tau2 = ((ii - 1) / ii) .* mean_model.tau2 + (1 / ii) .* model.tau2;
                mean_model.A    = ((ii - 1) / ii) .* mean_model.A    + (1 / ii) .* model.A;
                mean_model.E    = ((ii - 1) / ii) .* mean_model.E    + (1 / ii) .* model.E;
                if strcmpi(type, 'quadr_continuous')
                    mean_model.invS = ((ii - 1) / ii) .* mean_model.invS + (1 / ii) .* model.invS;
                end
                for i=1:m
                    pred_T{i} = viterbi_crf_2nd_order(test_X{i}, mean_model);
                end
                
            % Get most likely sequences after voting
            else
                pred_T = cell(m, 1);
                for i=1:m
                    [~, pred_T{i}] = max(pred_tst_T{i}, [], 1);
                end
            end
            
            % Compute current test error
            err = 0; len = 0;
            for i=1:m
                len = len + length(pred_T{i});
                err = err + sum(pred_T{i} ~= test_T{i});          
            end
            disp([' - test error: ' num2str(err / len)]);
        end
    end
    