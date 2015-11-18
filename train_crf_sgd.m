function [model, train_err, test_err] = train_crf_sgd(train_X, train_T, type, lambda, max_iter, eta, batch_size, no_hidden, test_X, test_T)
%TRAIN_CRF_SGD Trains a chain CRF using stochastic gradient descent
%
%   model = train_crf_sgd(train_X, train_T, type, lambda, max_iter, eta, batch_size)
%
% The function trains a chain Conditional Random Field (CRF) on the time
% sequences and targets specified in the cell arrays train_X and train_T.
% The number of discrete states / dimensionality of the data is specified
% by D, and the number of possible targets by K. The type is either
% 'discrete', 'continuous', 'drbm_discrete', or 'drbm_continuous. Optionally, 
% an L2 regularizer can be specified through lambda (default = 0). The default
% number of iterations max_iter is 500. The default batch_size is 1.
%
% This function differs from TRAIN_CRF in that it uses stochastic gradient
% descent (SGD) to train the CRF.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 10;
    end
    if ~exist('eta', 'var') || isempty(eta)
        eta = 1e-5;
    end
    if ~exist('batch_size', 'var') || isempty(batch_size)
        batch_size = 1;
    end
    annealing = 0.99;
    averaging = true;
    burnin_iter = round(2 * max_iter / 3);
    
    % Compute number of features / dimensionality and number of labels
    N = length(train_X);
    if any(strcmpi(type, {'discrete', 'drbm_discrete'}))
        D = 0;
        for i=1:N
            for j=1:length(train_X{i})
                D = max(D, max(train_X{i}{j}));
            end
        end
    elseif any(strcmpi(type, {'continuous', 'quadr_continuous', 'drbm_continuous'}))
        D = size(train_X{1}, 1);
    else
        error('Unknown emission function.');
    end
    K = 0;
    for i=1:length(train_T)
        K = max(K, max(train_T{i}));
    end

    % Initialize model (all parameters are in the log-domain)    
    model.type = type;
    model.pi  = zeros(K, 1);
    model.tau = zeros(K, 1);
    model.A   = randn(K, K) * .001;
    if any(strcmpi(model.type, {'discrete', 'continuous'}))
        model.E = randn(D, K) * .001;
        model.E_bias = zeros(1, K);
    elseif strcmpi(model.type, 'quadr_continuous')
        model.E = randn(D, K) * .001;
        model.invS = repmat(eye(D), [1 1 K]);
    elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
        model.E = randn(D, no_hidden) * .001;
        model.E_bias = zeros(1, no_hidden);
        model.labE = randn(K, no_hidden) * .001;
        model.labE_bias = zeros(1, K);
    else
        error('Unknown emission function.');
    end

    % Positive part of state prior
    disp('Precomputing positive part of gradient...');
    pos_pi  = zeros(K, N);
    pos_tau = zeros(K, N);
    for i=1:N
        pos_pi( train_T{i}(1),   i) = pos_pi( train_T{i}(1),   i) + 1;
        pos_tau(train_T{i}(end), i) = pos_tau(train_T{i}(end), i) + 1;
    end
    
    % Positive part of transition gradient (over empirical distribution)
    pos_A = zeros(K, K, N);
    for i=1:N
        for j=2:length(train_T{i})
            pos_A(train_T{i}(j - 1), train_T{i}(j), i) = pos_A(train_T{i}(j - 1), train_T{i}(j), i) + 1;
        end
    end
    
    % Positive part of emission gradient (over empirical distribution)
    if strcmpi(model.type, 'discrete')
        pos_E = zeros(D + 1, K);
    elseif strcmpi(model.type, 'continuous')
        pos_E = zeros(D + 1, K, N);
        for i=1:N
            for k=1:K
                pos_E(1:D, k, i) = sum(train_X{i}(:,train_T{i} == k), 2);
                pos_E(end, k, i) = sum(train_T{i} == k);
            end
        end
    elseif strcmpi(model.type, 'quadr_continuous')
        pos_E = zeros(numel(model.E) + numel(model.invS), 1);
    elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
        pos_E = zeros(numel(model.E) + numel(model.labE) + numel(model.E_bias) + numel(model.labE_bias), 1);
    else
        error('Unkown emission function.');
    end
    
    % Initialize parameter vector
    if any(strcmpi(model.type, {'discrete', 'continuous'}))
        x = [model.pi(:); model.tau(:); model.A(:); model.E(:); model.E_bias(:)];
    elseif strcmpi(model.type, 'quadr_continuous')
        x = [model.pi(:); model.tau(:); model.A(:); model.E(:); model.invS(:)];
    else
        x = [model.pi(:); model.tau(:); model.A(:); model.E(:); model.labE(:); model.E_bias(:); model.labE_bias(:)];
    end
    
    % Perform minimization using stochastic gradient descent
    disp('Performing optimization using SGD...');
    train_err = zeros(max_iter, 1);
    test_err  = zeros(max_iter, 1);
    err = zeros(max_iter, 1);
    for iter=1:max_iter
        
        % Prepare for sweep
        old_x = x;
        rand_ind = randperm(N); 
        eta = eta * annealing;
        
        % Loop over all time series
        for i=1:batch_size:N
        
            % Perform gradient update for single time series
            cur_ind = rand_ind(i:min(i + batch_size - 1, N));
            if strcmpi(model.type, 'discrete')
                pos_E = zeros(D + 1, K);
                for j=1:length(cur_ind)
                    for h=1:length(train_T{cur_ind(j)})
                        pos_E(train_X{cur_ind(j)}{h}, train_T{cur_ind(j)}(h)) = ...
                        pos_E(train_X{cur_ind(j)}{h}, train_T{cur_ind(j)}(h)) + 1; % NOTE: This should be speeded up!
                        pos_E(end, train_T{cur_ind(j)}(h)) = pos_E(end, train_T{cur_ind(j)}(h)) + 1;
                    end
                end
                [C, ~, x] = crf_grad(x, train_X(cur_ind), train_T(cur_ind), model, lambda, sum(pos_pi(:,cur_ind), 2), sum(pos_tau(:,cur_ind), 2), sum(pos_A(:,:,cur_ind), 3), pos_E, eta / batch_size);
            elseif strcmpi(model.type, 'continuous')
                [C, ~, x] = crf_grad(x, train_X(cur_ind), train_T(cur_ind), model, lambda, sum(pos_pi(:,cur_ind), 2), sum(pos_tau(:,cur_ind), 2), sum(pos_A(:,:,cur_ind), 3), sum(pos_E(:,:,cur_ind), 3), eta / batch_size);
            else
                [C, ~, x] = crf_grad(x, train_X(cur_ind), train_T(cur_ind), model, lambda, sum(pos_pi(:,cur_ind), 2), sum(pos_tau(:,cur_ind), 2), sum(pos_A(:,:,cur_ind), 3), pos_E, eta / batch_size);
            end
            err(iter) = err(iter) + C;
        end
        err(iter) = err(iter) / N;
        disp(['Iteration ' num2str(iter) ' of ' num2str(max_iter) ': error is ~' num2str(err(iter))]);
        
        % Store current model in model struct
        ind = 1;
        model.pi  = reshape(x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
        model.tau = reshape(x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);    
        model.A   = reshape(x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
        model.E   = reshape(x(ind:ind + numel(model.E)   - 1), size(model.E));   ind = ind + numel(model.E);
        if any(strcmpi(model.type, {'continuous', 'discrete'}))
            model.E_bias = reshape(x(ind:ind + numel(model.E_bias) - 1), size(model.E_bias));
        elseif strcmpi(model.type, 'quadr_continuous')
            model.invS = reshape(x(ind:ind + numel(model.invS) - 1), size(model.invS));
        elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
            model.labE      = reshape(x(ind:ind + numel(model.labE)      - 1), size(model.labE));   ind = ind + numel(model.labE);
            model.E_bias    = reshape(x(ind:ind + numel(model.E_bias)    - 1), size(model.E_bias)); ind = ind + numel(model.E_bias);
            model.labE_bias = reshape(x(ind:ind + numel(model.labE_bias) - 1), size(model.labE_bias));
        end
    
        % Perform averaging after certain number of iterations
        if iter < burnin_iter || ~averaging
            ii = 0; 
            mean_x = x; 
            mean_model = model;
        else
            ii = ii + 1; 
            mean_x = ((ii - 1) / ii) .* mean_x + (1 / ii) .* x;            
            ind = 1;
            mean_model.pi  = reshape(mean_x(ind:ind + numel(model.pi)  - 1), size(model.pi));  ind = ind + numel(model.pi);
            mean_model.tau = reshape(mean_x(ind:ind + numel(model.tau) - 1), size(model.tau)); ind = ind + numel(model.tau);    
            mean_model.A   = reshape(mean_x(ind:ind + numel(model.A)   - 1), size(model.A));   ind = ind + numel(model.A);
            mean_model.E   = reshape(mean_x(ind:ind + numel(model.E)   - 1), size(model.E));   ind = ind + numel(model.E);
            if any(strcmpi(model.type, {'continuous', 'discrete'}))
                mean_model.E_bias = reshape(mean_x(ind:ind + numel(model.E_bias) - 1), size(model.E_bias));
            elseif strcmpi(model.type, 'quadr_continuous')
                mean_model.invS = reshape(mean_x(ind:ind + numel(model.invS) - 1), size(model.invS));
            elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
                mean_model.labE      = reshape(mean_x(ind:ind + numel(model.labE)      - 1), size(model.labE));   ind = ind + numel(model.labE);
                mean_model.E_bias    = reshape(mean_x(ind:ind + numel(model.E_bias)    - 1), size(model.E_bias)); ind = ind + numel(model.E_bias);
                mean_model.labE_bias = reshape(mean_x(ind:ind + numel(model.labE_bias) - 1), size(model.labE_bias));
            end
        end
        
        % Print norm of update
        change = sum(abs(old_x - x));
        disp(['Cumulative parameter change: ' num2str(change)]);
        
        % Estimate training error
        if nargout > 1
            tot = 0;
            for i=1:length(train_X)
                tot = tot + length(train_T{i});
                train_err(iter) = train_err(iter) + sum(viterbi_crf(train_X{i}, mean_model) ~= train_T{i});
            end
            train_err(iter) = train_err(iter) / tot;
            disp(['  - training error: ~' num2str(train_err(iter))]);
        end
        
        % Estimate test error
        if nargout > 2 && exist('test_X', 'var') && exist('test_T', 'var') && ~isempty(test_X) && ~isempty(test_T)
            tot = 0;
            for i=1:length(test_X)
                tot = tot + length(test_T{i});
                test_err(iter) = test_err(iter) + sum(viterbi_crf(test_X{i}, mean_model) ~= test_T{i});
            end
            test_err(iter) = test_err(iter) / tot;
            disp(['  - test error: ~' num2str(test_err(iter))]);
        end
    end
    model = mean_model;
    