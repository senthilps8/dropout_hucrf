function model = train_crf(train_X, train_T, type, lambda, max_iter, no_hidden, initialization, optimizer)
%TRAIN_CRF Trains a chain CRF with discrete or linear continuous predictors
%
%   model = train_crf(train_X, train_T, type, lambda, max_iter)
%   model = train_crf(train_X, train_T, model, lambda, max_iter)
%
% The function trains a chain Conditional Random Field (CRF) on the time
% sequences and targets specified in the cell arrays train_X and train_T.
% The number of discrete states / dimensionality of the data is specified
% by D, and the number of possible targets by K. The type is either
% 'discrete', 'continuous', 'drbm_discrete', or 'drbm_continuous. Optionally, 
% an L2 regularizer can be specified through lambda (default = 0). The default
% number of iterations max_iter is 500.
%
% This function differs from TRAIN_CRF_SGD in that it uses L-BFGS to train
% the CRF.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0;
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
        max_iter = 500;
    end  
    if ~exist('initialization', 'var') || isempty(initialization)
        initialization = 'random';
    end
    if ~exist('optimizer', 'var') || isempty(optimizer)
        optimizer = 'lbfgs';
    end
    
    % Compute number of features / dimensionality and number of labels
    if isstruct(type)
        model = type;
        type = model.type;
    end
    if any(strcmpi(type, {'discrete', 'drbm_discrete'}))
        D = 0;
        for i=1:length(train_X)
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
    if ~exist('model', 'var') || isempty(model)
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
        elseif strcmpi(model.type, 'gsc_continuous')
            model.E = randn(D, no_hidden, K) * .001;
        else
            error('Unknown emission function.');
        end
    end
    
    % Pretraining not implemented
    if any(strcmpi(initialization, {'drbm', 'rbm'}))
        error('Pre-training not implemented.');
    end
    
    % Positive part of state priors
    disp('Precomputing positive part of gradient...');
    pos_pi  = zeros(K, 1);
    pos_tau = zeros(K, 1);
    for i=1:length(train_X)
        pos_pi( train_T{i}(1))   = pos_pi( train_T{i}(1))   + 1;
        pos_tau(train_T{i}(end)) = pos_tau(train_T{i}(end)) + 1;
    end
    
    % Positive part of transition gradient (over empirical distribution)
    pos_A = zeros(K, K);
    for i=1:length(train_X)
        for j=2:length(train_T{i})
            pos_A(train_T{i}(j - 1), train_T{i}(j)) = pos_A(train_T{i}(j - 1), train_T{i}(j)) + 1;
        end
    end
    
    % Positive part of emission gradient (over empirical distribution)
    pos_E = zeros(D + 1, K);
    for i=1:length(train_X)
        if strcmpi(model.type, 'discrete')
            for j=1:length(train_T{i})
                pos_E(train_X{i}{j}, train_T{i}(j)) = pos_E(train_X{i}{j}, train_T{i}(j)) + 1;
                pos_E(end, train_T{i}(j)) = pos_E(end, train_T{i}(j)) + 1;
            end
        elseif strcmpi(model.type, 'continuous')
            for k=1:K
                pos_E(:,k) = pos_E(:,k) + [sum(train_X{i}(:,train_T{i} == k), 2); sum(train_T{i} == k)];
            end
        elseif strcmpi(model.type, 'quadr_continuous')
            if i == 1
                pos_E = zeros(numel(model.E) + numel(model.invS), 1);
            end
        elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
            if i == 1
                pos_E = zeros(numel(model.E) + numel(model.labE) + numel(model.E_bias) + numel(model.labE_bias), 1);
            end
        else
            error('Unkown emission function.');
        end
    end

    % Perform minimization using L-BFGS or conjugate gradients
    if strcmpi(optimizer, 'cg')
        disp('Performing optimization using conjugate gradients...');        
    else
        disp('Performing optimization using L-BFGS...');
    end
    addpath('minFunc');
    options.Method = 'lbfgs';
    options.Display = 'on';
    options.TolFun = 1e-2;
    options.TolX = 1e-2;
    options.MaxIter = max_iter;
    if any(strcmpi(model.type, {'discrete', 'continuous'}))
        if strcmpi(optimizer, 'cg')
            x = minimize([model.pi(:); model.tau(:); model.A(:); model.E(:); model.E_bias(:)], 'crf_grad', max_iter, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);        
        else
            x = minFunc(@crf_grad, [model.pi(:); model.tau(:); model.A(:); model.E(:); model.E_bias(:)], options, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);
        end
    elseif strcmpi(model.type, 'quadr_continuous')
        if strcmpi(optimizer, 'cg')
            x = minimize([model.pi(:); model.tau(:); model.A(:); model.E(:); model.invS(:)], 'crf_grad', max_iter, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);        
        else        
            x = minFunc(@crf_grad, [model.pi(:); model.tau(:); model.A(:); model.E(:); model.invS(:)], options, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);
        end
    else
        if strcmpi(optimizer, 'cg')
            x = minimize([model.pi(:); model.tau(:); model.A(:); model.E(:); model.labE(:); model.E_bias(:); model.labE_bias(:)], 'crf_grad', max_iter, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);        
        else
            %keyboard();
            x = minFunc(@crf_grad, [model.pi(:); model.tau(:); model.A(:); model.E(:); model.labE(:); model.E_bias(:); model.labE_bias(:)], options, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E);
        end
    end
    
    % Decode solution
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
    
    % Measure per-frame tagging error (on training set)
    err = 0; tot = 0;
    for i=1:length(train_T)
        tot = tot + length(train_T{i});
        err = err + sum(viterbi_crf(train_X{i}, model) ~= train_T{i});
    end
    disp(['Per-frame error on training set: ' num2str(err / tot)]);
    
