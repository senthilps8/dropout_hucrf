function [alpha, beta, rho, emission, energy] = forward_backward_crf_2nd_order(X, model)
%FORWARD_BACKWARD_CRF_2ND_ORDER Performs forward-backward algorithm in an CRF
%
%   [alpha, beta, rho, emission] = forward_backward_crf_2nd_order(X, model) 
%
% Performs the forward-backward algorithm on time series X in the CRF
% specified in model. The messages are returned in alpha and beta. The
% function also returns the normalization constants of the messages in rho,
% as well as the emission probabilities at all time steps.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Initialize some variables
    N = size(X, 2);
    K = numel(model.pi);
    alpha = zeros(K, N);
    beta  = zeros(K, N);
    rho   = zeros(1, N);
    
    % Compute emission log-probabilities
    if strcmpi(model.type, 'continuous')
        emission = bsxfun(@plus, model.E' * X, model.E_bias');
        emission = exp(bsxfun(@minus, emission, max(emission, [], 1)));
    elseif strcmpi(model.type, 'discrete')
        emission = zeros(K, size(X, 2));
        for j=1:size(X, 2)
            emission(:,j) = sum(model.E(X{j},:), 1)';
        end
        emission = bsxfun(@plus, emission, model.E_bias');
        emission = exp(bsxfun(@minus, emission, max(emission, [], 1)));
    elseif strcmpi(model.type, 'quadr_continuous')
        emission = zeros(K, size(X, 2));
        for k=1:K
            zero_mean_X = bsxfun(@minus, X, model.E(:,k));
            emission(k,:) = sum((zero_mean_X' * model.invS(:,:,k)) .* zero_mean_X', 2);
        end
        emission = exp(bsxfun(@minus, emission, max(emission, [], 1)));
    elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
        no_hidden = size(model.labE, 2);
        if strcmpi(model.type, 'drbm_continuous')
            Ex = bsxfun(@plus, X' * model.E, model.E_bias);
        else
            Ex = zeros(N, no_hidden);
            for j=1:size(X, 2)
                Ex(j,:) = sum(model.E(X{j},:), 1);
            end
            Ex = bsxfun(@plus, Ex, model.E_bias);
        end
        energy = zeros(N, no_hidden, K);
        emission = zeros(N, K);
        for k=1:K
            energy(:,:,k) = bsxfun(@plus, model.labE(k,:), Ex);
            emission(:,k) = model.labE_bias(k) + sum(log(1 + exp(energy(:,:,k))), 2);
        end
        emission = exp(bsxfun(@minus, emission, max(emission, [], 2)))';    % note the transpose!
    else
        error('Unknown emission function.');
    end
    emission = bsxfun(@rdivide, emission, sum(emission, 1));
    
    % Compute message for first hidden variable
    alpha(:,1) = exp(model.pi) .* emission(:,1);
    rho(1) = sum(alpha(:,1)) + realmin;
    alpha(:,1) = alpha(:,1) ./ rho(1);
    
    % Compute message for second hidden variable
    if N > 1
        alpha(:,2) = sum(exp(model.pi2) .* emission(:,2), 1);
        rho(2) = sum(alpha(:,2)) + realmin;
        alpha(:,2) = alpha(:,2) ./ rho(2);
    end    
    
    % Perform forward pass
    exp_A = exp(model.A);
    for n=3:N
        alpha(:,n) = emission(:,n) .* (exp_A' * alpha(:,n - 1));
        alpha(:,n) = emission(:,n) .* sum(sum(bsxfun(@times, exp_A, alpha(:,n - 2) * alpha(:,n - 1)'), 1), 2);
        if n == N, alpha(:,n) = alpha(:,n) .* exp(model.tau); end
        % !!! should do something with model.tau2 !!!
        rho(n) = sum(alpha(:,n)) + realmin;
        alpha(:,n) = alpha(:,n) ./ rho(n);
    end    
    
    % Perform backward pass
    if nargout > 1
        beta(:,N) = 1;
        if N > 1
            beta(:,N - 1) = 1;
        end
        for n=N-1:-1:1
            beta(:,n) = sum(sum(exp_A' .* (beta(:,n + 2) * (beta(:,n + 1) .* emission(:,n + 1))'), 1), 2);
            beta(:,n) = beta(:,n) ./ (sum(beta(:,n)) + realmin);
        end
    end
