function [sequence, L, posterior] = viterbi_crf(X, model, T, rho)
%VITERBI_CRF Performs Viterbi algorithm to find most likely state sequence
%
%   [sequence, L, posterior] = viterbi_crf(X, model)
%   [sequence, L, posterior] = viterbi_crf(X, model, T, rho)
%
% Performs the Viterbi algorithm on time series X in the Conditional Random
% Field (CRF) specified in model, to find the most likely underlying state 
% sequence. The log-likelihood of the sequence is returned in L, and the
% per-frame posterior in posterior.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Initialize some variables
    N = size(X, 2);
    K = numel(model.pi);
    omega = zeros(K, N);
    ind = zeros(K, N);
    sequence = zeros(1, N);
    if N == 0
        L = [];
        return;
    end
    
    % Compute emission log-probabilities
    if strcmpi(model.type, 'continuous')
        emission = bsxfun(@plus, model.E' * X, model.E_bias');
    elseif strcmpi(model.type, 'discrete')
        emission = zeros(K, N);
        for n=1:N
            emission(:,n) = sum(model.E(X{n},:), 1)';
        end
        emission = bsxfun(@plus, emission, model.E_bias');
    elseif strcmpi(model.type, 'quadr_continuous')
        emission = zeros(K, size(X, 2));
        for k=1:K
            zero_mean_X = bsxfun(@minus, X, model.E(:,k));
            emission(k,:) = sum((zero_mean_X' * model.invS(:,:,k)) .* zero_mean_X', 2);
        end
    elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
        no_hidden = size(model.labE, 2);
        if strcmpi(model.type, 'drbm_continuous')
            EX = bsxfun(@plus, X' * model.E, model.E_bias);
        else
            EX = zeros(N, no_hidden);
            for n=1:N
                EX(n,:) = sum(model.E(X{n},:), 1);
            end
            EX = bsxfun(@plus, EX, model.E_bias);
        end
        emission = zeros(N, K);
        for k=1:K
            lab = zeros(1, K); lab(k) = 1;
            emission(:,k) = model.labE_bias(k) + sum(log(1 + exp(bsxfun(@plus, lab * model.labE, EX))), 2);
        end
        emission = emission';
    elseif strcmpi(model.type, 'gsc_continuous')
        emission = zeros(K, N);
        for k=1:K
            emission(k,:) = prod(1 + exp(X' * model.E(:,:,k)), 2)';
        end
    else
        error('Unknown emission function.');
    end    
    
    % Add margin constraint to emissions
    if exist('T', 'var') && exist('rho', 'var') && ~isempty(T) && ~isempty(rho) && rho > 0
        ii = sub2ind(size(emission), T, 1:length(T));
        emission = emission + rho;
        emission(ii) = emission(ii) - rho;
    end

    % Compute message for first hidden variable
    omega(:,1) = model.pi + emission(:,1);
    
    % Perform forward pass
    for n=2:N
        [omega(:,n), ind(:,n)] = max(bsxfun(@plus, model.A, omega(:,n - 1)), [], 1);    % the max is over the variable n-1
        omega(:,n) = omega(:,n) + emission(:,n);
    end
    
    % Add message for last hidden variable
    omega(:,N) = omega(:,N) + model.tau;
    
    % Perform backtracking to determine final sequence
    [L, sequence(N)] = max(real(omega(:,N)));                                           % the max is over variable N
    for n=N - 1:-1:1
        sequence(n) = ind(sequence(n + 1), n + 1);
    end
    
    % Compute marginal posteriors
    if nargout > 2
        posterior = exp(bsxfun(@minus, omega, max(omega, [], 1)));
        posterior = bsxfun(@rdivide, posterior, sum(posterior, 1) + realmin);
    end
