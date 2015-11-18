function [sequence, L] = viterbi_crf_2nd_order(X, model, T, rho)
%VITERBI_CRF_2ND_ORDER Performs Viterbi algorithm in second-order chain
%
%   [sequence, L] = viterbi_crf_2nd_order(X, model)
%   [sequence, L] = viterbi_crf_2nd_order(X, model, T, rho)
%
% Performs the Viterbi algorithm on time series X in the Conditional Random
% Field (CRF) specified in model, to find the most likely underlying state 
% sequence. The log-likelihood of the sequence is returned in L.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Initialize some variables
    N = size(X, 2);
    K = size(model.pi, 1);
    ind = zeros(K, K, N);
    sequence = zeros(1, N);
    
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
            Ex = bsxfun(@plus, X' * model.E, model.E_bias);
        else
            Ex = zeros(N, no_hidden);
            for n=1:N
                Ex(n,:) = sum(model.E(X{n},:), 1);
            end
            Ex = bsxfun(@plus, Ex, model.E_bias);
        end
        energy = zeros(N, no_hidden, K);
        emission = zeros(N, K);
        for k=1:K
            lab = zeros(1, K); lab(k) = 1;
            energy(:,:,k) = bsxfun(@plus, lab * model.labE, Ex);
            emission(:,k) = model.labE_bias(k) + sum(log(1 + exp(energy(:,:,k))), 2);
        end
        emission = exp(bsxfun(@minus, emission, max(emission, [], 2)));
        emission = log(bsxfun(@rdivide, emission, sum(emission, 2))' + realmin);
    else
        error('Unknown emission function.');
    end    
    
    % Add margin constraint to emissions
    if exist('T', 'var') && exist('rho', 'var') && ~isempty(T) && ~isempty(rho) && rho > 0
        ii = sub2ind(size(emission), T, 1:length(T));
        emission = emission + rho;
        emission(ii) = emission(ii) - rho;
    end
    
    % Compute message for first two state variables
    omega = model.pi + emission(:,1);
    if N > 1
        omega = bsxfun(@plus, bsxfun(@plus, model.pi2, omega), emission(:,2)');
    end
    
    % Perform forward pass
    for n=3:N
        [omega, ind(:,:,n)] = max(bsxfun(@plus, model.A, omega), [], 1);    % max over variable n - 2
        omega = bsxfun(@plus, squeeze(omega), emission(:,n)');
    end
    
    % Add message for last hidden variable
    if N > 1
        omega = bsxfun(@plus, omega, model.tau');
        omega = omega + model.tau2;
    else
        omega = omega + model.tau;
    end
    
    % Perform backtracking to determine final sequence
    if N > 1
        [L, ii] = max(omega(:));                                            % max over variable N and N - 1
        [sequence(N - 1), sequence(N)] = ind2sub(size(omega), ii);
    else
        [L, sequence(N)] = max(omega, [], 1);
    end
    for n=N - 2:-1:1
        sequence(n) = ind(sequence(n + 1), sequence(n + 2), n + 2);
    end
