function [sequence, L, Z] = viterbi_hidden_crf_2nd_order(X, model, T, rho, EX)
%VITERBI_HIDDEN_CRF_2ND_ORDER Runs Viterbi to find most likely state sequence
%
%   [sequence, L, Z] = viterbi_hidden_crf_2nd_order(X, model)
%   [sequence, L, Z] = viterbi_hidden_crf_2nd_order(X, model, T, rho, EX)
%
% Performs the Viterbi algorithm on time series X in the hidden unit CRF
% specified in model, to find the most likely underlying state sequence.
% Optionally, the data-hidden potentials can be specified in EX.
% The corresponding hidden unit states are returned in Z. The log-likelihood 
% of the sequence is returned in L.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    % Initialize some variables
    N = size(X, 2);
    K = numel(model.pi);
    no_hidden = size(model.E, 2);
    ind = zeros(K, K, N);
    sequence = zeros(1, N);
    if N == 0
        L = [];
        return;
    end
    
    % Precompute data-hidden potentials
    if ~exist('EX', 'var') || isempty(EX)
        if strcmpi(model.type, 'drbm_continuous')
            EX = bsxfun(@plus, model.E' * X, model.E_bias');
        elseif strcmpi(model.type, 'drbm_discrete')
            EX = zeros(no_hidden, length(X));
            for j=1:length(X)
                EX(:,j) = sum(model.E(X{j},:), 1)';
            end
            EX = bsxfun(@plus, EX, model.E_bias');
        end
    end
    
    % Compute emission log-probabilities
    emission = zeros(K, N);
    hidden = repmat(false, [no_hidden N K]);
    for k=1:K
        message = bsxfun(@plus, EX, model.labE(k,:)');
        hidden(:,:,k) = (message > 0);
        emission(k,:) = sum(message .* hidden(:,:,k), 1) + model.labE_bias(k);
    end
    
    % Add margin constraint to emissions
    if exist('T', 'var') && exist('rho', 'var') && ~isempty(T) && ~isempty(rho)
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
    
    % Construct matrix with hidden unit states
    if nargout > 2
        Z = repmat(false, [no_hidden N]);
        for n=1:N
            Z(:,n) = hidden(:,n,sequence(n));
        end
    end
