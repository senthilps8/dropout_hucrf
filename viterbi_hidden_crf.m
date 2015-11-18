function [sequence, L, posterior, Z] = viterbi_hidden_crf(X, model, T, rho, EX)
%VITERBI_HIDDEN_CRF Runs Viterbi to find most likely state sequence
%
%   [sequence, L, posterior, Z] = viterbi_hidden_crf(X, model)
%   [sequence, L, posterior, Z] = viterbi_hidden_crf(X, model, T, rho, EX)
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
    omega = zeros(K, N);
    ind = zeros(K, N);
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
    if exist('T', 'var') && exist('rho', 'var') && ~isempty(T) && ~isempty(rho) && rho > 0
        ii = sub2ind(size(emission), T, 1:length(T));
        emission = emission + rho;
        emission(ii) = emission(ii) - rho;
    end
    
    % Compute message for first hidden variable
    omega(:,1) = model.pi + emission(:,1);
    
    % Perform forward pass
    for n=2:N
        [omega(:,n), ind(:,n)] = max(bsxfun(@plus, model.A, omega(:,n - 1)), [], 1);
        omega(:,n) = omega(:,n) + emission(:,n);
    end
    
    % Add message for last hidden variable
    omega(:,N) = omega(:,N) + model.tau;
    
    % Perform backtracking to determine final sequence
    [L, sequence(N)] = max(real(omega(:,N)));
    for n=N - 1:-1:1
        sequence(n) = ind(sequence(n + 1), n + 1);
    end
    
    % Compute per-frame posterior
    if nargout > 2        
        posterior = exp(bsxfun(@minus, omega, max(omega, [], 1)));
        posterior = bsxfun(@rdivide, posterior, sum(posterior, 1) + realmin);
    end
    
    % Construct matrix with hidden unit states
    if nargout > 3
        Z = repmat(false, [no_hidden N]);
        for n=1:N
            Z(:,n) = hidden(:,n,sequence(n));
        end
    end
