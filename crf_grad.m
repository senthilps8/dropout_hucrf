function [C, dC, x] = crf_grad(x, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E, eta)
%CRF_GRAD Compute CRF conditional log-likelihood and gradient
%
%   [C, dC, x] = crf_grad(x, train_X, train_T, model, lambda, pos_pi, pos_tau, pos_A, pos_E, eta)
%
% Compute negative conditional log-likelihood C and the corresponding
% gradient on the specified training time series (train_X, train_T). The
% CRF model and positive parts of the gradient must also be specified.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('eta', 'var') || isempty(eta)
        eta = 0;
    end
    
    % Decode current solution
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
    K = length(model.pi);
    D = size(model.E, 1);
    % Changes - avemula
    H = size(model.E, 2); % Number of hidden states in the model
    % Initialize negative sums
    L = 0;
    if nargout > 1
        neg_pi  = zeros(K, 1);
        neg_tau = zeros(K, 1);
        neg_A   = zeros(K, K);
        neg_E   = zeros(size(model.E));
        if strcmpi(model.type, 'quadr_continuous')
            neg_invS = zeros(D, D, K);
        end
        if any(strcmpi(model.type, {'continuous', 'discrete', 'drbm_discrete', 'drbm_continuous'}))
            neg_E_bias = zeros(size(model.E_bias));
        end
        if any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
            neg_labE      = zeros(size(model.labE));
            neg_labE_bias = zeros(size(model.labE_bias));
        end
        exp_A = exp(model.A);
    end
        
    back_model = model;
    % Loop over training sequences
    for i=1:length(train_X)
        
        % Randomly generate the vector r of binary random variables
        % according to a bernoulli distribution
        r = rand(H, 1); % Random vector of size H
        r = r < 0.5; % Use a coin toss distribution for now
        
	model.E = back_model.E(:,r);
	model.labE = back_model.labE(:,r);
	model.E_bias = back_model.E_bias(:,r);
        % Drop columns of E, labE for which r has value 0
        %model.E 
	
	%TODO: Drop Column from model.(E, labE, E_bias)
	
        % Perform forward-backward algorithm for CRFs
        if any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
            [alpha, beta, rho, emission, energy] = forward_backward_crf(train_X{i}, model);
        else
            [alpha, beta, rho, emission] = forward_backward_crf(train_X{i}, model);
        end
        
	%TODO: Check dimensions of alpha, beta, rho, emission, energy
	%TODO: Check if LogLL can be computed without changes
	
        % Sum conditional log-likelihood         
        L = L + model.pi( train_T{i}(1));
        L = L + model.tau(train_T{i}(end));
        for j=2:length(train_T{i})
            L = L + model.A(train_T{i}(j - 1), train_T{i}(j));
        end
        for j=1:length(train_T{i})
            L = L + log(emission(train_T{i}(j), j) + realmin);
        end
        logZ = sum(log(rho + realmin));
        L = L - logZ;
        
        % Only compute gradient if required
        if nargout > 1
            
            % Compute emission beliefs (per-frame posteriors)           
            gamma = alpha .* beta;
            gamma = bsxfun(@rdivide, gamma, sum(gamma, 1) + realmin);
            
	    %TODO: Check dimensions of neg_pi, neg_tau, neg_A
            % Sum gradient with respect to transition factors
            neg_pi  = neg_pi  + gamma(:,1);
            neg_tau = neg_tau + gamma(:,end);
            for j=2:length(train_T{i})
                tmp = exp_A .* (alpha(:,j - 1) * (beta(:,j) .* emission(:,j))');
                neg_A = neg_A + tmp ./ (sum(tmp(:)) + realmin);
            end

            % Sum gradient with respect to emission factors
            if strcmpi(model.type, 'discrete')
                warning('Use the C++ version of CRF_GRAD for better performance on discrete data.');
                for j=1:length(train_T{i})                    
                    neg_E(train_X{i}{j},:) = bsxfun(@plus, neg_E(train_X{i}{j},:), gamma(:,j)');
                end
                neg_E_bias = neg_E_bias + sum(gamma, 2)';
            elseif strcmpi(model.type, 'continuous')
                neg_E = neg_E + train_X{i} * gamma';
                neg_E_bias = neg_E_bias + sum(gamma, 2)';
            elseif strcmpi(model.type, 'quadr_continuous')
                for k=1:K
                    zero_mean_X = bsxfun(@minus, train_X{i}, model.E(:,k));                    
                    diff = bsxfun(@times, (train_T{i} == k) - gamma(k,:), zero_mean_X);
                    neg_E(:,k) = neg_E(:,k) + 2 * sum(model.invS(:,:,k) * diff, 2);
                    neg_invS(:,:,k) = neg_invS(:,:,k) - diff * zero_mean_X';
                end
            elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
                
		%TODO: Check if sigmoids_label_post can be computed
                % Precompute sigmoids of energies, and products with label posterior
                gamma = gamma';
                sigmoids = 1 ./ (1 + exp(-energy));
                sigmoids_label_post = zeros(size(sigmoids));
                for k=1:K
                    sigmoids_label_post(:,:,k) = bsxfun(@times, sigmoids(:,:,k), gamma(:,k)); 
                end
		
		%TODO: Modify size of sigmoids_label_post and sigmoids
    		
		%TODO: Check dimensions of neg_E (should be DxH-1)
                % Sum gradient with respect to the data-hidden weights                
                if strcmpi(model.type, 'drbm_continuous')
		    keyboard()
                    for k=1:K
                        neg_E = neg_E + train_X{i} * (bsxfun(@times, (train_T{i} == k)', sigmoids(:,:,k)) - sigmoids_label_post(:,:,k));
                    end
                elseif strcmpi(model.type, 'drbm_discrete')
                %    warning('Use the C++ version of CRF_GRAD for better performance on discrete data.');                
                %    for k=1:K
                %        ind = find(train_T{i} == k);
                %        for j=1:length(ind)
                %            neg_E(train_X{i}{ind(j)},:) = bsxfun(@plus, neg_E(train_X{i}{ind(j)},:), sigmoids(ind(j),:,k));
                %        end                                        
                %    end
                %    for k=1:K       % NOTE: This can be done more efficiently!
                %        for j=1:length(train_T{i})
                %            neg_E(train_X{i}{j},:) = bsxfun(@minus, neg_E(train_X{i}{j},:), sigmoids_label_post(j,:,k));
                %        end
                %    end
                end

		%TODO: Check dimension of neg_labE (should be KxH-1)
                % Compute gradient with respect to label-hidden weights
                for k=1:K          % NOTE: This can be done more efficiently!
                    neg_labE(k,:) = neg_labE(k,:) + sum(sigmoids(train_T{i} == k,:,k), 1) - ...
                                                    sum(sigmoids_label_post(:,:,k), 1);
                end
		
		%TODO: Check dimension of neg_E_bias (should be 1xH-1)
                % Compute gradient with respect to bias on hidden units
                for k=1:K
                    neg_E_bias = neg_E_bias + sum(sigmoids(train_T{i} == k,:,k), 1) - ...
                                              sum(sigmoids_label_post(:,:,k), 1);
                end

                % Compute the gradient with respect to bias on labels
                label_matrix = zeros(length(train_T{i}), K);
                label_matrix(sub2ind([length(train_T{i}) K], 1:length(train_T{i}), train_T{i})) = 1;
                neg_labE_bias = neg_labE_bias + sum(label_matrix - gamma, 1);
		%TODO: Change dimensions of neg_E_bias, neg_labE, neg_E
            else
                error('Unknown type.');
            end
        end
    end

    % Return cost function and gradient
    C = -L + lambda .* sum(x .^ 2);
    if nargout > 1
        if any(strcmpi(model.type, {'continuous', 'discrete'}))
            pos_E_bias = pos_E(end,:);
            pos_E = pos_E(1:end - 1,:);            
            pos_E = [pos_E(:); pos_E_bias(:)];
            neg_E = [neg_E(:); neg_E_bias(:)];
        elseif strcmpi(model.type, 'quadr_continuous')
            neg_E = [neg_E(:); neg_invS(:)];
        elseif any(strcmpi(model.type, {'drbm_discrete', 'drbm_continuous'}))
            neg_E = -[neg_E(:); neg_labE(:); neg_E_bias(:); neg_labE_bias(:)];
        end
        dC = -[pos_pi(:)  - neg_pi(:);  ...
               pos_tau(:) - neg_tau(:); ...
               pos_A(:)   - neg_A(:);   ...
               pos_E(:)   - neg_E(:)] + 2 .* lambda .* x;   
    end
    
    % Return new solution
    if nargout > 2
        x = x - eta * dC;
    end
