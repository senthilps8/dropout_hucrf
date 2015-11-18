function mexall(debugging)
%MEXALL Compiles all MEX-files of hidden-unit CRF implementation
%
%   mexall
%
% Compiles all MEX-files of hidden-unit CRF implementation. 
% NOTE: Use MEX-implementations for discrete data only!
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


    if ~exist('debugging', 'var') || isempty(debugging)
        debugging = false;
    end

    if debugging
        warning('Compiling debug versions. Compile release versions for better performance!');
        mex -g crf_herding.cpp util.cpp
        mex -g crf_herding_2nd_order.cpp util.cpp
        mex -g hidden_crf_herding.cpp util.cpp 
        mex -g hidden_crf_herding_2nd_order.cpp util.cpp 
        mex -g forward_backward_crf.cpp
        mex -g viterbi_crf.cpp util.cpp 
        mex -g viterbi_hidden_crf.cpp util.cpp
        mex -g viterbi_crf_2nd_order.cpp util.cpp
        mex -g viterbi_hidden_crf_2nd_order.cpp util.cpp
        mex -g crf_grad.cpp
    else
        mex -O crf_herding.cpp util.cpp
        mex -O crf_herding_2nd_order.cpp util.cpp
        mex -O hidden_crf_herding.cpp util.cpp 
        mex -O hidden_crf_herding_2nd_order.cpp util.cpp 
        mex -O forward_backward_crf.cpp
        mex -O viterbi_crf.cpp util.cpp 
        mex -O viterbi_hidden_crf.cpp util.cpp
        mex -O viterbi_crf_2nd_order.cpp util.cpp
        mex -O viterbi_hidden_crf_2nd_order.cpp util.cpp
        mex -O crf_grad.cpp
    end