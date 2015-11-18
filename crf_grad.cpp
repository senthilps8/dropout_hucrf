#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"

using namespace std;

/** SHOULD USE IMPLEMENTATION IN FORWARD_BACKWARD_CRF.CPP HERE!!! **/

void emission_gradient_linear(const mxArray* X, const double* gamma, int len, int K, int D, double* neg_E, bool is_discrete);
void emission_gradient_hidden(const mxArray* X, const int* T, const double* gamma, const double* energy, int len, int K, int D, int no_hidden, double* neg_E, bool is_discrete);
void emission_gradient_quadr(const mxArray* X, const int* T, const double* gamma, const double* E, const double* invS, int len, int K, int D, double* neg_E, bool is_discrete);
void sum_cond_ll(const int* T, const double* P, const double* PP, const double* A, const double* emission, const double* rho, int K, int len, double* C);
void forward_backward_linear_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* E_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission);
void forward_backward_hidden_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* E_bias, int no_hidden, const double* labE, const double* labE_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission, double* energy);
void forward_backward_quadr_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* invS, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission);
void forward_backward_pass(const double* emission, int len, const double* P, const double* T, const double* exp_A, int K, double* alpha, double* beta, double* rho);
void log_emission2emission(double* emission, int len, int K);


/**
 *
 * Compute the gradient of the CRF conditional log-likelihood.
 *
 *      [C, dC] = crf_grad(x, train_X, train_T, model, lambda, pos_pi, pos_A, pos_E)
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs < 9 || nrhs > 10)
        mexErrMsgTxt("Function should have nine or ten inputs.");
    if(nlhs == 3 && nrhs != 10)
        mexErrMsgTxt("Step size should be specified to get new solution.");        
    if(!mxIsDouble(prhs[0]))
        mexErrMsgTxt("First input should be a double vector.");
    if(!mxIsCell(prhs[1]))
        mexErrMsgTxt("Second input should be a cell-array.");
    if(!mxIsCell(prhs[2]))
        mexErrMsgTxt("Third input should be a cell-array.");    
    if(!mxIsStruct(prhs[3]))
        mexErrMsgTxt("Fourth input should be a struct.");
    if(!mxIsDouble(prhs[4]))
        mexErrMsgTxt("Fifth input should be a double scalar.");    
    if(!mxIsDouble(prhs[5]))
        mexErrMsgTxt("Sixth input should be a double vector.");
    if(!mxIsDouble(prhs[6]))
        mexErrMsgTxt("Seventh input should be a double vector.");
    if(!mxIsDouble(prhs[7]))
        mexErrMsgTxt("Eigth input should be a double matrix.");
    if(!mxIsDouble(prhs[8]))
        mexErrMsgTxt("Ninth input should be a double matrix.");    
    
    /* Get inputs */
    double* x = mxGetPr(prhs[0]);
    const mxArray* train_X = prhs[1];
    const mxArray* train_T = prhs[2];    
    const mxArray* model = prhs[3];
    double lambda = *mxGetPr(prhs[4]);
    double* pos_P = mxGetPr(prhs[5]);
    double* pos_T = mxGetPr(prhs[6]);    
    double* pos_A = mxGetPr(prhs[7]);
    double* pos_E = mxGetPr(prhs[8]);
    double eta = (nrhs == 10) ? *mxGetPr(prhs[9]) : 0.0f;
    const mxArray* tmp = mxGetField(model, 0, "type");
    char* type = (char*) malloc((mxGetN(tmp) + 1) * sizeof(char));
    mxGetString(tmp, type, mxGetN(tmp) + 1);
    bool is_discrete  = (strcmp(type, "discrete") == 0 || strcmp(type, "drbm_discrete") == 0 || strcmp(type, "quadr_discrete") == 0) ? true : false;
    bool is_hidden    = (strcmp(type,  "drbm_continuous") == 0 || strcmp(type,  "drbm_discrete") == 0) ? true : false;
    bool is_quadratic = (strcmp(type, "quadr_continuous") == 0 || strcmp(type, "quadr_discrete") == 0) ? true : false;       // quadr_discrete is not actually supported!!!
    
    /* Get characteristics of model and data */
    int N = mxGetM(train_X) * mxGetN(train_X);
    int K = mxGetM(mxGetField(model, 0, "A"));
    int D = mxGetM(mxGetField(model, 0, "E"));  
    int no_hidden = is_hidden ? mxGetN(mxGetField(model, 0, "E")) : 0;
    int grad_length = mxGetM(prhs[0]) * mxGetN(prhs[0]);
    
    /* Decode current solution x */
    int ind = 0;
    double* P         = x + ind; ind += K;
    double* PP        = x + ind; ind += K;    
    double* A         = x + ind; ind += K * K;
    double* E         = x + ind; if(is_hidden) ind += D * no_hidden; else ind += D * K;
    double* labE      = is_hidden     ? x + ind : NULL; if(is_hidden)     ind += no_hidden * K; 
    double* E_bias    = !is_quadratic ? x + ind : NULL; 
    if(is_hidden) ind += no_hidden;
    else if(!is_quadratic) ind += K;
    double* labE_bias = is_hidden     ? x + ind : NULL; if(is_hidden)     ind += K;
    double* invS      = is_quadratic  ? x + ind : NULL; if(is_quadratic)  ind += D * D * K;
    if(ind > grad_length) mexErrMsgTxt("The solution vector was too short for the specified model.");
    if(ind < grad_length) mexErrMsgTxt("The solution vector was too long for the specified model.");

    /* Initialize the gradient solution */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    if(nlhs > 1) plhs[1] = mxCreateDoubleMatrix(grad_length, 1, mxREAL);
    if(nlhs > 2) plhs[2] = mxCreateDoubleMatrix(grad_length, 1, mxREAL); 
    double* C  = mxGetPr(plhs[0]);
    double* dC = (nlhs > 1) ? mxGetPr(plhs[1]) : NULL;
    double* nx = (nlhs > 2) ? mxGetPr(plhs[2]) : NULL;
    double* neg_P = (nlhs > 1) ? (double*) calloc(K * 1, sizeof(double)) : NULL;
    double* neg_T = (nlhs > 1) ? (double*) calloc(K * 1, sizeof(double)) : NULL;
    double* neg_A = (nlhs > 1) ? (double*) calloc(K * K, sizeof(double)) : NULL;
    double* neg_E = (nlhs > 1) ? (double*) calloc(mxGetM(prhs[8]) * mxGetN(prhs[8]), sizeof(double)) : NULL;
    
    /* Compute exp(A) */
    double* exp_A = (double*) malloc(K * K * sizeof(double));
    for(int i = 0; i < K * K; i++) exp_A[i] = exp(A[i]);
    
    /* Loop over all time series */
    for(int i = 0; i < N; i++) {

        /* Get time series (and get rid of stupid Matlab indexing in labels) */
        mxArray* X = mxGetCell(train_X, i);
        double* matlab_T = mxGetPr(mxGetCell(train_T, i));
        int len = mxGetN(X);
        int* T = (int*) malloc(len * sizeof(int));
        for(int j = 0; j < len; j++) T[j] = (int) matlab_T[j] - 1;
    
        /* Run the forward-backward algorithm */
        double* alpha    = (double*) malloc(K * len * sizeof(double));
        double* beta     = (double*) malloc(K * len * sizeof(double));
        double* rho      = (double*) malloc(1 * len * sizeof(double));
        double* emission = (double*) malloc(K * len * sizeof(double));
        double* energy   = is_hidden ? (double*) malloc(len * no_hidden * K * sizeof(double)) : NULL;
        if(is_hidden) {
            forward_backward_hidden_crf(X, len, D, P, PP, exp_A, E, E_bias, no_hidden, labE, labE_bias, K, is_discrete, alpha, beta, rho, emission, energy);
        }
        else {
            if(is_quadratic) forward_backward_quadr_crf(X, len, D, P, PP, exp_A, E, invS, K, is_discrete, alpha, beta, rho, emission);
            else             forward_backward_linear_crf(X, len, D, P, PP, exp_A, E, E_bias, K, is_discrete, alpha, beta, rho, emission);
        }
            
        /* Sum the value of the cost function */
        sum_cond_ll(T, P, PP, A, emission, rho, K, len, C);
        
        /* Perform the gradient computations */
        if(nlhs > 1) {
            
            /* Compute emission beliefs */
            double* gamma = (double*) malloc(K * len * sizeof(double));
            double sum_gamma;
            for(int j = 0; j < len; j++) {
                sum_gamma = FLT_MIN;
                for(int k = 0; k < K; k++) {
                    gamma[j * K + k] = alpha[j * K + k] * beta[j * K + k];
                    sum_gamma += gamma[j * K + k];
                }
                for(int k = 0; k < K; k++) gamma[j * K + k] /= sum_gamma;
            }     
            
            /* Sum gradients w.r.t. transition parameters */
            for(int k = 0; k < K; k++) neg_P[k] += gamma[k];
            for(int k = 0; k < K; k++) neg_T[k] += gamma[k + (len - 1) * K];
            double* tmp = (double*) malloc(K * K * sizeof(double));
            double sum_tmp, tmp_prod;
            for(int j = 1; j < len; j++) {
                sum_tmp = FLT_MIN;
                for(int h = 0; h < K; h++) {
                    tmp_prod = beta[j * K + h] * emission[j * K + h];
                    for(int k = 0; k < K; k++) {
                        tmp[h * K + k] = exp_A[h * K + k] * alpha[(j - 1) * K + k] * tmp_prod;
                        sum_tmp += tmp[h * K + k];
                    }
                }
                for(int h = 0; h < K * K; h++) neg_A[h] += (tmp[h] / sum_tmp);
            }
            
            /* Compute gradients w.r.t. emission parameters */
            if(is_hidden)         emission_gradient_hidden(X, T, gamma, energy, len, K, D, no_hidden, neg_E, is_discrete);
            else if(is_quadratic) emission_gradient_quadr(X, T, gamma, E, invS, len, K, D, neg_E, is_discrete);
            else                  emission_gradient_linear(X, gamma, len, K, D, neg_E, is_discrete);
            
            /* Clean up memory */
            free(tmp);
            free(gamma);
        }
        
        /* Clean up memory */
        free(T);
        free(alpha);
        free(beta);
        free(rho);
        free(emission);
        if(is_hidden) free(energy);
    }
    
    /* Incorporate regularization in cost function */
    *C = -(*C);
    for(int i = 0; i < grad_length; i++) *C += lambda * (x[i] * x[i]);
    
    /* Construct final gradient */
    if(nlhs > 1) {
        ind = 0;
        for(int i = 0; i < K; i++) dC[i + ind] = neg_P[i] - pos_P[i];
        ind += K;
        for(int i = 0; i < K; i++) dC[i + ind] = neg_T[i] - pos_T[i];
        ind += K;
        for(int i = 0; i < K * K; i++) dC[i + ind] = neg_A[i] - pos_A[i];
        ind += K * K;
        if(is_hidden) {
            for(int i = 0; i < (D + 1) * no_hidden + (no_hidden + 1) * K; i++) dC[i + ind] = -neg_E[i];
            ind += (D + 1) * no_hidden + (no_hidden + 1) * K;
        }
        else if(is_quadratic) {
            for(int i = 0; i < D * K + D * D * K; i++) dC[i + ind] = neg_E[i];
            ind += D * K + D * D * K;
        }
        else {
            for(int i = 0; i < D; i++) {
                for(int j = 0; j < K; j++) {
                    dC[j * D + i + ind] = neg_E[j * D + i] - pos_E[j * (D + 1) + i];            // NOTE: Matrix pos_E has funny ordering!
                }
            }
            ind += D * K;
            for(int i = 0; i < K; i++) dC[i + ind] = neg_E[K * D + i] - pos_E[i * (D + 1) + D]; // NOTE: Matrix pos_E has funny ordering!
            ind += K;
        }
        if(lambda != 0.0f) {
            for(int i = 0; i < grad_length; i++) dC[i] += 2 * lambda * x[i];
        }
    }
    
    /* Compute new solution */
    if(nlhs > 2) {
        for(int i = 0; i < grad_length; i++) nx[i] = x[i] - eta * dC[i];
    }
    
    /* Clean up memory */
    free(exp_A);
    free(type);
    if(nlhs > 1) free(neg_P);
    if(nlhs > 1) free(neg_T);
    if(nlhs > 1) free(neg_A);
    if(nlhs > 1) free(neg_E);
}


/* Performs summing over the linear CRF emission gradient */
void emission_gradient_linear(const mxArray* X, const double* gamma, int len, int K, int D, double* neg_E, bool is_discrete) {
    if(is_discrete) {                                                       // for discrete data
        for(int i = 0; i < len; i++) {
            mxArray* Xi = mxGetCell(X, i);
            double* Xip = mxGetPr(Xi);
            for(int j = 0; j < mxGetM(Xi) * mxGetN(Xi); j++) {
                for(int k = 0; k < K; k++) {
                    neg_E[k * D + (int) Xip[j] - 1] += gamma[i * K + k];
                }
            }
        }
    }
    else {                                                                  // for continuous data
        double* Xp = mxGetPr(X);
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < D; j++) {
                for(int k = 0; k < K; k++) {
                    neg_E[k * D + j] += (gamma[i * K + k] * Xp[i * D + j]);
                }
            }
        }
    }
    
    /* Negative gradient part for the label bias */
    for(int i = 0; i < len; i++) {
        for(int k = 0; k < K; k++) {
            neg_E[K * D + k] += gamma[i * K + k];
        }
    }
}


/* Performs summing over the hidden CRF emission gradients */
void emission_gradient_hidden(const mxArray* X, const int* T, const double* gamma, const double* energy, int len, int K, int D, int no_hidden, double* neg_E, bool is_discrete) {
    
    /* Initialize some memory */
    double* sigmoids            = (double*) malloc(len * no_hidden * K * sizeof(double));
    double* sigmoids_label_post = (double*) malloc(len * no_hidden * K * sizeof(double));
    
    /* Compute sigmoids and sigmoids times gamma */
    for(int i = 0; i < len * no_hidden * K; i++) sigmoids[i] = 1.0f / (1.0f + exp(-energy[i]));
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < no_hidden; j++) {
            for(int k = 0; k < K; k++) {
                sigmoids_label_post[i * K * no_hidden + j * K + k] = 
                           sigmoids[i * K * no_hidden + j * K + k] * gamma[i * K + k];
            }
        }
    }
    
    /* Compute gradient with respect to data-hidden weights */
    int ind = 0;
    if(is_discrete) {                                                       // for discrete data
        for(int i = 0; i < len; i++) {
            mxArray* Xi = mxGetCell(X, i);
            double* Xip = mxGetPr(Xi);
            for(int h = 0; h < no_hidden; h++) {
                for(int j = 0; j < mxGetM(Xi) * mxGetN(Xi); j++) {
                    neg_E[ind + h * D + (int) Xip[j] - 1] += sigmoids[i * K * no_hidden + h * K + T[i]];                    // positive part
                    for(int k = 0; k < K; k++) {
                        neg_E[ind + h * D + (int) Xip[j] - 1] -= sigmoids_label_post[i * K * no_hidden + h * K + k];        // negative part
                    }
                }
            }
        }
    }
    else {                                                                  // for continuous data
        double* Xp = mxGetPr(X);
        for(int i = 0; i < len; i++) {
           for(int h = 0; h < no_hidden; h++) {
                for(int j = 0; j < D; j++) {
                    neg_E[ind + h * D + j] += Xp[i * D + j] * sigmoids[i * K * no_hidden + h * K + T[i]];               // positive part
                    for(int k = 0; k < K; k++) {
                        neg_E[ind + h * D + j] -= Xp[i * D + j] * sigmoids_label_post[i * K * no_hidden + h * K + k];   // negative part
                    }
                }
            }
        }
    }
    
    /* Compute gradient with respect label-hidden weights */
    ind += D * no_hidden;
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < no_hidden; j++) {
            neg_E[ind + j * K + T[i]] += sigmoids[i * K * no_hidden + j * K + T[i]];                                    // positive part
            for(int k = 0; k < K; k++) {
                neg_E[ind + j * K + k] -= sigmoids_label_post[i * K * no_hidden + j * K + k];                           // negative part
            }
        }
    }
    
    /* Compute gradient with respect to bias on hidden units */
    ind += K * no_hidden;
    for(int i = 0; i < len; i++) {            
        for(int j = 0; j < no_hidden; j++) {
            neg_E[ind + j] += sigmoids[i * K * no_hidden + j * K + T[i]];                                               // positive part
            for(int k = 0; k < K; k++) {
                neg_E[ind + j] -= sigmoids_label_post[i * K * no_hidden + j * K + k];                                   // negative part
            }
        }
    }
    
    /* Compute the gradient with respect to bias on labels */
    ind += no_hidden;
    for(int i = 0; i < len; i++) {  
        neg_E[ind + T[i]]++;                                                                                            // positive part
        for(int k = 0; k < K; k++) {
            neg_E[ind + k] -= gamma[i * K + k];                                                                         // negative part
        }
    }

    /* Clean up memory */
    free(sigmoids);
    free(sigmoids_label_post);
}


/* Performs summing over the quadratic CRF emission gradient */
void emission_gradient_quadr(const mxArray* X, const int* T, const double* gamma, const double* E, const double* invS, int len, int K, int D, double* neg_E, bool is_discrete) {
    
    /* Allocate some memory */
    if(is_discrete) mexErrMsgTxt("Discrete inputs are not yet supported for quadratic CRFs.\n");
    double* zero_mean_X = (double*) malloc(len * D * sizeof(double));
    double* diff = (double*) malloc(len * D * sizeof(double));
    double* neg_mE   = (double*) calloc(D * K, sizeof(double));
    double* neg_invS = (double*) calloc(D * D * K, sizeof(double));
    double* Xp = mxGetPr(X);
    
    /* Loop over all components */
    for(int k = 0; k < K; k++) {
        
        /* Subtract mean from data */
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < D; j++) {
                zero_mean_X[i * D + j] = Xp[i * D + j] - E[k * D + j];
            }
        }
        
        /* Compute vector diff */
        double val;
        for(int i = 0; i < len; i++) {
            val = -gamma[i * K + k];
            if(T[i] == k) val++;                
            for(int j = 0; j < D; j++) {
                diff[i * D + j] = val * zero_mean_X[i * D + j];
            }
        }
        
        /* Sum gradient w.r.t. mean */
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < D; j++) {
                for(int h = 0; h < D; h++) {
                    neg_mE[k * D + j] += 2 * invS[k * D * D + j * D + h] * diff[i * D + h];
                }
            }
        }
        
        /* Sum gradient w.r.t. covariance */
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < D; j++) {
                for(int h = 0; h < D; h++) {
                    neg_invS[k * D * D + j * D + h] -= diff[i * D + j] * zero_mean_X[i * D + h];
                }
            }
        }
    }
    
    /* Sum gradient into neg_E */
    int ind = 0;
    for(int i = 0; i < K * D; i++) neg_E[ind + i] += neg_mE[i];
    ind += K * D;
    for(int i = 0; i < K * D * D; i++) neg_E[ind + i] += neg_invS[i];    
    
    /* Clean up memory */
    free(zero_mean_X);
    free(diff);
    free(neg_mE);
    free(neg_invS);
}


/* Performs summing of the cond. log-likelihood in C */
void sum_cond_ll(const int* T, const double* P, const double* PP, const double* A, const double* emission, const double* rho, int K, int len, double* C) {    
    *C += P[T[0]];
    *C += PP[T[len - 1]];
    for(int i = 1; i < len; i++) *C += A[T[i] * K + T[i - 1]];
    for(int i = 0; i < len; i++) *C += log(emission[i * K + T[i]] + FLT_MIN);
    for(int i = 0; i < len; i++) *C -= log(rho[i] + FLT_MIN);
}

/* Forward-backward algorithm for linear model without hidden units */
void forward_backward_linear_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* E_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission) {
    
    /* Compute emission log-probabilities */
    if(is_discrete) {                                                       // for discrete data
        for(int i = 0; i < len; i++) {
            mxArray* Xi = mxGetCell(X, i);
            double* Xip = mxGetPr(Xi);
            for(int k = 0; k < K; k++) {
                emission[i * K + k] = E_bias[k];
                for(int j = 0; j < mxGetM(Xi) * mxGetN(Xi); j++) {
                    emission[i * K + k] += E[k * D + (int) Xip[j] - 1];            
                }
            }
        }
    }
    else {                                                                  // for continuous data
        double* Xp = mxGetPr(X);
        for(int i = 0; i < len; i++) {
            for(int k = 0; k < K; k++) {                
                emission[i * K + k] = E_bias[k];
                for(int j = 0; j < D; j++) {
                    emission[i * K + k] += E[k * D + j] * Xp[i * D + j];              
                }
            }
        }
    }
    
    /* Convert to normalized emission probabilities */
    log_emission2emission(emission, len, K);
    
    /* Perform forward and backward pass */
    forward_backward_pass(emission, len, P, T, exp_A, K, alpha, beta, rho);    
}

/* Forward-backward algorithm for model with hidden units */
void forward_backward_hidden_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* E_bias, int no_hidden, const double* labE, const double* labE_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission, double* energy) {   

    /* Precompute data-hidden potentials */
    double* EX = (double*) malloc(no_hidden * len * sizeof(double));
    if(is_discrete) {                                                       // for discrete data
        for(int i = 0; i < len; i++) {
            mxArray* Xi = mxGetCell(X, i);
            double* Xip = mxGetPr(Xi);
            for(int j = 0; j < no_hidden; j++) {                
                EX[i * no_hidden + j] = E_bias[j];
                for(int g = 0; g < mxGetM(Xi) * mxGetN(Xi); g++) {
                    EX[i * no_hidden + j] += E[j * D + (int) Xip[g] - 1];            
                }
            }
        }
    }
    else {                                                                  // for continuous data
        double* Xp = mxGetPr(X);
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < no_hidden; j++) {                
                EX[i * no_hidden + j] = E_bias[j];
                for(int g = 0; g < D; g++) {
                    EX[i * no_hidden + j] += E[j * D + g] * Xp[i * D + g];
                }
            }
        }
    }
    
    /* Compute unnormalized emission log-probabilities */
    for(int i = 0; i < len; i++) {
        for(int k = 0; k < K; k++) {
            emission[i * K + k] = labE_bias[k];
        }
    }
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < no_hidden; j++) {
            for(int k = 0; k < K; k++) {
                energy[i * K * no_hidden + j * K + k] = labE[j * K + k] + EX[i * no_hidden + j];
                emission[i * K + k] += log(1.0f + exp(energy[i * K * no_hidden + j * K + k]));
            }
        }
    }
    
    /* Convert to normalized emission probabilities */
    log_emission2emission(emission, len, K);
    
    /* Perform forward and backward pass */
    forward_backward_pass(emission, len, P, T, exp_A, K, alpha, beta, rho);
    
    /* Clean up memory */
    free(EX);
}


/* Compute quadratic emission potentials */
void forward_backward_quadr_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* exp_A, const double* E, const double* invS, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission) {
    
    /* Loop over labels */
    double* Xp = mxGetPr(X);
    double* zero_mean_X = (double*) malloc(D * len * sizeof(double));
    for(int k = 0; k < K; k++) {
        
        /* Compute X - mean */
        for(int j = 0; j < len; j++) {
            for(int g = 0; g < D; g++) {
                zero_mean_X[j * D + g] = Xp[j * D + g] - E[k * D + g];
            }
        }
        
        /* Compute emission log-probability */
        for(int j = 0; j < len; j++) {
            emission[j * K + k] = 0.0f;
            for(int g = 0; g < D; g++) {
                double value = 0.0f;
                for(int h = 0; h < D; h++) {
                    value += zero_mean_X[j * D + h] * invS[k * D * D + g * D + h];
                }
                emission[j * K + k] += zero_mean_X[j * D + g] * value;
            }
        }
    }
    
    /* Convert to normalized emission probabilities */
    log_emission2emission(emission, len, K);
    
    /* Perform forward and backward pass */
    forward_backward_pass(emission, len, P, T, exp_A, K, alpha, beta, rho);
    
    /* Clean up memory */
    free(zero_mean_X);
}


/* Perform the actual forward-backward pass */
void forward_backward_pass(const double* emission, int len, const double* P, const double* T, const double* exp_A, int K, double* alpha, double* beta, double* rho) {
    
    /* Compute exp(A), and initialize messages */
    for(int i = 0; i < K * len; i++) alpha[i] = 0.0f;   
    for(int i = 0; i < K * (len - 1); i++) beta[i] = 0.0f;
    for(int k = 0; k < K; k++) beta[(len - 1) * K + k] = 1.0f;
    for(int i = 0; i < len; i++) rho[i] = 0.0f;
    
    /* Compute message for first hidden variable */
    for(int k = 0; k < K; k++) {
        alpha[k] = exp(P[k]) * emission[k];
        rho[0] += alpha[k];
    }
    for(int k = 0; k < K; k++) {
        alpha[k] /= rho[0];
    }
    
    /* Perform forward pass */
    for(int i = 1; i < len; i++) {
        for(int k = 0; k < K; k++) {
            for(int h = 0; h < K; h++) {
                alpha[i * K + k] += exp_A[k * K + h] * alpha[(i - 1) * K + h];
            }
            alpha[i * K + k] *= emission[i * K + k];   
            if(i == len - 1) alpha[i * K + k] *= exp(T[k]);
            rho[i] += alpha[i * K + k];
        }
        for(int k = 0; k < K; k++) {
            alpha[i * K + k] /= rho[i];
        }
    }   
    
    /* Perform backward pass */
    for(int i = len - 2; i >= 0; i--) {
        for(int k = 0; k < K; k++) {
            double val = beta[(i + 1) * K + k] * emission[(i + 1) * K + k];
            for(int h = 0; h < K; h++) {
                beta[i * K + h] += exp_A[k * K + h] * val;
            }
        }
        double sum_beta = beta[i * K];
        for(int k = 1; k < K; k++) sum_beta += beta[i * K + k];
        for(int k = 0; k < K; k++) beta[i * K + k] /= sum_beta;             // could divide by K * rho[i + 1]?
    }   
}


/* Convert log-probabilities to normalized probabilities */
void log_emission2emission(double* emission, int len, int K) {
    
    /* Perform the conversion */
    for(int i = 0; i < len; i++) {
        double sum_emission = 0.0f;
        double max_emission = emission[i * K];
        for(int k = 1; k < K; k++) {
            if(emission[i * K + k] > max_emission) max_emission = emission[i * K + k];
        }
        for(int k = 0; k < K; k++) {
            emission[i * K + k] = exp(emission[i * K + k] - max_emission);
            sum_emission += emission[i * K + k];
        }
        for(int k = 0; k < K; k++) {
            emission[i * K + k] /= sum_emission;
        }
    }
}
