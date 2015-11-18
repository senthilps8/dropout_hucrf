#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"

using namespace std;


/** SHOULD USE FUNNY MODEL STRUCTURE **/

void forward_backward_linear_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* E_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission);
void forward_backward_hidden_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* E_bias, int no_hidden, const double* labE, const double* labE_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission, double* energy);
void forward_backward_quadr_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* invS, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission);
void forward_backward_pass(const double* emission, int len, const double* P, const double* T, const double* A, int K, double* alpha, double* beta, double* rho);
void log_emission2emission(double* emission, int len, int K);


/**
 *
 * Perform the forward-backward algorithm in the specified CRF.
 *
 *      [alpha, beta, rho, emission, energy] = forward_backward_crf(train_X, model);
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs != 2)
        mexErrMsgTxt("Function should have two inputs.");
    if(!mxIsCell(prhs[0]) && !mxIsDouble(prhs[0]))
        mexErrMsgTxt("First input should be a cell-array or double matrix.");
    if(!mxIsStruct(prhs[1]))
        mexErrMsgTxt("Second input should be a struct.");
    
    /* Get inputs */
    const mxArray* X = prhs[0];
    const mxArray* model = prhs[1];
    const mxArray* tmp = mxGetField(model, 0, "type");
    char* type = (char*) malloc((mxGetN(tmp) + 1) * sizeof(char));
    mxGetString(tmp, type, mxGetN(tmp) + 1);
    bool is_discrete  = mxIsCell(prhs[0]) ? true : false;
    bool is_hidden    = (strcmp(type,  "drbm_continuous") == 0 || strcmp(type,  "drbm_discrete") == 0) ? true : false;
    bool is_quadratic = (strcmp(type, "quadr_continuous") == 0 || strcmp(type, "quadr_discrete") == 0) ? true : false;       // quadr_discrete is not actually supported!!!
    
    /* Get fields and characteristics from model */
    int len = mxGetN(X);
    int K = mxGetM(mxGetField(model, 0, "A"));
    int D = mxGetM(mxGetField(model, 0, "E"));  
    int no_hidden = is_hidden ? mxGetN(mxGetField(model, 0, "E")) : 0;
    double* P         = mxGetPr(mxGetField(model, 0, "pi"));
    double* T         = mxGetPr(mxGetField(model, 0, "tau"));
    double* A         = mxGetPr(mxGetField(model, 0, "A"));
    double* E         = mxGetPr(mxGetField(model, 0, "E"));
    double* labE      = is_hidden ? mxGetPr(mxGetField(model, 0, "labE")) : NULL;
    double* labE_bias = is_hidden ? mxGetPr(mxGetField(model, 0, "labE_bias")) : NULL;
    double* E_bias    = !is_quadratic ? mxGetPr(mxGetField(model, 0, "E_bias")) : NULL;    
    double* invS      =  is_quadratic ? mxGetPr(mxGetField(model, 0, "invS")) : NULL;
    
    /* Run the forward-backward algorithm */
    double* alpha    = (double*) malloc(K * len * sizeof(double));
    double* beta     = (double*) malloc(K * len * sizeof(double));
    double* rho      = (double*) malloc(1 * len * sizeof(double));
    double* emission = (double*) malloc(K * len * sizeof(double));
    double* energy   = is_hidden ? (double*) malloc(len * no_hidden * K * sizeof(double)) : NULL;
    if(is_hidden) {    
        forward_backward_hidden_crf(X, len, D, P, T, A, E, E_bias, no_hidden, labE, labE_bias, K, is_discrete, alpha, beta, rho, emission, energy);
    }
    else {
        if(is_quadratic) forward_backward_quadr_crf(X, len, D, P, T, A, E, invS, K, is_discrete, alpha, beta, rho, emission);
        else             forward_backward_linear_crf(X, len, D, P, T, A, E, E_bias, K, is_discrete, alpha, beta, rho, emission);
    }
    
    /* Copy results back to Matlab */
    if(nlhs > 0) {
        plhs[0] = mxCreateDoubleMatrix(K, len, mxREAL);
        memcpy(mxGetPr(plhs[0]), alpha, K * len * sizeof(double));
    }
    if(nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(K, len, mxREAL);
        memcpy(mxGetPr(plhs[1]), beta, K * len * sizeof(double));
    }
    if(nlhs > 2) {
        plhs[2] = mxCreateDoubleMatrix(1, len, mxREAL);
        memcpy(mxGetPr(plhs[2]), rho, len * sizeof(double));
    }
    if(nlhs > 3) {
        plhs[3] = mxCreateDoubleMatrix(K, len, mxREAL);
        memcpy(mxGetPr(plhs[3]), emission, K * len * sizeof(double));
    }
    if(nlhs > 4 && is_hidden) {
        int dims[] = {len, no_hidden, K};
        plhs[4] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        memcpy(mxGetPr(plhs[4]), energy, K * len * no_hidden * sizeof(double));
    }
    
    /* Clean up memory */
    free(type);
    free(alpha);
    free(beta);
    free(rho);
    free(emission);
    if(energy) free(energy);
}


/* Forward-backward algorithm for linear model without hidden units */
void forward_backward_linear_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* E_bias, int K, bool is_discrete, 
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
    forward_backward_pass(emission, len, P, T, A, K, alpha, beta, rho);
    
}

/* Forward-backward algorithm for model with hidden units */
void forward_backward_hidden_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* E_bias, int no_hidden, const double* labE, const double* labE_bias, int K, bool is_discrete, 
                                                double* alpha, double* beta, double* rho, double* emission, double* energy) {   

    /* Precompute data-hidden potentials */
    double* EX  = (double*) malloc(no_hidden * len * sizeof(double));
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
    for(int k = 0; k < K; k++) {
        for(int j = 0; j < no_hidden; j++) {
            for(int i = 0; i < len; i++) {
                energy[k * no_hidden * len + j * len + i] = labE[j * K + k] + EX[i * no_hidden + j];    // arrangement in energy is not very efficient!
                emission[i * K + k] += log(1.0f + exp(energy[k * no_hidden * len + j * len + i]));
            }
        }
    }
    
    /* Convert to normalized emission probabilities */
    log_emission2emission(emission, len, K);
    
    /* Perform forward and backward pass */
    forward_backward_pass(emission, len, P, T, A, K, alpha, beta, rho);
    
    /* Clean up memory */
    free(EX);
}


/* Compute quadratic emission potentials */
void forward_backward_quadr_crf(const mxArray* X, int len, int D, const double* P, const double* T, const double* A, const double* E, const double* invS, int K, bool is_discrete, 
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
    forward_backward_pass(emission, len, P, T, A, K, alpha, beta, rho);
    
    /* Clean up memory */
    free(zero_mean_X);
}


/* Perform the actual forward-backward pass */
void forward_backward_pass(const double* emission, int len, const double* P, const double* T, const double* A, int K, double* alpha, double* beta, double* rho) {
    
    /* Compute exp(A), and initialize messages */
    double* exp_A = (double*) malloc(K * K * sizeof(double));
    for(int i = 0; i < K * K; i++) exp_A[i] = exp(A[i]);
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
    
    /* Clean up memory */
    free(exp_A);
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
