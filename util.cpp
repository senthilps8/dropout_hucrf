#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"
#include "util.h"

using namespace std;


void linear_emission(const mxArray* X, int len, model* crf, bool is_discrete, double* emission) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
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
}


void hidden_emission(const mxArray* X, int len, model* crf, bool is_discrete, double* EX, double* emission) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Precompute data-hidden potentials */
    double* energy = (double*) malloc(len * no_hidden * K * sizeof(double));
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
    
    /* Clean up memory */
    free(energy);
}


void quadr_emission(const mxArray* X, int len, model* crf, bool is_discrete, double* emission) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
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
    free(zero_mean_X);
}

void viterbi_linear_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L) {

    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Compute emission log-probabilities */
    double* emission = (double*) malloc(K * len * sizeof(double));
    linear_emission(X, len, crf, is_discrete, emission);  
    
    /* Run Viterbi decoder */
    viterbi_crf_2nd_order(emission, crf, T, rho, len, pred_T, L);
    free(emission);
}


void viterbi_linear_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Compute emission log-probabilities */
    double* emission = (double*) malloc(K * len * sizeof(double));
    linear_emission(X, len, crf, is_discrete, emission);  
    
    /* Run Viterbi decoder */
    viterbi_crf(emission, crf, T, rho, len, pred_T, L, omega);
    free(emission);
}

void viterbi_hidden_crf(const mxArray* X, int len, model* crf, bool is_discrete, int* pred_T) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Run Viterbi */
    double* T = NULL; 
    double rho = 0.0f; 
    double L = 0.0f;
    vector<bool> Z(no_hidden * len);
    double* EX  = (double*) malloc(no_hidden * len * sizeof(double));
    double* omega = (double*) malloc(K * len * sizeof(double));    
    viterbi_hidden_crf(X, T, rho, len, crf, is_discrete, pred_T, &L, omega, EX);
    free(EX);
    free(omega);
}

void viterbi_hidden_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega, double* EX) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Compute emission potentials */
    double* emission = (double*) malloc(len * K * sizeof(double));
    hidden_emission(X, len, crf, is_discrete, EX, emission);
    
    /* Run Viterbi decoder */
    viterbi_crf(emission, crf, T, rho, len, pred_T, L, omega);
    free(emission);   
}

void viterbi_hidden_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Run Viterbi */
    double* EX  = (double*) malloc(no_hidden * len * sizeof(double));
    viterbi_hidden_crf_2nd_order(X, T, rho, len, crf, is_discrete, pred_T, L, EX);
    free(EX);
}

void viterbi_hidden_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* EX) {

    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Compute emission potentials */
    double* emission = (double*) malloc(len * K * sizeof(double));
    hidden_emission(X, len, crf, is_discrete, EX, emission);
    
    /* Run Viterbi decoder */
    viterbi_crf_2nd_order(emission, crf, T, rho, len, pred_T, L);
    free(emission);   
}


void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, int* pred_T, bool is_discrete) {
    
    /* Get variables from model */
    int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Running Viterbi */
    vector<bool> Z(len * no_hidden);
    double* omega = (double*) malloc(K * len * sizeof(double));
    double* EX = (double*) malloc(len * no_hidden * sizeof(double));
    double* T = (double*) malloc(len * sizeof(double));
    joint_viterbi_hidden_crf(X, T, len, crf, pred_T, EX, &Z, omega, 0.0f, is_discrete);
    free(T);
    free(EX);
    free(omega);
}

void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, int* pred_T, double* EX, vector<bool>* Z, double* omega, bool is_discrete) {
    
    /* Get variables from model */
    int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden;
    
    /* Running Viterbi */
    double rho = -1.0f;
    double* T = NULL;
    joint_viterbi_hidden_crf(X, T, len, crf, pred_T, EX, Z, omega, rho, is_discrete);
}


/* This function assumes EX is given */
void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, const double* EX, int* pred_T, vector<bool>* Z, double* omega, bool is_discrete) {
    
    /* Get variables from model */
    int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden;
    
    /* Running Viterbi */
    double rho = -1.0f;
    double* T = NULL;
    joint_viterbi_hidden_crf(X, T, len, crf, EX, pred_T, Z, omega, rho, is_discrete);
}

/* This function computes EX */
void joint_viterbi_hidden_crf(const mxArray* X, const double* T, int len, model* crf, int* pred_T, double* EX, vector<bool>* Z, double* omega, double rho, bool is_discrete) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E1 = crf->E; double* E1_bias = crf->E_bias; double* E2 = crf->labE; double* E2_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Precompute data-hidden potentials */
    if(is_discrete) {                                                       // for discrete data
        for(int i = 0; i < len; i++) {
            mxArray* Xi = mxGetCell(X, i);
            double* Xip = mxGetPr(Xi);
            for(int j = 0; j < no_hidden; j++) {                
                EX[i * no_hidden + j] = E1_bias[j];
                for(int g = 0; g < mxGetM(Xi) * mxGetN(Xi); g++) {
                    EX[i * no_hidden + j] += E1[j * D + (int) Xip[g] - 1];            
                }
            }
        }
    }
    else {                                                                  // for continuous data
        double* Xp = mxGetPr(X);
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < no_hidden; j++) {                
                EX[i * no_hidden + j] = E1_bias[j];
                for(int g = 0; g < D; g++) {
                    EX[i * no_hidden + j] += E1[j * D + g] * Xp[i * D + g];
                }
            }
        }
    }
    
    /* Perform Viterbi decoding */
    joint_viterbi_hidden_crf(X, T, len, crf, EX, pred_T, Z, omega, rho, is_discrete);
}
    
/* This function assumes EX is given */
void joint_viterbi_hidden_crf(const mxArray* X, const double* T, int len, model* crf, const double* EX, int* pred_T, vector<bool>* Z, double* omega, double rho, bool is_discrete) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E1 = crf->E; double* E1_bias = crf->E_bias; double* E2 = crf->labE; double* E2_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; bool second_order = crf->second_order;
    
    /* Allocate some memory */
    double* emiss = (double*) calloc(K * len, sizeof(double));
    vector<bool> hidden(no_hidden * len * K);
    
    /* Compute emission log-probabilities */
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < no_hidden; j++) {
            for(int k = 0; k < K; k++) {
                double value = E2[j * K + k] + EX[i * no_hidden + j];
                hidden[k * no_hidden * len + i * no_hidden + j] = (value > 0) ? true : false;
                if(hidden[k * no_hidden * len + i * no_hidden + j]) {
                    emiss[i * K + k] += value;
                }
            }
        }
    }
    for(int i = 0; i < len; i++) {
        for(int k = 0; k < K; k++) {
            emiss[i * K + k] += E2_bias[k];
        }
    }
    
    /* Run Viterbi */
    double L = 0.0f;
    if(!second_order) {
        viterbi_crf(emiss, crf, T, rho, len, pred_T, &L, omega);
    }
    else {
        viterbi_crf_2nd_order(emiss, crf, T, rho, len, pred_T, &L);
    }
    
    /* Fill matrix Z with hidden unit activations */
    for(int i = 0; i < len; i++) {
        for(int j = 0; j < no_hidden; j++) {
            Z->at(i * no_hidden + j) = hidden[pred_T[i] * no_hidden * len + i * no_hidden + j];
        }
    }
    
    /* Clean up memory */
    free(emiss);
}


void viterbi_quadr_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega) {

    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Loop over labels */
    double* emission = (double*) malloc(K * len * sizeof(double));
    quadr_emission(X, len, crf, is_discrete, emission);
    
    /* Run Viterbi decoder */
    viterbi_crf(emission, crf, T, rho, len, pred_T, L, omega);
    free(emission);
}


void viterbi_quadr_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L) {

    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Loop over labels */
    double* emission = (double*) malloc(K * len * sizeof(double));
    quadr_emission(X, len, crf, is_discrete, emission);
    
    /* Run Viterbi decoder */
    viterbi_crf_2nd_order(emission, crf, T, rho, len, pred_T, L);
    free(emission);
}


/* Performs Viterbi decoding given the emission probabilities */
void viterbi_crf(double* emission, model* crf, const double* T, double rho, int len, int* pred_T, double* L, double* omega) {
    
    /* Get variables from model */
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Allocate some memory */
    double* score = (double*) malloc(K * K * sizeof(double));
    int* ind = (int*) malloc(K * len * sizeof(double));
    
    /* Add maximum margin constraints */
    if(rho > 0.0f) {
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < K; j++) {
                if((int) T[i] != j) emission[i * K + j] += rho;
            }
        }
    }
    
    /* Compute message for first hidden variable */
    for(int j = 0; j < K; j++) {
        omega[j] = P[j] + emission[j];
    }
    
    /* Perform forward pass */
    for(int i = 1; i < len; i++) {
        
        /* Compute scores coming in from previous time step */        
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                score[j * K + h] = A[j * K + h] + omega[(i - 1) * K + h];
            }
        }
        
        /* Perform maximization to get new message */
        for(int j = 0; j < K; j++) {
			ind[i * K + j] = 0;
            omega[i * K + j] = score[j * K];
            for(int h = 1; h < K; h++) {
                if(score[j * K + h] > omega[i * K + j]) {
					ind[i * K + j] = h;
                    omega[i * K + j] = score[j * K + h];
                }
            }
        }
        
        /* Add in emission scores */
        for(int j = 0; j < K; j++) {
            omega[i * K + j] += emission[i * K + j];
        }
    }
        
    /* Add in final prior */
    for(int j = 0; j < K; j++) {
        omega[(len - 1) * K + j] += PP[j];
    }
    
    /* Perform backtracking to determine final sequence */
    int max_ind = 0; 
	*L = omega[(len - 1) * K];
    for(int j = 1; j < K; j++) {
        if(omega[(len - 1) * K + j] > *L) {
            *L = omega[(len - 1) * K + j];
            max_ind = j;
        }
    }
    pred_T[len - 1] = max_ind;
    for(int i = len - 2; i >= 0; i--) {
        pred_T[i] = ind[(i + 1) * K + pred_T[i + 1]];
    }
	
    /* Clean up memory */
    free(score);
    free(ind);
}


/* Performs Viterbi decoding in a 2nd-order chain given the emission probabilities */
void viterbi_crf_2nd_order(double* emission, model* crf, const double* T, double rho, int len, int* pred_T, double* L) {
    
    /* Get variables from model */
    if(len == 0) return;
    double* P = crf->P; double* PP = crf->PP; double* R = crf->R; double* RR = crf->RR; double* A = crf->A; double* E = crf->E; double* E_bias = crf->E_bias; double* invS = crf->invS; double* labE = crf->labE; double* labE_bias = crf->labE_bias; char* type = crf->type; int K = crf->K; int D = crf->D; int no_hidden = crf->no_hidden; 
    
    /* Allocate some memory */
    double* omega = (double*) malloc(K * K * sizeof(double));
    double* score = (double*) malloc(K * K * K * sizeof(double));
    int* ind = (int*) malloc(K * K * len * sizeof(double));
    
    /* Add maximum margin constraints */
    if(rho > 0.0f) {
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < K; j++) {
                if((int) T[i] != j) emission[i * K + j] += rho;
            }
        }
    }
    
    /* Compute message for first hidden variable */
    for(int j = 0; j < K; j++) {
        for(int h = 0; h < K; h++) {
            omega[j * K + h] = P[h] + emission[h];
        }
    }
    
    /* Compute message for the second hidden variable */
    if(len > 1) {
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                omega[j * K + h] += PP[j * K + h] + emission[K + j];
            }
        }
    }
    
    /* Perform forward pass */
    for(int i = 2; i < len; i++) {
        
        /* Compute scores coming in from previous time step */        
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                for(int g = 0; g < K; g++) {
                    score[j * K * K + h * K + g] = A[j * K * K + h * K + g] + omega[h * K + g];
                }
            }
        }
        
        /* Perform maximization to get new message */
        for(int j = 0; j < K; j++) {
			for(int h = 0; h < K; h++) {
                ind[i * K * K + j * K + h] = 0;
                omega[j * K + h] = score[j * K * K + h * K];
                for(int g = 1; g < K; g++) {
                    if(score[j * K * K + h * K + g] > omega[j * K + h]) {
                        ind[i * K * K + j * K + h] = g;
                        omega[j * K + h] = score[j * K * K + h * K + g];
                    }
                }
            }
        }
        
        /* Add in emission scores */
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                omega[j * K + h] += emission[i * K + j];
            }
        }
    }
    
    /* Add in final prior */
    for(int j = 0; j < K; j++) {
        for(int h = 0; h < K; h++) {
            omega[j * K + h] += R[j];
        }
    }
    if(len > 1) {
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                omega[j * K + h] += RR[j * K + h];
            }
        }
    }
    
    /* Perform backtracking to determine final sequence */
    if(len > 1) {
        int max_row = 0; 
        int max_col = 0;
        *L = omega[0];
        for(int j = 0; j < K; j++) {
            for(int h = 0; h < K; h++) {
                if(omega[j * K + h] > *L) {
                    *L = omega[j * K + h];
                    max_row = h;
                    max_col = j;                
                }
            }
        }
        pred_T[len - 1] = max_col;
        pred_T[len - 2] = max_row;
        for(int i = len - 3; i >= 0; i--) {
            pred_T[i] = ind[(i + 2) * K * K + pred_T[i + 2] * K + pred_T[i + 1]];
        }
    }
    else {
        pred_T[0] = 0;
        *L = omega[0];
        for(int j = 1; j < K; j++) {
            if(omega[j] > *L) {
                *L = omega[j];
                pred_T[0] = j;
            }
        }            
    }
	
    /* Clean up memory */
    free(omega);
    free(score);
    free(ind);
}


void randperm(int* array, int n) {
	
	// Fill array with indices
	for(int i = 0; i < n; i++) array[i] = i;
	
	// Permute the indices
	for(int last = n; last > 1; last--) { 
		int randomNum = rand() % last; 
		int temporary = array[randomNum]; 
		array[randomNum] = array[last - 1];
		array[last - 1] = temporary; 
	}
}

double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}
