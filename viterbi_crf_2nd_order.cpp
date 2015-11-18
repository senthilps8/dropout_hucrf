#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"
#include "util.h"

using namespace std;


void viterbi_linear_crf(const mxArray* X, const double* T, double rho, int len, int D, const double* P, const double* R, const double* A, const double* E, int K, bool is_discrete, int* pred_T, double* L, double* omega);
void viterbi_hidden_crf(const mxArray* X, const double* T, double rho, int len, int D, const double* P, const double* R, const double* A, const double* E, const double* E_bias, int no_hidden, const double* labE, const double* labE_bias, int K, bool is_discrete, int* pred_T, double* L, double* omega);
void viterbi_quadr_crf(const mxArray* X, const double* T, double rho, int len, int D, const double* P, const double* R, const double* A, const double* E, const double* invS, int K, bool is_discrete, int* pred_T, double* L, double* omega);
void viterbi_crf(double* emission, const double* P, const double* R, const double* A, const double* T, double rho, int len, int K, int* pred_T, double* L, double* omega);

    
/**
 *
 * Performs Viterbi algorithm in the specified CRF (with second order chain).
 *
 *      [sequence, L] = viterbi_crf_2nd_order(X, model)
 *      [sequence, L] = viterbi_crf_2nd_order(X, model, T, rho)
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs != 2 && nrhs != 4)
        mexErrMsgTxt("Function should have two or four inputs.");
    if(!mxIsDouble(prhs[0]) && !mxIsCell(prhs[0]))
        mexErrMsgTxt("First input should be a double matrix or cell array.");
    if(!mxIsStruct(prhs[1]))
        mexErrMsgTxt("Second input should be a struct.");
    
    mexPrintf("VITERBI_CRF_2ND_ORDER: This implementation may have a bug?\n");
    
    /* Get data and model */
    const mxArray* X = prhs[0];
    const mxArray* pmodel = prhs[1];
    const mxArray* tmp = mxGetField(pmodel, 0, "type");
    char* type = (char*) malloc((mxGetN(tmp) + 1) * sizeof(char));
    mxGetString(tmp, type, mxGetN(tmp) + 1);
    bool is_discrete  = (strcmp(type, "discrete") == 0 || strcmp(type, "drbm_discrete") == 0 || strcmp(type, "quadr_discrete") == 0) ? true : false;
    bool is_hidden    = (strcmp(type,  "drbm_continuous") == 0 || strcmp(type,  "drbm_discrete") == 0) ? true : false;
    bool is_quadratic = (strcmp(type, "quadr_continuous") == 0 || strcmp(type, "quadr_discrete") == 0) ? true : false;       // quadr_discrete is not actually supported!!!
    double* P      = mxGetPr(mxGetField(pmodel, 0, "pi"));
    double* PP     = mxGetPr(mxGetField(pmodel, 0, "pi2"));
    double* R      = mxGetPr(mxGetField(pmodel, 0, "tau"));
    double* RR     = mxGetPr(mxGetField(pmodel, 0, "tau2"));
    double* A      = mxGetPr(mxGetField(pmodel, 0, "A"));
    double* E      = mxGetPr(mxGetField(pmodel, 0, "E"));
    double* E_bias = mxGetPr(mxGetField(pmodel, 0, "E_bias"));
    double* labE      = is_hidden ? mxGetPr(mxGetField(pmodel, 0, "labE")) : NULL;
    double* labE_bias = is_hidden ? mxGetPr(mxGetField(pmodel, 0, "labE_bias")) : NULL;
    double* invS   = is_quadratic ? mxGetPr(mxGetField(pmodel, 0, "invS")) : NULL;
    double* T  = nrhs > 2 ?  mxGetPr(prhs[2]) : NULL;
    double rho = nrhs > 3 ? *mxGetPr(prhs[3]) : -1.0f; 
    
    /* Get characteristics of model and data */
    int K = mxGetM(mxGetField(pmodel, 0, "A"));
    int D = mxGetM(mxGetField(pmodel, 0, "E"));  
    int no_hidden = is_hidden ? mxGetN(mxGetField(pmodel, 0, "E")) : 0;
    int len = is_discrete ? mxGetM(X) * mxGetN(X) : mxGetN(X);
    
    /* Store stuff in model struct */
    model acrf; model* crf = &acrf; crf->P = P; crf->PP = PP; crf->R = R; crf->RR = RR; crf->A = A; crf->E = E; crf->E_bias = E_bias; crf->labE = labE; crf->labE_bias = labE_bias; crf->invS = invS; crf->type = type; crf->K = K; crf->D = D; crf->no_hidden = no_hidden; crf->second_order = true;
    
    /* Perform Viterbi decoding */
    int* pred_T = (int*) malloc(len * sizeof(int)); double L;
    if(is_hidden)         viterbi_hidden_crf_2nd_order(X, T, rho, len, crf, is_discrete, pred_T, &L);
    else if(is_quadratic) viterbi_quadr_crf_2nd_order(X, T, rho, len, crf, is_discrete, pred_T, &L);
    else                  viterbi_linear_crf_2nd_order(X, T, rho, len, crf, is_discrete, pred_T, &L);
    
    /* Copy results to Matlab */
    plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
    double* matlab_T = mxGetPr(plhs[0]);
    for(int i = 0; i < len; i++) matlab_T[i] = (double) (pred_T[i] + 1);
    if(nlhs > 1) {                                          // return likelihood
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
		double* matlab_L = mxGetPr(plhs[1]);
        matlab_L[0] = L;
    }
    
    /* Clean up memory */
    free(type);
    free(pred_T);
}
