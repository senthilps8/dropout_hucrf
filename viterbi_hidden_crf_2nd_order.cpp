#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"
#include "util.h"

using namespace std;

    
    
/**
 *
 * Performs Viterbi algorithm over y and z in the specified hidden-unit CRF.
 *
 *      [sequence, L, Z] = viterbi_hidden_crf_2nd_order(X, model, T, rho, EX)
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs != 2 && nrhs != 4 && nrhs != 5)
        mexErrMsgTxt("Function should have two, four, or five inputs.");
    if(!mxIsDouble(prhs[0]) && !mxIsCell(prhs[0]))
        mexErrMsgTxt("First input should be a double matrix or cell array.");
    if(!mxIsStruct(prhs[1]))
        mexErrMsgTxt("Second input should be a struct.");
    
    mexPrintf("VITERBI_HIDDEN_CRF_2ND_ORDER: This implementation appears to have a bug!\n");
    
    /* Get data and model */
    const mxArray* X = prhs[0];
    const mxArray* pmodel = prhs[1];
    const mxArray* tmp = mxGetField(pmodel, 0, "type");
    char* type = (char*) malloc((mxGetN(tmp) + 1) * sizeof(char));
    mxGetString(tmp, type, mxGetN(tmp) + 1);
    bool is_discrete  = (strcmp(type, "discrete") == 0 || strcmp(type, "drbm_discrete") == 0) ? true : false;
    double* P      = mxGetPr(mxGetField(pmodel, 0, "pi"));
    double* PP     = mxGetPr(mxGetField(pmodel, 0, "pi2"));
    double* R      = mxGetPr(mxGetField(pmodel, 0, "tau"));
    double* RR     = mxGetPr(mxGetField(pmodel, 0, "tau2"));
    double* A      = mxGetPr(mxGetField(pmodel, 0, "A"));
    double* E      = mxGetPr(mxGetField(pmodel, 0, "E"));
    double* E_bias = mxGetPr(mxGetField(pmodel, 0, "E_bias"));    
    double* labE      = mxGetPr(mxGetField(pmodel, 0, "labE"));
    double* labE_bias = mxGetPr(mxGetField(pmodel, 0, "labE_bias"));
    double rho = nrhs > 3 ? *mxGetPr(prhs[3]) : -1.0f; 
    double* T  = nrhs > 2 ?  mxGetPr(prhs[2]) : NULL;
    
    /* Get characteristics of model and data */
    int K = mxGetM(mxGetField(pmodel, 0, "A"));
    int D = mxGetM(mxGetField(pmodel, 0, "E"));  
    int no_hidden = mxGetN(mxGetField(pmodel, 0, "E"));
    int len = is_discrete ? mxGetM(X) * mxGetN(X) : mxGetN(X);
    
    /* Store stuff in model struct */
    model acrf; model* crf = &acrf; crf->P = P; crf->PP = PP; crf->R = R; crf->RR = RR; crf->A = A; crf->E = E; crf->E_bias = E_bias; crf->labE = labE; crf->labE_bias = labE_bias; crf->type = type; crf->K = K; crf->D = D; crf->no_hidden = no_hidden; crf->second_order = true;
    
    /* Perform Viterbi decoding */
    int* pred_T = (int*) malloc(len * sizeof(int)); double L = 0.0f;
    vector<bool> Z(no_hidden * len);
    double* omega = (double*) malloc(K * len * sizeof(double));
    if(nrhs > 4) {
        const double* EX = nrhs > 4 ? mxGetPr(prhs[4]) : NULL;
        joint_viterbi_hidden_crf(X, T, len, crf, EX, pred_T, &Z, omega, rho, is_discrete);
    }
    else {
        double* EX = (double*) malloc(len * no_hidden * sizeof(double));
        joint_viterbi_hidden_crf(X, T, len, crf, pred_T, EX, &Z, omega, rho, is_discrete);
        free(EX);
    }
    
    /* Copy results to Matlab */
    plhs[0] = mxCreateDoubleMatrix(1, len, mxREAL);
    double* matlab_T = mxGetPr(plhs[0]);
    for(int i = 0; i < len; i++) matlab_T[i] = (double) (pred_T[i] + 1);
    if(nlhs > 1) {                                          // return likelihood
        plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
		double* matlab_L = mxGetPr(plhs[1]);
        matlab_L[0] = L;
    }
    if(nlhs > 2) {                                          // return hidden unit states
        int dims[] = {no_hidden, len};
        plhs[2] = mxCreateLogicalArray(2, dims);
        mxLogical* ZZ = (mxLogical*) mxGetData(plhs[2]);
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < no_hidden; j++) {
                ZZ[i * no_hidden + j] = Z[i * no_hidden + j];
            }
        }
    }    
    
    /* Clean up memory */
    free(type);
    free(pred_T);
    free(omega);
}
