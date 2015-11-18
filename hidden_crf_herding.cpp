#include <string.h>
#include <math.h>
#include <vector>
#include "mex.h"
#include "util.h"

using namespace std;



/* Main hidden CRF herding function.
 *
 * Function call:
 *
 *		[pred_test_T, model] = hidden_crf_herding(train_X, train_T, test_X, test_T, type, no_hidden, average_models, base_eta, rho, max_iter, burnin_iter);
 *		[pred_test_T, model] = hidden_crf_herding(train_X, train_T, test_X, test_T, model, no_hidden, average_models, base_eta, rho, max_iter, burnin_iter);
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs < 6 && nrhs > 11)
        mexErrMsgTxt("Function should have six to eleven inputs.");
    if(!mxIsCell(prhs[0]))
        mexErrMsgTxt("First input should be a cell-array.");
    if(!mxIsCell(prhs[1]))
        mexErrMsgTxt("Second input should be a cell-array.");
    if(mxGetNumberOfFields(prhs[0]) != mxGetNumberOfFields(prhs[1])) 
        mexErrMsgTxt("First two cell array should have the same number of entries.");    
    if(!mxIsCell(prhs[2]))
        mexErrMsgTxt("Third input should be a cell-array.");
    if(!mxIsCell(prhs[3]))
        mexErrMsgTxt("Fourth input should be a cell-array.");
	if(!mxIsChar(prhs[4]) && !mxIsStruct(prhs[4]))
        mexErrMsgTxt("Fifth input should be a string or struct.");
    if(!mxIsDouble(prhs[5]))
        mexErrMsgTxt("Sixth input should be a scalar double.");
    if(nrhs > 6 && !mxIsLogical(prhs[6]))
        mexErrMsgTxt("Seventh input should be logical.");
    if(nrhs > 7 && !mxIsDouble(prhs[7]))
        mexErrMsgTxt("Eighth input should be scalar double.");
    if(nrhs > 8 && !mxIsDouble(prhs[8]))
        mexErrMsgTxt("Ninth input should be scalar double.");
    if(nrhs > 9 && !mxIsDouble(prhs[9]))
        mexErrMsgTxt("Tenth input should be scalar double.");
    if(nrhs > 10 && !mxIsDouble(prhs[10]))
        mexErrMsgTxt("Eleventh input should be scalar double.");
    
    /* Process inputs */
    const mxArray* train_X = prhs[0];
    const mxArray* train_T = prhs[1];
    const mxArray* test_X  = prhs[2];
	const mxArray* test_T  = prhs[3];
    int N = mxGetN(train_X) * mxGetM(train_X);
    int M = mxGetN(test_X)  * mxGetM(test_X);
    char* type = mxIsStruct(prhs[4]) ? (char*) mxGetPr(mxGetField(prhs[4], 0, "type")) : (char*) malloc((mxGetN(prhs[4]) + 1) * sizeof(char));
    if(mxIsStruct(prhs[4])) mxGetString(mxGetField(prhs[4], 0, "type"), type, mxGetN(mxGetField(prhs[4], 0, "type")) + 1);
    else                    mxGetString(prhs[4], type, mxGetN(prhs[4]) + 1);
    bool is_discrete = strcmp(type, "drbm_discrete") == 0 ? true : false;
    int no_hidden = (int) *mxGetPr(prhs[5]);
    bool average_models = nrhs > 6 ? *mxGetLogicals(prhs[6]) : false;
    double base_eta = nrhs > 7  ?       *mxGetPr(prhs[7])  : 1.0f;
    double rho      = nrhs > 8  ?       *mxGetPr(prhs[8])  : 0.0f;
    int max_iter    = nrhs > 9  ? (int) *mxGetPr(prhs[9])  : 100;
    int burnin_iter = nrhs > 10 ? (int) *mxGetPr(prhs[10]) : 10;
    
    /* Compute total length of training sequences, number of discrete features, and number of labels */
	int D = 0;
	int K = 0;
    double total_length = 0.0f;
    for(int i = 0; i < N; i++) {
        mxArray* tmp_X = mxGetCell(train_X, i);
		mxArray* tmp_T = mxGetCell(train_T, i);
		double* tmp_Tp = mxGetPr(tmp_T);
        total_length += (double) (mxGetM(tmp_T) * mxGetN(tmp_T));
		for(int j = 0; j < mxGetN(tmp_T); j++) {
             
			/* Update number of states and dimensions */
			K = max(K, (int) tmp_Tp[j]);
            if(is_discrete) {
                mxArray* tmp = mxGetCell(tmp_X, j);
                double* tmpp = mxGetPr(tmp);
                for(int h = 0; h < mxGetM(tmp) * mxGetN(tmp); h++) {
                    D = max(D, (int) tmpp[h]);
                }
            }
            else {
                D = mxGetM(tmp_X);
            }
		}
    }
	if(is_discrete) mexPrintf("Running on %d training and %d test series with dimensionality %d (discrete), %d hidden units, and %d states (eta = %f)...\n", N, M, D, no_hidden, K, base_eta);
    else			mexPrintf("Running on %d training and %d test series with dimensionality %d (continuous), %d hidden units, and %d states (eta = %f)...\n", N, M, D, no_hidden, K, base_eta);
	if(average_models) mexPrintf("There are a total number of %d frames, and we do use model averaging...\n",     (int) total_length);
    else               mexPrintf("There are a total number of %d frames, and we do NOT use model averaging...\n", (int) total_length);
    
    /* Correct for stupid Matlab labeling */
    for(int i = 0; i < N; i++) {
        mxArray* tmp_T = mxGetCell(train_T, i);
		double* tmp_Tp = mxGetPr(tmp_T);
        for(int j = 0; j < mxGetN(tmp_T); j++) tmp_Tp[j]--;
    }
    
    /* Construct model and outputs */    
    double* P  = (double*) calloc(K * 1, sizeof(double));
    double* PP = (double*) calloc(K * 1, sizeof(double));
	double* A  = (double*) calloc(K * K, sizeof(double));
	double* E1 = (double*) calloc(D * no_hidden, sizeof(double));
    double* E2 = (double*) calloc(K * no_hidden, sizeof(double));
    double* E1_bias = (double*) calloc(no_hidden, sizeof(double));
    double* E2_bias = (double*) calloc(K,         sizeof(double));
    mxArray* matlab_P  = mxCreateDoubleMatrix(K, 1, mxREAL); double* mean_P  = mxGetPr(matlab_P);
    mxArray* matlab_PP = mxCreateDoubleMatrix(K, 1, mxREAL); double* mean_PP = mxGetPr(matlab_PP);
    mxArray* matlab_A  = mxCreateDoubleMatrix(K, K, mxREAL); double* mean_A  = mxGetPr(matlab_A);
	int dims[] = {D, no_hidden};
    mxArray* matlab_E1  = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E1 = mxGetPr(matlab_E1);
    dims[0] = K, dims[1] = no_hidden;
    mxArray* matlab_E2  = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E2 = mxGetPr(matlab_E2);
	dims[0] = 1; dims[1] = no_hidden;
	mxArray* matlab_E1_bias = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E1_bias = mxGetPr(matlab_E1_bias);
	dims[0] = 1; dims[1] = K;
	mxArray* matlab_E2_bias = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E2_bias = mxGetPr(matlab_E2_bias);
	dims[0] = M;
    mxArray* pred_test_T = mxCreateCellArray(1, dims);
    plhs[0] = mxCreateCellArray(1, dims);	
    for(int i = 0; i < M; i++) {
		mxArray* tmp = mxGetCell(test_X, i);
        mxSetCell(pred_test_T, i, mxCreateDoubleMatrix(K, mxGetN(tmp), mxREAL));
        mxSetCell(plhs[0], i, mxCreateDoubleMatrix(1, mxGetN(tmp), mxREAL));
    }
    
    /* If we were given a model as initialization, use this model */
    if(mxIsStruct(prhs[4])) {
        mexPrintf("Initializing with given model..\n");
        memcpy(P,  (double*) mxGetPr(mxGetField(prhs[4], 0, "pi")),   K * 1 * sizeof(double));
        memcpy(PP, (double*) mxGetPr(mxGetField(prhs[4], 0, "tau")),  K * 1 * sizeof(double));
        memcpy(A,  (double*) mxGetPr(mxGetField(prhs[4], 0, "A")),    K * K * sizeof(double));
        memcpy(E1, (double*) mxGetPr(mxGetField(prhs[4], 0, "E")),    D * no_hidden * sizeof(double));
        memcpy(E2, (double*) mxGetPr(mxGetField(prhs[4], 0, "labE")), K * no_hidden * sizeof(double));
        memcpy(E1_bias, (double*) mxGetPr(mxGetField(prhs[4], 0, "E_bias")),    1 * no_hidden * sizeof(double));
        memcpy(E2_bias, (double*) mxGetPr(mxGetField(prhs[4], 0, "labE_bias")), 1 * K * sizeof(double));
    }
    else {
        for(int i = 0; i < D * no_hidden; i++) E1[i] = randn() * .0001;
        for(int i = 0; i < K * no_hidden; i++) E2[i] = randn() * .0001;
    }
    
    /* Store stuff in model struct */
    model acrf; model* crf = &acrf; crf->P = P; crf->PP = PP; crf->A = A; crf->E = E1; crf->E_bias = E1_bias; crf->labE = E2; crf->labE_bias = E2_bias; crf->type = type; crf->K = K; crf->D = D; crf->no_hidden = no_hidden; crf->second_order = false;
    model amean_crf; model* mean_crf = &amean_crf; mean_crf->P = mean_P; mean_crf->PP = mean_PP; mean_crf->A = mean_A; mean_crf->E = mean_E1; mean_crf->E_bias = mean_E1_bias; mean_crf->labE = mean_E2; mean_crf->labE_bias = mean_E2_bias; mean_crf->type = type; mean_crf->K = K; mean_crf->D = D; mean_crf->no_hidden = no_hidden; mean_crf->second_order = false;
    
    /* Initialize the step sizes */
    double eta_P   = base_eta / (total_length * K);
    double eta_A   = base_eta / (total_length * K * K);
    double eta_E1  = base_eta / (total_length * D * no_hidden);
    double eta_E2  = base_eta / (total_length * K * no_hidden);
    double eta_bE1 = base_eta / (total_length * D * no_hidden);
    double eta_bE2 = base_eta / (total_length * K * no_hidden);
    if(is_discrete) { eta_E1 *= (D / no_hidden); eta_bE1 *= (D / no_hidden); }
    int pred_interval = min(2000, N / 10);
    eta_E1  *= 5.0f;
	eta_bE1 *= 5.0f;
	
	/* Allocate some working memory */
	double* old_P   = (double*) malloc(K * 1 * sizeof(double));
	double* old_PP  = (double*) malloc(K * 1 * sizeof(double));	
    double* old_A   = (double*) malloc(K * K * sizeof(double));
    double* old_E1  = (double*) malloc(D * no_hidden * sizeof(double));
    double* old_E2  = (double*) malloc(K * no_hidden * sizeof(double));
    double* old_bE1 = (double*) malloc(no_hidden * sizeof(double));
    double* old_bE2 = (double*) malloc(K * sizeof(double));
    
    /* Perform herding iterations */
    for(int iter = 0; iter < max_iter; iter++) {
		
		/* Print out progress */
		mexPrintf("Iteration %d of %d...\n", iter + 1, max_iter);
		
		/* Make parameter copies (for computing parameter change) */	
		memcpy(old_P,   P,  K * 1 * sizeof(double));
		memcpy(old_PP,  PP, K * 1 * sizeof(double));		
        memcpy(old_A,   A,  K * K * sizeof(double));
		memcpy(old_E1,  E1, D * no_hidden * sizeof(double));
        memcpy(old_E2,  E2, K * no_hidden * sizeof(double));
        memcpy(old_bE1, E1_bias, no_hidden * sizeof(double));
        memcpy(old_bE2, E2_bias, K * sizeof(double));
        
		/* Randomly reorder data */
		int perm_ind[N];
        randperm(perm_ind, N);
        int train_err = 0;
        
        /* Loop over training set */
        for(int i = 0; i < N; i++) {            
            
			/* Gather all information about current time series */
			int len = mxGetN(mxGetCell(train_T, perm_ind[i]));
			mxArray* train_Xi = mxGetCell(train_X, perm_ind[i]);
            double* train_Ti = mxGetPr(mxGetCell(train_T, perm_ind[i]));
			
            /* Allocate some memory for hidden unit activations, etc. */
            int* pred_Ti  =    (int*) malloc(len * sizeof(int));
            double* EX    = (double*) malloc(no_hidden * len * sizeof(double));
			double* omega = (double*) malloc(K * len * sizeof(double));
            vector<bool> Z_pos(no_hidden * len);
            vector<bool> Z_neg(no_hidden * len);
			
			/* Perform Viterbi decoding (negative phase; also computes EX and negative hidden states) */
            joint_viterbi_hidden_crf(train_Xi, train_Ti, len, crf, pred_Ti, EX, &Z_neg, omega, rho, is_discrete);
            for(int j = 0; j < len; j++) {
                if(pred_Ti[j] != (int) train_Ti[j]) train_err++;
            }
            
            /* Compute states of hidden units (positive phase) */
            for(int j = 0; j < len; j++) {
                for(int h = 0; h < no_hidden; h++) {
                    Z_pos[j * no_hidden + h] = (EX[j * no_hidden + h] + E2[h * K + (int) train_Ti[j]] > 0.0f) ? true : false;
                }
            }
            
            /* Perform transition parameter updates (positive phase) */
			P[(int) train_Ti[0]] += eta_P; 
            PP[(int) train_Ti[len - 1]] += eta_P;            
            for(int j = 1; j < len; j++) {
                A[(int) train_Ti[j] * K + (int) train_Ti[j - 1]] += eta_A;
            }
            
            /* Perform data-hidden parameter updates (positive phase) */
            if(is_discrete) {                                               // for discrete data
                for(int j = 0; j < len; j++) {
                    mxArray* train_Xij = mxGetCell(train_Xi, j);
                    double* train_Xijp = mxGetPr(train_Xij);
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_pos[j * no_hidden + h] && !Z_neg[j * no_hidden + h]) {
                            for(int g = 0; g < mxGetM(train_Xij) * mxGetN(train_Xij); g++) {                            
                                E1[h * D + (int) train_Xijp[g] - 1] += eta_E1;
                            }
                            E1_bias[h] += eta_bE1;
                        }
                    }
                }
            }
            else {                                                          // for continuous data
                double* train_Xip = mxGetPr(train_Xi);
                for(int j = 0; j < len; j++) {
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_pos[j * no_hidden + h] && !Z_neg[j * no_hidden + h]) {
                            for(int g = 0; g < D; g++) {
                                E1[h * D + g] += eta_E1 * train_Xip[j * D + g];                                
                            }
                            E1_bias[h] += eta_bE1; 
                        }
                    }
                }                
            }
            
            /* Perform hidden-label parameter updates (positive phase) */
            for(int j = 0; j < len; j++) {
                if(train_Ti[j] != pred_Ti[j]) {
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_pos[j * no_hidden + h]) {
                            E2[h * K + (int) train_Ti[j]] += eta_E2;
                        }
                    }
                    E2_bias[(int) train_Ti[j]] += eta_bE2;
                }
            }
			
            /* Perform transition parameter updates (negative phase) */
			P[(int) pred_Ti[0]] -= eta_P;
            PP[(int) pred_Ti[len - 1]] -= eta_P;
            for(int j = 1; j < len; j++) {
                A[(int) pred_Ti[j] * K + (int) pred_Ti[j - 1]] -= eta_A;
            }
            
            /* Perform data-hidden parameter updates (negative phase) */
            if(is_discrete) {                                               // for discrete data
                for(int j = 0; j < len; j++) {
                    mxArray* train_Xij = mxGetCell(train_Xi, j);
                    double* train_Xijp = mxGetPr(train_Xij);
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_neg[j * no_hidden + h] && !Z_pos[j * no_hidden + h]) {
                            for(int g = 0; g < mxGetM(train_Xij) * mxGetN(train_Xij); g++) {                            
                                E1[h * D + (int) train_Xijp[g] - 1] -= eta_E1;                                
                            }
                            E1_bias[h] -= eta_bE1;
                        }
                    }
                }
            }
            else {                                                          // for continuous data
                double* train_Xip = mxGetPr(train_Xi);
                for(int j = 0; j < len; j++) {
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_neg[j * no_hidden + h] && !Z_pos[j * no_hidden + h]) {
                            for(int g = 0; g < D; g++) {
                                E1[h * D + g] -= eta_E1 * train_Xip[j * D + g];                                
                            }
                            E1_bias[h] -= eta_bE1;
                        }
                    }
                }                
            }
            
            /* Perform hidden-label parameter updates (negative phase) */
            for(int j = 0; j < len; j++) {
                if(pred_Ti[j] != train_Ti[j]) {
                    for(int h = 0; h < no_hidden; h++) {
                        if(Z_neg[j * no_hidden + h]) {
                            E2[h * K + (int) pred_Ti[j]] -= eta_E2;
                        }
                    }
                    E2_bias[pred_Ti[j]] -= eta_bE2;
                }
            }
            
            /* Clean up memory */
            free(pred_Ti);
            free(omega);
            free(EX);
            
			/* Perform predictions on test data */
			if(!average_models && iter + 1 >= burnin_iter && i % pred_interval == 0) {
				for(int j = 0; j < M; j++) {
					
					/* Perform Viterbi decoding */
					int len = mxGetN(mxGetCell(test_X, j));
					mxArray* test_Xi = mxGetCell(test_X, j);
					double* pred_test_Ti = mxGetPr(mxGetCell(pred_test_T, j));
					int* pred_Ti = (int*) malloc(len * sizeof(int));                    
                    joint_viterbi_hidden_crf(test_Xi, len, crf, pred_Ti, is_discrete);

					/* Perform voting */
					for(int h = 0; h < len; h++) {
						pred_test_Ti[h * K + pred_Ti[h]]++;
					}
					free(pred_Ti);
				}
			}			
        }
        
        /* Compute change in parameters (ignores bias) */
        double param_change = 0.0f;
        for(int i = 0; i < K * 1; i++) param_change += abs(P[i]  - old_P[i]);
        for(int i = 0; i < K * 1; i++) param_change += abs(PP[i] - old_PP[i]);
        for(int i = 0; i < K * K; i++) param_change += abs(A[i]  - old_A[i]);
        for(int i = 0; i < D * no_hidden; i++) param_change += abs(E1[i] - old_E1[i]);
        for(int i = 0; i < K * no_hidden; i++) param_change += abs(E2[i] - old_E2[i]);        
        for(int i = 0; i < no_hidden; i++) param_change += abs(E1_bias[i] - old_bE1[i]);        
        for(int i = 0; i < K; i++) param_change += abs(E2_bias[i] - old_bE2[i]);        
        mexPrintf("Cumulative parameter change: %f\n", param_change);
        mexPrintf("Training error this iteration: %f\n", (double) train_err / (double) total_length);
		
        /* Only when we already made predictions */
		if(iter + 1 >= burnin_iter) {
			if(average_models) {
                
                /* Perform averaging of the models */
                double ii = (double) (iter + 2 - burnin_iter);
                for(int i = 0; i < K * 1; i++) mean_P[i]  = ((ii - 1) / ii) * mean_P[i]  + (1 / ii) * P[i];
                for(int i = 0; i < K * 1; i++) mean_PP[i] = ((ii - 1) / ii) * mean_PP[i] + (1 / ii) * PP[i];
                for(int i = 0; i < K * K; i++) mean_A[i]  = ((ii - 1) / ii) * mean_A[i]  + (1 / ii) * A[i];
                for(int i = 0; i < D * no_hidden; i++) mean_E1[i] = ((ii - 1) / ii) * mean_E1[i] + (1 / ii) * E1[i];
                for(int i = 0; i < K * no_hidden; i++) mean_E2[i] = ((ii - 1) / ii) * mean_E2[i] + (1 / ii) * E2[i];                
                for(int i = 0; i < 1 * no_hidden; i++) mean_E1_bias[i] = ((ii - 1) / ii) * mean_E1_bias[i] + (1 / ii) * E1_bias[i];
                for(int i = 0; i < 1 * K; i++)         mean_E2_bias[i] = ((ii - 1) / ii) * mean_E2_bias[i] + (1 / ii) * E2_bias[i];
                
                /* Perform test predictions with mean model */
                for(int i = 0; i < M; i++) {
					
					/* Perform Viterbi decoding */
					int len = mxGetN(mxGetCell(test_X, i));
					mxArray* test_Xi = mxGetCell(test_X, i);
					int* tmp_Ti = (int*) malloc(len * sizeof(int));
                    joint_viterbi_hidden_crf(test_Xi, len, mean_crf, tmp_Ti, is_discrete);
                    double* pred_Ti = mxGetPr(mxGetCell(plhs[0], i));
                    for(int j = 0; j < len; j++) pred_Ti[j] = (double) (tmp_Ti[j] + 1);
                    free(tmp_Ti);
				}
            }
            else {
                
                /* Determine states with the most votes for test data */
                for(int i = 0; i < M; i++) {
                    int len = mxGetN(mxGetCell(pred_test_T,  i));
                    double* pred_test_Ti = mxGetPr(mxGetCell(pred_test_T,  i));
                    double* pred_Ti = mxGetPr(mxGetCell(plhs[0], i));
                    for(int j = 0; j < len; j++) {
                        double max_val = pred_test_Ti[j * K];
                        pred_Ti[j] = 1.0f;
                        for(int h = 1; h < K; h++) {
                            if(pred_test_Ti[j * K + h] > max_val) {
                                max_val = pred_test_Ti[j * K + h];
                                pred_Ti[j] = (double) (h + 1);
                            }
                        }
                    }
                }
            }
			
			/* Compute error on test set */
			int error_count = 0;
			int counter = 0;
			for(int i = 0; i < M; i++) {
				int len = mxGetN(mxGetCell(test_T, i));
				double* test_Ti = mxGetPr(mxGetCell(test_T, i));
				double* pred_Ti = mxGetPr(mxGetCell(plhs[0], i));
				for(int j = 0; j < len; j++) {
					counter++;
					if(test_Ti[j] != pred_Ti[j]) error_count++;
				}
			}
			mexPrintf(" - error on test set: %f\n", (double) error_count / (double) counter);
		}
	}
	
	/* Return model */
	if(nlhs > 1) {
        if(!average_models) {
            memcpy(mean_P,  P,  K * 1 * sizeof(double));
            memcpy(mean_PP, PP, K * 1 * sizeof(double));
            memcpy(mean_A,  A,  K * K * sizeof(double));
            memcpy(mean_E1, E1, D * no_hidden * sizeof(double));
            memcpy(mean_E2, E2, K * no_hidden * sizeof(double));            
            memcpy(mean_E1_bias, E1_bias, no_hidden * sizeof(double));
            memcpy(mean_E2_bias, E2_bias, K         * sizeof(double));
        }
		dims[0] = 1;
		const char* field_names[] = {"pi", "tau", "A", "E", "labE", "E_bias", "labE_bias", "type"};
		plhs[1] = mxCreateStructArray(1, dims, 8, field_names);		
		mxSetField(plhs[1], 0, "pi",        matlab_P);
		mxSetField(plhs[1], 0, "tau",       matlab_PP);
		mxSetField(plhs[1], 0, "A",         matlab_A);
		mxSetField(plhs[1], 0, "E",         matlab_E1);
		mxSetField(plhs[1], 0, "labE",      matlab_E2);
        mxSetField(plhs[1], 0, "E_bias",    matlab_E1_bias);
        mxSetField(plhs[1], 0, "labE_bias", matlab_E2_bias);        
		mxSetField(plhs[1], 0, "type",      mxCreateString(type));
	}
	else {
        mxDestroyArray(matlab_P);
        mxDestroyArray(matlab_PP);
        mxDestroyArray(matlab_A);
        mxDestroyArray(matlab_E1);
        mxDestroyArray(matlab_E2);
        mxDestroyArray(matlab_E1_bias);
        mxDestroyArray(matlab_E2_bias);
    }
    
    /* Correct for stupid Matlab labeling */
    for(int i = 0; i < N; i++) {
        mxArray* tmp_T = mxGetCell(train_T, i);
		double* tmp_Tp = mxGetPr(tmp_T);
        for(int j = 0; j < mxGetN(tmp_T); j++) tmp_Tp[j]++;
    }
    
    /* Clean up memory */
    free(P);
    free(PP);
    free(A);
    free(E1);
    free(E2);
    free(E1_bias);
    free(E2_bias);
    if(!mxIsStruct(prhs[4])) free(type);
	free(old_P);
    free(old_PP);
	free(old_A);
    free(old_E1);
	free(old_E2); 
    free(old_bE1);
    free(old_bE2);
	/*for(int i = 0; i < M; i++) {
		mxFree(mxGetPr(mxGetCell(pred_test_T, i)));
	}
	mxFree(mxGetPr(pred_test_T));*/
}
