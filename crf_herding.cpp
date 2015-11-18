#include <string.h>
#include <math.h>
#include "mex.h"
#include "util.h"

void quadratic_update(const double* train_Xip, const double* train_Ti, const int* pred_Ti, int len, double* E, double* E_bias, double* invS, int D, int K, double eta_E, double eta_invS, bool positive);


/* Main CRF herding function.
 *
 * Function call:
 *
 *		[pred_test_T, model] = crf_herding(train_X, train_T, test_X, test_T, type, average_models, base_eta, rho, max_iter, burnin_iter);
 *
 * When using mini-batches, the function assumes the training data has been properly shuffled.
 * If you don't want to use a target for q(z), please set target to 0 (or don't specify it).
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Check inputs */
    if(nrhs < 5 && nrhs > 10)
        mexErrMsgTxt("Function should have five to ten inputs.");
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
    if(nrhs > 5 && !mxIsLogical(prhs[5]))
        mexErrMsgTxt("Sixth input should be logical.");        
    if(nrhs > 6 && !mxIsDouble(prhs[6]))
        mexErrMsgTxt("Seventh input should be scalar double.");
    if(nrhs > 7 && !mxIsDouble(prhs[7]))
        mexErrMsgTxt("Eighth input should be scalar double.");
    if(nrhs > 8 && !mxIsDouble(prhs[8]))
        mexErrMsgTxt("Ninth input should be scalar double.");
    if(nrhs > 9 && !mxIsDouble(prhs[9]))
        mexErrMsgTxt("Tenth input should be scalar double.");
    
    /* Process inputs */
    const mxArray* train_X = prhs[0];
    const mxArray* train_T = prhs[1];
    const mxArray* test_X  = prhs[2];
	const mxArray* test_T  = prhs[3];
    int N = mxGetN(train_X) * mxGetM(train_X);
    int M = mxGetN(test_X)  * mxGetM(test_X);
    bool is_discrete  = false;
    bool is_quadratic = false;
    char* type = mxIsStruct(prhs[4]) ? (char*) mxGetPr(mxGetField(prhs[4], 0, "type")) : (char*) malloc((mxGetN(prhs[4]) + 1) * sizeof(char));
    if(mxIsStruct(prhs[4])) mxGetString(mxGetField(prhs[4], 0, "type"), type, mxGetN(mxGetField(prhs[4], 0, "type")) + 1);
    else                    mxGetString(prhs[4], type, mxGetN(prhs[4]) + 1);
    if(strcmp(type, "discrete") == 0)         is_discrete = true;
    if(strcmp(type, "quadr_continuous") == 0) is_quadratic = true;
    bool average_models = nrhs > 5 ? *mxGetLogicals(prhs[5]) : false;
    double base_eta = nrhs > 6 ? *mxGetPr(prhs[6]) : 1.0f;
    double rho      = nrhs > 7 ? *mxGetPr(prhs[7]) : 0.0f;
    int max_iter    = nrhs > 8 ? (int) *mxGetPr(prhs[8]) : 100;
    int burnin_iter = nrhs > 9 ? (int) *mxGetPr(prhs[9]) : 10;
    
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
             
			/* Update number of states and dimensions and maximum norm */
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
	if(is_discrete) mexPrintf("Running on %d training and %d test series with dimensionality %d (discrete) and %d states (eta = %f)...\n", N, M, D, K, base_eta);
    else			mexPrintf("Running on %d training and %d test series with dimensionality %d (continuous) and %d states (eta = %f)...\n", N, M, D, K, base_eta);
	if(average_models) mexPrintf("There are a total number of %d frames, and we do use model averaging...\n",     (int) total_length);
    else               mexPrintf("There are a total number of %d frames, and we do NOT use model averaging...\n", (int) total_length);
        
    /* Correct for stupid Matlab labeling */
    for(int i = 0; i < N; i++) {
        mxArray* tmp_T = mxGetCell(train_T, i);
		double* tmp_Tp = mxGetPr(tmp_T);
        for(int j = 0; j < mxGetN(tmp_T); j++) tmp_Tp[j]--;
    }
    
    /* Construct model and outputs */
    double* P      = (double*) calloc(K * 1, sizeof(double));
	double* PP     = (double*) calloc(K * 1, sizeof(double));
	double* A      = (double*) calloc(K * K, sizeof(double));
	double* E      = (double*) calloc(D * K, sizeof(double));
    double* E_bias = (double*) calloc(1 * K, sizeof(double));
    double* invS = NULL; double* mean_invS = NULL; mxArray* matlab_invS = NULL; 
    int dims[] = {D, D, K};
    if(is_quadratic) { 
        invS = (double*) calloc(D * D * K, sizeof(double));
        matlab_invS = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); mean_invS = mxGetPr(matlab_invS);
        for(int k = 0; k < K; k++) {
            for(int g = 0; g < D; g++) {
                     invS[k * D * D + g * (D + 1)] = 1.0f;
                mean_invS[k * D * D + g * (D + 1)] = 1.0f;
            }
        }
        matlab_invS = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL); mean_invS = mxGetPr(matlab_invS);
    }
    mxArray* matlab_P  = mxCreateDoubleMatrix(K, 1, mxREAL); double* mean_P  = mxGetPr(matlab_P);
    mxArray* matlab_PP = mxCreateDoubleMatrix(K, 1, mxREAL); double* mean_PP = mxGetPr(matlab_PP);
    mxArray* matlab_A  = mxCreateDoubleMatrix(K, K, mxREAL); double* mean_A  = mxGetPr(matlab_A);
	dims[0] = D; dims[1] = K;
    mxArray* matlab_E = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E = mxGetPr(matlab_E);
    dims[0] = 1; dims[1] = K;
	mxArray* matlab_E_bias = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); double* mean_E_bias  = mxGetPr(matlab_E_bias);
    dims[0] = M;
    mxArray* pred_test_T = mxCreateCellArray(1, dims);
    plhs[0] = mxCreateCellArray(1, dims);	
    for(int i = 0; i < M; i++) {
		mxArray* tmp = mxGetCell(test_X, i);
        mxSetCell(pred_test_T, i, mxCreateDoubleMatrix(K, mxGetN(tmp), mxREAL));
        mxSetCell(plhs[0], i, mxCreateDoubleMatrix(1, mxGetN(tmp), mxREAL));
    }
    
    /* Store stuff in model struct */
    model acrf; model* crf = &acrf; crf->P = P; crf->PP = PP; crf->A = A; crf->E = E; crf->E_bias = E_bias; crf->invS = invS; crf->type = type; crf->K = K; crf->D = D; crf->second_order = false;
    model amean_crf; model* mean_crf = &amean_crf; mean_crf->P = mean_P; mean_crf->PP = mean_PP; mean_crf->A = mean_A; mean_crf->E = mean_E; mean_crf->E_bias = mean_E_bias; mean_crf->invS = mean_invS; mean_crf->type = type; mean_crf->K = K; mean_crf->D = D; mean_crf->second_order = false;
    
    /* If we were given a model as initialization, use this model */
    if(mxIsStruct(prhs[4])) {
        memcpy(P,  (double*) mxGetPr(mxGetField(prhs[4], 0, "pi")),  K * 1 * sizeof(double));
        memcpy(PP, (double*) mxGetPr(mxGetField(prhs[4], 0, "tau")), K * 1 * sizeof(double));
        memcpy(A,  (double*) mxGetPr(mxGetField(prhs[4], 0, "A")),   K * K * sizeof(double));
        memcpy(E,  (double*) mxGetPr(mxGetField(prhs[4], 0, "E")),   D * K * sizeof(double));
        memcpy(E_bias, (double*) mxGetPr(mxGetField(prhs[4], 0, "E_bias")), K * sizeof(double));
        if(is_quadratic) {
            memcpy(invS, (double*) mxGetPr(mxGetField(prhs[4], 0, "invS")), D * D * K * sizeof(double));
        }
    }
    else {
        for(int i = 0; i < D * K; i++) E[i] = randn() * .0001;
    }
    
    /* Initialize the step sizes, and herding prediction interval */
    double eta_P      = base_eta / (total_length * K);
    double eta_A      = base_eta / (total_length * K * K);
    double eta_E      = is_discrete ? base_eta / (total_length * K * K) : base_eta / (total_length * D * K);
    double eta_E_bias = is_discrete ? base_eta / (total_length * K * K) : base_eta / (total_length * D * K);
    double eta_invS   = base_eta / (total_length * D * D * K);
    int pred_interval = min(2000, N / 10);
    
	/* Allocate some working memory */
	double* old_P  = (double*) malloc(K * 1 * sizeof(double));
	double* old_PP = (double*) malloc(K * 1 * sizeof(double));
	double* old_A  = (double*) malloc(K * K * sizeof(double));
	double* old_E  = (double*) malloc(D * K * sizeof(double));
    double* old_bE = (double*) malloc(K * sizeof(double));
    double* old_iS = is_quadratic ? (double*) malloc(D * D * K * sizeof(double)) : NULL;
    
    /* Perform herding iterations */
    for(int iter = 0; iter < max_iter; iter++) {
		
		/* Print out progress */
		mexPrintf("Iteration %d of %d...\n", iter + 1, max_iter);
		
		/* Make parameter copies (for computing parameter change) */	
		memcpy(old_P,   P, K * 1 * sizeof(double));
		memcpy(old_PP, PP, K * 1 * sizeof(double));
		memcpy(old_A,   A, K * K * sizeof(double));
		memcpy(old_E,   E, D * K * sizeof(double));
        memcpy(old_bE, E_bias, K * sizeof(double));
        if(is_quadratic) memcpy(old_iS, invS, D * D * K * sizeof(double));
        
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
			int* pred_Ti = (int*) malloc(len * sizeof(int));
            double* omega = (double*) malloc(K * len * sizeof(double));            
			double L = 0.0f;
            
			/* Perform Viterbi decoding */
            if(is_quadratic) {           
                viterbi_quadr_crf(train_Xi, train_Ti, rho, len, crf, is_discrete, pred_Ti, &L, omega);
            }
            else {
                viterbi_linear_crf(train_Xi, train_Ti, rho, len, crf, is_discrete, pred_Ti, &L, omega);
            }
            for(int j = 0; j < len; j++) {
                if(pred_Ti[j] != (int) train_Ti[j]) train_err++;
            }
            
            /* Perform positive parameter update */
			P[(int) train_Ti[0]] += eta_P; 
            PP[(int) train_Ti[len - 1]] += eta_P; 
            for(int j = 1; j < len; j++) {
                A[(int) train_Ti[j] * K + (int) train_Ti[j - 1]] += eta_A;
            }
            if(is_discrete) {                                               // for discrete data
                for(int j = 0; j < len; j++) {
                    if(train_Ti[j] != pred_Ti[j]) {
                        mxArray* train_Xij = mxGetCell(train_Xi, j);
                        double* train_Xijp = mxGetPr(train_Xij);
                        for(int g = 0; g < mxGetM(train_Xij) * mxGetN(train_Xij); g++) {
                            E[(int) train_Ti[j] * D + (int) train_Xijp[g] - 1] += eta_E;
                        }
                        E_bias[(int) train_Ti[j]] += eta_E_bias;
                    }
                }
            }
            else {                                                          // for continuous data
                double* train_Xip = mxGetPr(train_Xi);
                if(is_quadratic) {
                    quadratic_update(train_Xip, train_Ti, pred_Ti, len, E, E_bias, invS, D, K, eta_E, eta_invS, true);
                }
                else {
                    for(int j = 0; j < len; j++) {
                        if(train_Ti[j] != pred_Ti[j]) {
                            for(int g = 0; g < D; g++) {
                                E[(int) train_Ti[j] * D + g] += eta_E * train_Xip[j * D + g];
                            }
                            E_bias[(int) train_Ti[j]] += eta_E_bias;
                        }
                    }
                }
            }
			
            /* Perform negative parameter update */
			P[pred_Ti[0]] -= eta_P; 
            PP[pred_Ti[len - 1]] -= eta_P; 
            for(int j = 1; j < len; j++) {
                A[pred_Ti[j] * K + pred_Ti[j - 1]] -= eta_A;
            }
            if(is_discrete) {                                               // for discrete data
                for(int j = 0; j < len; j++) {
                    if(pred_Ti[j] != train_Ti[j]) {
                        mxArray* train_Xij = mxGetCell(train_Xi, j);
                        double* train_Xijp = mxGetPr(train_Xij);
                        for(int g = 0; g < mxGetM(train_Xij) * mxGetN(train_Xij); g++) {
                            E[pred_Ti[j] * D + (int) train_Xijp[g] - 1] -= eta_E; 
                        }
                        E_bias[pred_Ti[j]] -= eta_E_bias;
                    }
                }
            }
            else {                                                          // for continuous data
                double* train_Xip = mxGetPr(train_Xi);
                if(is_quadratic) {
                    quadratic_update(train_Xip, train_Ti, pred_Ti, len, E, E_bias, invS, D, K, eta_E, eta_invS, false);
                }
                else {
                    for(int j = 0; j < len; j++) {
                        if(pred_Ti[j] != train_Ti[j]) {
                            for(int g = 0; g < D; g++) {
                                E[pred_Ti[j] * D + g] -= eta_E * train_Xip[j * D + g];
                            }
                            E_bias[pred_Ti[j]] -= eta_E_bias;
                        }
                    }
                }
            }
            
            /* Clean up memory */
            free(omega);
            free(pred_Ti);
			
			/* Perform predictions on test data */
			if(!average_models && iter + 1 >= burnin_iter && i % pred_interval == 0) {
				for(int j = 0; j < M; j++) {
					
					/* Perform Viterbi decoding */
					int len = mxGetN(mxGetCell(test_X, j));
					mxArray* test_Xi = mxGetCell(test_X, j);
					double* pred_test_Ti = mxGetPr(mxGetCell(pred_test_T, j));
					int* pred_Ti = (int*) malloc(len * sizeof(int));
                    double* omega = (double*) malloc(K * len * sizeof(double));            
                    double L = 0.0f; double* T = NULL;                 
                    if(is_quadratic)                
                        viterbi_quadr_crf(test_Xi, T, 0.0f, len, crf, is_discrete, pred_Ti, &L, omega);
                    else
                        viterbi_linear_crf(test_Xi, T, 0.0f, len, crf, is_discrete, pred_Ti, &L, omega);
                    
					/* Perform voting */
					for(int h = 0; h < len; h++) {
						pred_test_Ti[h * K + pred_Ti[h]]++;
					}
					free(pred_Ti);
                    free(omega);
				}
			}			
        }
        
        /* Compute change in parameters (excludes bias) */
        double param_change = 0.0f;
        for(int i = 0; i < K * 1; i++) param_change += abs(P[i]  - old_P[i]);
        for(int i = 0; i < K * 1; i++) param_change += abs(PP[i] - old_PP[i]);
        for(int i = 0; i < K * K; i++) param_change += abs(A[i]  - old_A[i]);
        for(int i = 0; i < D * K; i++) param_change += abs(E[i]  - old_E[i]);
        for(int i = 0; i < K; i++) param_change += abs(E_bias[i] - old_bE[i]);
        if(is_quadratic) {
            for(int i = 0; i < D * D * K; i++) param_change += abs(invS[i] - old_iS[i]);    
        }
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
                for(int i = 0; i < D * K; i++) mean_E[i]  = ((ii - 1) / ii) * mean_E[i]  + (1 / ii) * E[i];
                for(int i = 0; i < K; i++) mean_E_bias[i] = ((ii - 1) / ii) * mean_E_bias[i] + (1 / ii) * E_bias[i];     
                if(is_quadratic) {
                    for(int i = 0; i < D * D * K; i++) mean_invS[i] = ((ii - 1) / ii) * mean_invS[i] + (1 / ii) * invS[i];
                }
                
                /* Perform test predictions with mean model */
                for(int i = 0; i < M; i++) {
					
					/* Perform Viterbi decoding */
					int len = mxGetN(mxGetCell(test_X, i));
					mxArray* test_Xi = mxGetCell(test_X, i);
					int* tmp_Ti = (int*) malloc(len * sizeof(int));
                    double* omega = (double*) malloc(K * len * sizeof(double));            
                    double L = 0.0f; double* T = NULL;                    
                    if(is_quadratic)                
                        viterbi_quadr_crf(test_Xi, T, 0.0f, len, mean_crf, is_discrete, tmp_Ti, &L, omega);
                    else
                        viterbi_linear_crf(test_Xi, T, 0.0f, len, mean_crf, is_discrete, tmp_Ti, &L, omega);
                    double* pred_Ti = mxGetPr(mxGetCell(plhs[0], i));
                    for(int j = 0; j < len; j++) pred_Ti[j] = (double) (tmp_Ti[j] + 1);
                    free(tmp_Ti);
                    free(omega);
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
			double error_count = 0;
			double counter = 0;
			for(int i = 0; i < M; i++) {
				int len = mxGetN(mxGetCell(test_T, i));
				double* test_Ti = mxGetPr(mxGetCell(test_T, i));
				double* pred_Ti = mxGetPr(mxGetCell(plhs[0], i));
				for(int j = 0; j < len; j++) {
					counter++;
					if(test_Ti[j] != pred_Ti[j]) error_count++;
				}
			}
			mexPrintf(" - error on test set: %f\n", error_count / counter);
		}
	}
	
	/* Return model */
	if(nlhs > 1) {
        if(!average_models) {
            memcpy(mean_P,  P,  K * 1 * sizeof(double));
            memcpy(mean_PP, PP, K * 1 * sizeof(double));
            memcpy(mean_A,  A,  K * K * sizeof(double));
            memcpy(mean_E,  E,  D * K * sizeof(double));
            memcpy(mean_E_bias, E_bias, K * sizeof(double));
            if(is_quadratic) memcpy(mean_invS, invS, D * D * K * sizeof(double));
        }
		dims[0] = 1;
        if(is_quadratic) {
            const char* field_names[] = {"invS", "pi", "tau", "A", "E", "E_bias", "type"};
            plhs[1] = mxCreateStructArray(1, dims, 7, field_names);
            mxSetField(plhs[1], 0, "invS", matlab_invS);
        }
        else {
            const char* field_names[] = {"pi", "tau", "A", "E", "E_bias", "type"};
            plhs[1] = mxCreateStructArray(1, dims, 6, field_names);		
        }
		mxSetField(plhs[1], 0, "pi",  matlab_P);
		mxSetField(plhs[1], 0, "tau", matlab_PP);
		mxSetField(plhs[1], 0, "A",   matlab_A);
		mxSetField(plhs[1], 0, "E",   matlab_E);
		mxSetField(plhs[1], 0, "E_bias", matlab_E_bias);
		mxSetField(plhs[1], 0, "type", mxCreateString(type));
	}
    else {
        mxDestroyArray(matlab_P);
        mxDestroyArray(matlab_PP);
        mxDestroyArray(matlab_A);
        mxDestroyArray(matlab_E);
        mxDestroyArray(matlab_E_bias);
        if(is_quadratic) mxDestroyArray(matlab_invS);
    }
    
    /* Compute posteriors */
    if(nlhs > 2) {
        if(!average_models) {                   // for herding
            plhs[2] = pred_test_T;
            for(int i = 0; i < M; i++) {
                mxArray* pred_test_Ti = mxGetCell(pred_test_T, i);
                int len = mxGetN(pred_test_Ti);
                double* posterior = (double*) mxGetPr(pred_test_Ti);
                for(int j = 0; j < len; j++) {
                    double count = 0.0f;
                    for(int k = 0; k < K; k++) count += posterior[j * K + k];
                    for(int k = 0; k < K; k++) posterior[j * K + k] /= count;
                }
            }
        }
        else {                                  // for averaged model
            dims[0] = M;
            plhs[2] = mxCreateCellArray(1, dims); 
            for(int i = 0; i < M; i++) {
                
                /* Make prediction */
                int len = mxGetN(mxGetCell(test_X, i));
                mxArray* test_Xi = mxGetCell(test_X, i);
                int* tmp_Ti = (int*) malloc(len * sizeof(int));
                double* omega = (double*) malloc(K * len * sizeof(double));            
                double L = 0.0f; double* T = NULL;                    
                if(is_quadratic)                
                    viterbi_quadr_crf(test_Xi, T, 0.0f, len, mean_crf, is_discrete, tmp_Ti, &L, omega);
                else
                    viterbi_linear_crf(test_Xi, T, 0.0f, len, mean_crf, is_discrete, tmp_Ti, &L, omega);
                
                /* Compute per-frame posteriors from messages */
                mxSetCell(plhs[2], i, mxCreateDoubleMatrix(K, len, mxREAL));
                double* posterior = (double*) mxGetPr(mxGetCell(plhs[2], i));
                double max_val; double sum_val;
                for(int j = 0; j < len; j++) {
                    max_val = omega[j * K]; sum_val = FLT_MIN;
                    for(int k = 1; k < K; k++) {
                        if(omega[j * K + k] > max_val) max_val = omega[j * K + k];
                    }
                    sum_val = 0.0f;
                    for(int k = 0; k < K; k++) {
                        posterior[j * K + k] = exp(omega[j * K + k] - max_val);
                        sum_val += posterior[j * K + k];
                    }
                    for(int k = 0; k < K; k++) posterior[j * K + k] /= sum_val;
                }
            }
        }
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
    free(E);
    free(E_bias);
    if(is_quadratic) free(invS);
    if(!mxIsStruct(prhs[4])) free(type);	
	free(old_P);
    free(old_PP);
	free(old_A);
	free(old_E);
    free(old_bE);
    if(is_quadratic) free(old_iS);
    /*for(int i = 0; i < M; i++) {
		mxFree(mxGetPr(mxGetCell(pred_test_T, i)));
	}
	mxFree(mxGetPr(pred_test_T));*/
}


/* Compute quadratic potential update */
void quadratic_update(const double* train_Xip, const double* train_Ti, const int* pred_Ti, int len, double* E, double* E_bias, double* invS, int D, int K, double eta_E, double eta_invS, bool positive) {
    
    /* Make sure we do the right update */
    if(!positive) {
        eta_E = -eta_E;
        eta_invS = -eta_invS;
    }
    
    /* Loop over labels */
    for(int k = 0; k < K; k++) {
        
        /* Compute X - mean */
        int count = 0;
        for(int j = 0; j < len; j++) {
            if( positive && (int) train_Ti[j] == k && (int)  pred_Ti[j] != k) count++;
            if(!positive && (int)  pred_Ti[j] == k && (int) train_Ti[j] != k) count++;
        }
        double* zero_mean_X = (double*) malloc(D * count * sizeof(double)); int ii = 0;
        for(int j = 0; j < len; j++) {
            if(( positive && (int) train_Ti[j] == k && (int)  pred_Ti[j] != k) ||
                !positive && (int)  pred_Ti[j] == k && (int) train_Ti[j] != k) {
                for(int g = 0; g < D; g++) {
                    zero_mean_X[ii * D + g] = train_Xip[j * D + g] - E[k * D + g];
                }
                ii++;
            }
        }
        
        /* Update mean for current label */
        for(int j = 0; j < count; j++) {
            for(int g = 0; g < D; g++) {
                for(int h = 0; h < D; h++) {
                    E[k * D + g] -= eta_E * 2 * zero_mean_X[j * D + h] * invS[k * D * D + g * D + h];
                }
            }
        }
        
        /* Update inverse covariance for current label */
        for(int j = 0; j < count; j++) {
            for(int g = 0; g < D; g++) {
                invS[k * D * D + g * D + g] += eta_invS * zero_mean_X[j * D + g] * zero_mean_X[j * D + g];
                for(int h = g + 1; h < D; h++) {
                    invS[k * D * D + g * D + h] += eta_invS * zero_mean_X[j * D + h] * zero_mean_X[j * D + g];
                    invS[k * D * D + h * D + g] = invS[k * D * D + g * D + h];
                }
            }
        }
        free(zero_mean_X);
    }
}
