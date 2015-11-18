
#include <vector>

using namespace std;


#ifndef UTIL2_H
#define UTIL2_H

struct model {
    double* P;              // init prior
    double* PP;             // final prior
    double* R;              // init prior (2nd order)
    double* RR;             // final prior (2nd order)
    double* A;              // transitions
    double* E;              // emission
    double* E_bias;         // emission bias
    double* invS;           // inverse covariances (quadratic only)
    double* labE;           // hidden-label weight (hiddens only)
    double* labE_bias;      // label bias (hiddens only)
    char* type;             // type
    int K;                  // number of labels 
    int D;                  // data dimensionality
    int no_hidden;          // number of hidden units (hiddens only)
    bool second_order;      // true for second-order model
};

void linear_emission(const mxArray* X, int len, model* crf, bool is_discrete, double* emission);
void hidden_emission(const mxArray* X, int len, model* crf, bool is_discrete, double* EX, double* emission);
void quadr_emission(const mxArray* X, int len, model* crf, int K, bool is_discrete, double* emission);

void viterbi_linear_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega);
void viterbi_linear_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L);

void viterbi_hidden_crf(const mxArray* X, int len, int D, model* crf, bool is_discrete, int* pred_T);
void viterbi_hidden_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega, double* EX);
void viterbi_hidden_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L);
void viterbi_hidden_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* EX);

void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, int* pred_T, bool is_discrete);
void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, int* pred_T, double* EX, vector<bool>* Z, double* omega, bool is_discrete);
void joint_viterbi_hidden_crf(const mxArray* X, int len, model* crf, const double* EX, int* pred_T, vector<bool>* Z, double* omega, bool is_discrete);
void joint_viterbi_hidden_crf(const mxArray* X, const double* T, int len, model* crf, int* pred_T, double* EX, vector<bool>* Z, double* omega, double rho, bool is_discrete);
void joint_viterbi_hidden_crf(const mxArray* X, const double* T, int len, model* crf, const double* EX, int* pred_T, vector<bool>* Z, double* omega, double rho, bool is_discrete);

void viterbi_quadr_crf(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L, double* omega);
void viterbi_quadr_crf_2nd_order(const mxArray* X, const double* T, double rho, int len, model* crf, bool is_discrete, int* pred_T, double* L);

void viterbi_crf(double* emission, model* crf, const double* T, double rho, int len, int* pred_T, double* L, double* omega);
void viterbi_crf_2nd_order(double* emission, model* crf, const double* T, double rho, int len, int* pred_T, double* L);

double randn();
void randperm(int* array, int n);

inline double abs(double r) { return r > 0 ? r : -r; }
inline double max(int r1, int r2) { return r1 > r2 ? r1 : r2; }

#endif
