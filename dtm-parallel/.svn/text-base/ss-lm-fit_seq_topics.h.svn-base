/*
 * state space language model variational inference
 *
 */

#ifndef SSLM_H
#define SSLM_H

#include "gsl-wrappers.h"
#include "params.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <assert.h>
#include <math.h>

#include "data.h"

#define SSLM_MAX_ITER 2 // maximum number of optimization iters
#define SSLM_FIT_THRESHOLD 1e-6 // convergence criterion for fitting sslm
#define INIT_MULT 1000 // multiplier to variance for first obs
// #define OBS_NORM_CUTOFF 10 // norm cutoff after which we use all 0 obs
//#define OBS_NORM_CUTOFF 8 // norm cutoff after which we use all 0 obs
#define OBS_NORM_CUTOFF 2 // norm cutoff after which we use all 0 obs

namespace dtm {

/*
 * functions for variational inference
 *
 */

// allocate new state space language model variational posterior
sslm_var* sslm_var_alloc(int W, int T);

// allocate extra parameters for inference
void sslm_inference_alloc(sslm_var* var);

// free extra parameters for inference
void sslm_inference_free(sslm_var* var);

// initialize with zero observations
void sslm_zero_init(sslm_var* var,
                    double obs_variance,
                    double chain_variance);

// initialize with counts
void sslm_counts_init(sslm_var* var,
                      double obs_variance,
                      double chain_variance,
                      const gsl_vector* counts);

// initialize from variational observations
void sslm_obs_file_init(sslm_var* var,
                        double obs_variance,
                        double chain_variance,
                        const char* filename);


// compute E[\beta_{w,t}] for t = 1:T
void compute_post_mean(int w, sslm_var* var, double chain_variance);

// compute Var[\beta_{w,t}] for t = 1:T
void compute_post_variance(int w, sslm_var* var, double chain_variance);

// optimize \hat{beta}
void optimize_var_obs(sslm_var* var);

// compute dE[\beta_{w,t}]/d\obs_{w,s} for t = 1:T
void compute_mean_deriv(int word, int time, sslm_var* var,
                        gsl_vector* deriv);

// compute d bound/d obs_{w, t} for t=1:T.
void compute_obs_deriv(int word, gsl_vector* word_counts,
                       gsl_vector* total_counts, sslm_var* var,
                       gsl_matrix* mean_deriv_mtx, gsl_vector* deriv);

// update observations
void update_obs(gsl_matrix* word_counts, gsl_vector* totals,
                sslm_var* var);

// log probability bound
double compute_bound(gsl_matrix* word_counts, gsl_vector* totals,
                     sslm_var* var);


// fit variational distribution
double fit_sslm(sslm_var* var, gsl_matrix* word_counts);

// read and write variational distribution
void write_sslm_var(sslm_var* var, const char* out);
sslm_var* read_sslm_var(char* in);

void compute_expected_log_prob(sslm_var* var);
// !!! old function (from doc mixture...)
double expected_log_prob(int w, int t, sslm_var* var);

// update zeta
void update_zeta(sslm_var* var);

}  // namespace dtm

#endif
