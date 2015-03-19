#include "ss-lm.h"
#include "strutil.h"

#include "gflags.h"

DECLARE_string(model);
DECLARE_int32(max_number_time_points);

namespace dtm {

const double kMinLogFrequency = -14.0;

// local functions

static double* scaled_influence = NULL;

void fdf_obs(const gsl_vector *x,
	     void *params,
	     double *f,
	     gsl_vector *df);
void df_obs(const gsl_vector *x,
	    void *params,
	    gsl_vector *df);
double f_obs(const gsl_vector *x, void *params);
double f_obs_fixed(const gsl_vector *x, void *params);

/*
 * allocate a new state space language model variational posterior
 *
 */

sslm_var* sslm_var_alloc(int W, int T) {
  if (scaled_influence == NULL) {
    scaled_influence = NewScaledInfluence(FLAGS_max_number_time_points);
  }


    sslm_var* var = (sslm_var*) malloc(sizeof(sslm_var));
    var->W = W;
    var->T = T;

    // TODO(sgerrish): Free these parameters.
    var->obs = gsl_matrix_calloc(W, T);
    var->e_log_prob = gsl_matrix_calloc(W, T);
    var->mean = gsl_matrix_calloc(W, T+1);
    var->variance = gsl_matrix_calloc(W, T+1);

    var->w_phi_l = gsl_matrix_calloc(W, T);
    var->m_update_coeff = gsl_matrix_calloc(W, T);
    var->m_update_coeff_g = gsl_matrix_calloc(W, T);
    //     var->zeta = gsl_vector_calloc(W);

    return(var);
}

void sslm_inference_alloc(sslm_var* var) {
  int W = var->W;
  int T = var->T;
  var->fwd_mean = gsl_matrix_calloc(W, T+1);
  var->fwd_variance = gsl_matrix_calloc(W, T+1);
  var->zeta = gsl_vector_calloc(T);
  var->T_vct = gsl_vector_calloc(T);
  //    var->mean = gsl_matrix_calloc(W, T+1);
  //var->variance = gsl_matrix_calloc(W, T+1);
}


void sslm_inference_free(sslm_var* var) {
  gsl_matrix_free(var->fwd_mean);
  gsl_matrix_free(var->fwd_variance);
  //    gsl_matrix_free(var->mean);
  //    gsl_matrix_free(var->variance);
  gsl_vector_free(var->zeta);
  gsl_vector_free(var->T_vct);
}


/*
 * initialize with zero observations
 *
 */

void sslm_zero_init(sslm_var* var,
                    double obs_variance,
                    double chain_variance)
{
    int w, W = var->W;

    gsl_matrix_set_zero(var->obs);
    var->obs_variance = obs_variance;
    var->chain_variance = chain_variance;

    for (w = 0; w < W; w++) {
      compute_post_variance(w, var, var->chain_variance);
    }

    for (w = 0; w < W; w++) {
        // Run the forward-backward algorithm
      compute_post_mean(w, var, var->chain_variance);
    }

    update_zeta(var);
}


/*
 * initialize with counts
 *
 */

void sslm_counts_init(sslm_var* var,
                      double obs_variance,
                      double chain_variance,
                      const gsl_vector* counts) {
  int w, t, W = var->W, T = var->T;
  gsl_vector* log_norm_counts = gsl_vector_alloc(counts->size);
  sslm_inference_alloc(var);

  // normalize and take logs of the counts

  gsl_vector_memcpy(log_norm_counts, counts);
  gsl_vector_add_constant(log_norm_counts, 1.0/W);
  normalize(log_norm_counts);
  vct_log(log_norm_counts);

  // set variational observations to transformed counts

  for (t = 0; t < T; t++) {
    msetcol(var->obs, t, log_norm_counts);
  }

  // set variational parameters

  var->obs_variance = obs_variance;
  var->chain_variance = chain_variance;

  for (w = 0; w < W; w++) {
    compute_post_variance(w, var, var->chain_variance);
  }

  for (w = 0; w < W; w++) {
    // Run the forward-backward algorithm
    compute_post_mean(w, var, var->chain_variance);
  }

  update_zeta(var);
  compute_expected_log_prob(var);
  gsl_vector_free(log_norm_counts);
  sslm_inference_free(var);
}


/*
 * initialize from file of variational observations
 *
 */

void sslm_obs_file_init(sslm_var* var,
                        double obs_variance,
                        double chain_variance,
                        const char* filename) {
    int w, W = var->W;
    mtx_fscanf(filename, var->obs);
    // set variational parameters
    var->obs_variance = obs_variance;
    var->chain_variance = chain_variance;
    for (w = 0; w < W; w++) {
      compute_post_variance(w, var, var->chain_variance);
    }
    for (w = 0; w < W; w++) {
        // Run the forward-backward algorithm
      compute_post_mean(w, var, var->chain_variance);
    }
    update_zeta(var);
}


// Compute the expected log probability given values of m.
void compute_expected_log_prob(sslm_var* var) {
  int t, w, W = var->W;
  for (t = 0; t < var->T; t++) {
    for (w = 0; w < W; w++) {
      mset(var->e_log_prob, w, t,
	   mget(var->mean,w,t+1)
	   - log(vget(var->zeta, t)));
      if (mget(var->e_log_prob, w, t) < kMinLogFrequency) {
	mset(var->e_log_prob, w, t, kMinLogFrequency);
      }
    }
  }
}

double expected_log_prob(int w, int t, sslm_var* var) {
  return(mget(var->mean,w,t+1)
	 - log(vget(var->zeta, t)));
}

/*
 * forward-backward to compute E[\beta_{t,w}] for t = 1:T
 *
 */

void compute_post_mean(int word,
                       sslm_var* var,
		       double chain_variance) {
    int t;
    double w;

    // get the vectors for word w
    gsl_vector obs = gsl_matrix_row(var->obs, word).vector;
    gsl_vector mean = gsl_matrix_row(var->mean, word).vector;
    gsl_vector fwd_mean = gsl_matrix_row(var->fwd_mean, word).vector;
    gsl_vector fwd_variance = gsl_matrix_row(var->fwd_variance, word).vector;
    int T = var->T;

    // forward
    // (note, the observation corresponding to mean t is at t-1)
    vset(&fwd_mean, 0, 0);
    for (t = 1; t < T+1; t++)
    {
        // Eq. 1.14
        assert(fabs(vget(&fwd_variance, t-1) +
		    chain_variance + var->obs_variance) > 0.0);
        w = var->obs_variance /
            (vget(&fwd_variance, t-1) +
             chain_variance + var->obs_variance);
        vset(&fwd_mean, t,
             w * vget(&fwd_mean, t-1) + (1 - w) * vget(&obs, t-1));
	if (isnan(vget(&fwd_mean, t))) {
	  outlog("t %d word %d w %f %f %f", t, word, w, vget(&fwd_mean, t - 1), vget(&obs, t-1));
	  outlog("%f %f %f", var->obs_variance, vget(&fwd_variance, t-1),
		 var->obs_variance);
	}

    }

    // backward
    vset(&mean, T, vget(&fwd_mean, T));
    for (t = T-1; t >= 0; t--)
    {
        // Eq. 1.18
        if (chain_variance == 0.0) {
	  w = 0.0;
	} else {
	  w = chain_variance /
            (vget(&fwd_variance, t) + chain_variance);
	}
        vset(&mean, t, w * vget(&fwd_mean, t) + (1 - w) * vget(&mean, t+1));
	if (isnan(vget(&mean, t))) {
	  outlog("t %d w %f %f %f", t, w, vget(&fwd_mean, t), vget(&mean, t+1));
	}
        assert(!isnan(vget(&mean, t)));
    }
}


/*
 * update zeta variational parameter
 *
 */

void update_zeta(sslm_var* var)
{
    int word;
    int t;

    gsl_vector_set_zero(var->zeta);
    // gsl_matrix_set_zero(var->zeta_terms);
    for (word = 0; word < var->obs->size1; word++)
    {
        for (t = 0; t < var->obs->size2; t++)
        {
            double m = mget(var->mean, word, t+1);
            double v = mget(var->variance, word, t+1);
            double val = exp(m + v/2.0);
            // mset(var->zeta_terms, word, t, val);
            vinc(var->zeta, t, val);
        }
    }
}


/*
 * compute Var[\beta_{t,w}] for t = 1:T
 *
 */

void compute_post_variance(int word,
                           sslm_var* var,
			   double chain_variance) {
    int t, T = var->T;
    double w;

    // get the variance vector for word
    gsl_vector variance = gsl_matrix_row(var->variance, word).vector;
    gsl_vector fwd_variance = gsl_matrix_row(var->fwd_variance, word).vector;

    // forward
    // note, initial variance set very large
    vset(&fwd_variance, 0, chain_variance * INIT_MULT);
    for (t = 1; t < T+1; t++)
    {
        // Eq. 1.16
      if (var->obs_variance) {
        w = var->obs_variance /
	  (vget(&fwd_variance, t-1) +
	   chain_variance + var->obs_variance);
      } else {
	w = 0.0;
      }
      vset(&fwd_variance, t, w * (vget(&fwd_variance, t-1) +
				  chain_variance));
    }

    // backward
    vset(&variance, T, vget(&fwd_variance, T));
    for (t = T-1; t >= 0; t--)
    {
        // Eq. 1.21
      if (vget(&fwd_variance, t) > 0.0) {
        w = pow(vget(&fwd_variance, t) /
                (vget(&fwd_variance, t) + chain_variance), 2);
      } else {
	w = 0.0;
      }
      vset(&variance, t,
	   w * (vget(&variance, t+1) - chain_variance) +
	   (1 - w) * vget(&fwd_variance, t));
    }
}


/*
 * compute d E[\beta_{t,w}]/d obs_{s,w} for t = 1:T.
 * put the result in deriv, allocated T+1 vector
 *
 */

void compute_mean_deriv(int word,
                        int time,
                        sslm_var* var,
                        gsl_vector* deriv)
{
    int t, T = var->T;
    double w, val;

    // get the vectors for word
    gsl_vector fwd_variance = gsl_matrix_row(var->variance, word).vector;
    assert(deriv->size == T+1);

    // forward
    // (note, we don't need to store the forward pass in this case.)
    vset(deriv, 0, 0);
    for (t = 1; t <= T; t++) {
        // Eq. 1.37
        if (var->obs_variance > 0.0) {
	  w = var->obs_variance /
            (vget(&fwd_variance, t-1) +
             var->chain_variance + var->obs_variance);
	} else {
	  w = 0.0;
	}
        val = w * vget(deriv, t-1);
        // note that observations are indexed 1 from the means/variances
        if (time == t-1) {
	  val += (1 - w);
	}
        vset(deriv, t, val);
    }

    // backward
    for (t = T-1; t >= 0; t--)
    {
        // Eq. 1.39
        if (var->chain_variance == 0.0) {
	  w = 0.0;
        } else {
	  w = var->chain_variance /
            (vget(&fwd_variance, t) + var->chain_variance);
	}
	vset(deriv, t, w * vget(deriv, t) + (1 - w) * vget(deriv, t+1));
    }
}


/*
 * compute d bound/d obs_{w, t} for t=1:T.
 * put the result in deriv, allocated T vector
 *
 */

void compute_obs_deriv(int word,
                       gsl_vector* word_counts,
                       gsl_vector* totals,
                       sslm_var* var,
                       gsl_matrix* mean_deriv_mtx,
                       gsl_vector* deriv) {
  // Note: We found a seg fault in this function at
  // the string compare FLAGS_model == "dim".
  // We suspect there is memory corruption occurring
  // at some point.
  int t, u, T = var->T;

  gsl_vector mean = gsl_matrix_row(var->mean, word).vector;
  gsl_vector variance = gsl_matrix_row(var->variance, word).vector;
  
  // here the T vector in var is the zeta terms
  for (u = 0; u < T; u++) {
    vset(var->T_vct, u,
	 exp(vget(&mean, u+1) + vget(&variance, u+1)/2));
  }

  gsl_vector w_phi_l = gsl_matrix_row(var->w_phi_l, word).vector;
  gsl_vector m_update_coeff = gsl_matrix_row(var->m_update_coeff, word).vector;
  
  for (t = 0; t < T; t++) {
    gsl_vector mean_deriv = gsl_matrix_row(mean_deriv_mtx, t).vector;
    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;
    double term4 = 0.0;
    for (u = 1; u <= T; u++) {
      double mean_u = vget(&mean, u);
      double var_u_prev = vget(&variance, u - 1);
      double mean_u_prev = vget(&mean, u - 1);
      double dmean_u = vget(&mean_deriv, u);
      double dmean_u_prev = vget(&mean_deriv, u - 1);
      
      double dmean_u_window_prev = 0.0;
      double var_u_window_prev = 0.0;
      double mean_u_window_prev = 0.0;
      
      term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev);
      
      // note, observations indexed -1 from mean and variance
      term2 +=
	(vget(word_counts, u-1) -
	 (vget(totals, u-1) *
	  vget(var->T_vct, u-1) /
	  vget(var->zeta, u-1))) * dmean_u;
      
      if ((FLAGS_model == "dim"
	   || FLAGS_model == "fixed")
	  && u >= 1
	  && var->chain_variance > 0.0) {
	double dmean_u_window_prev = vget(&mean_deriv, u - 1);
	double var_u_window_prev = vget(&variance, u - 1);
	double mean_u_window_prev = vget(&mean, u - 1);
	
	term3 += (exp(- (mean_u_window_prev - 0.0 * vget(var->zeta, u-1)) + var_u_window_prev / 2.0)
		  * ((- mean_u + mean_u_window_prev - var_u_window_prev - 1)
		     * (dmean_u_window_prev)
		     + (dmean_u))
		  * vget(&w_phi_l, u - 1)
		  / var->chain_variance);
	term4 += (exp( -2.0 * (mean_u_window_prev - 0.0 * vget(var->zeta, u-1)) + 2.0 * var_u_window_prev)
		  * (vget(&m_update_coeff, u - 1))
		  / var->chain_variance
		  * dmean_u_window_prev);
      }
    }
    if (var->chain_variance) {
      term1 = -term1/var->chain_variance;
      term1 = term1 -
	vget(&mean, 0) * vget(&mean_deriv, 0) /
	(INIT_MULT * var->chain_variance);
    } else {
      term1 = 0.0;
    }

    // Enforce a hinge penalty for straying too low.
    double m = term1 + term2 + term3 + term4;
    vset(deriv, t, m);
  }
}

void compute_obs_deriv_fixed(int word,
			     gsl_vector* word_counts,
			     gsl_vector* totals,
			     sslm_var* var,
			     gsl_matrix* mean_deriv_mtx,
			     gsl_vector* deriv)
{
  // Note: We found a seg fault in this function at
  // the string compare FLAGS_model == "dim".
  // We suspect there is memory corruption occurring
  // at some point.
  int t, u, T = var->T;

  gsl_vector mean = gsl_matrix_row(var->mean, word).vector;
  gsl_vector variance = gsl_matrix_row(var->variance, word).vector;
  
  // here the T vector in var is the zeta terms
  for (u = 0; u < T; u++) {
    vset(var->T_vct, u,
	 exp(vget(&mean, u+1) + vget(&variance, u+1)/2));
  }
  
  gsl_vector m_update_coeff_g = gsl_matrix_row(
    var->m_update_coeff_g, word).vector;
  gsl_vector m_update_coeff_h = gsl_matrix_row(
    var->m_update_coeff, word).vector;

  for (t = 0; t < T; t++) {
    gsl_vector mean_deriv = gsl_matrix_row(mean_deriv_mtx, t).vector;
    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;
    double term4 = 0.0;
    double min_penalty = 0.0;
    for (u = 1; u <= T; u++) {
      double mean_u = vget(&mean, u);
      double var_u_prev = vget(&variance, u - 1);
      double mean_u_prev = vget(&mean, u - 1);
      double dmean_u = vget(&mean_deriv, u);
      double dmean_u_prev = vget(&mean_deriv, u - 1);
      
      double dmean_u_window_prev = 0.0;
      double var_u_window_prev = 0.0;
      double mean_u_window_prev = 0.0;

      if (mean_u < kMinLogFrequency) {
	min_penalty += (kMinLogFrequency - mean_u) * 1000.0;
      }

      term1 -= ((mean_u - mean_u_prev)
		* (dmean_u - dmean_u_prev)
		/ var->chain_variance);

      // note, observations indexed -1 from mean and variance
      term2 +=
	(vget(word_counts, u-1) -
	 (vget(totals, u-1) *
	  vget(var->T_vct, u-1) /
	  vget(var->zeta, u-1))) * dmean_u;
      
      if ((FLAGS_model == "fixed")
	  && u >= 1
	  && var->chain_variance > 0.0) {
	double dmean_u_window_prev = vget(&mean_deriv, u - 1);
	double var_u_window_prev = vget(&variance, u - 1);
	double mean_u_window_prev = vget(&mean, u - 1);
	
	term3 += - (exp(- (mean_u_window_prev - 0.0 * vget(var->zeta, u-1)) + var_u_window_prev / 2.0)
		    * ((mean_u - mean_u_window_prev + var_u_window_prev - 1)
		       * (dmean_u_window_prev)
		       + (dmean_u))
		    * vget(&m_update_coeff_g, u - 1)
		    / var->chain_variance);
	term4 += (exp( -2.0 * (mean_u_window_prev - 0.0 * vget(var->zeta, u-1)) + 2.0 * var_u_window_prev)
		  * (vget(&m_update_coeff_h, u - 1))
		  / var->chain_variance
		  * dmean_u_window_prev);
      }
    }
    if (var->chain_variance) {
      term1 = term1 -
	vget(&mean, 0) * vget(&mean_deriv, 0) /
	(INIT_MULT * var->chain_variance);
    } else {
      term1 = 0.0;
    }
    
    // min_penalty = 0.0;
    double m = term1 + term2 + term3 + term4 + min_penalty;
    vset(deriv, t, m);
  }
}


/*
 * log probability bound
 *
 */

double compute_bound(gsl_matrix* word_counts,
                     gsl_vector* totals,
                     sslm_var* var) {
  int t, T = var->T, w, W = var->W;
  double term1 = 0, term2 = 0, term3 = 0;
  double val = 0, m, v, prev_m, prev_v;
  double w_phi_l, exp_i;
  double ent = 0;

  const double chain_variance = var->chain_variance;

  for (w = 0; w < W; w++) {
    // Run the forward-backward algorithm
    compute_post_mean(w, var, chain_variance);
  }
  update_zeta(var);
  
  for (w = 0; w < W; w++)
    {
      val +=
	(mget(var->variance, w, 0) -
	 mget(var->variance, w, T))/
	2 * chain_variance;
    }
  
  outlog("Computing bound, all times.%s", "");
  for (t = 1; t <= T; t++)
    {
      term1 = term2 = ent = 0.0;
      for (w = 0; w < W; w++)
        {
	  m = mget(var->mean, w, t);
	  prev_m = mget(var->mean, w, t - 1);
	  v = mget(var->variance, w, t);
	  
	  // Values specifically related to document influence:
	  // Note that our indices are off by 1 here.
	  w_phi_l = mget(var->w_phi_l, w, t - 1);
	  exp_i = exp(-prev_m);
	  
	  term1 +=
	    pow(m - prev_m - w_phi_l * exp_i, 2) / (2 * chain_variance) -
	    v/chain_variance -
	    log(chain_variance);
	  
	  term2 += mget(word_counts, w, t-1) * m;
	  ent   += log(v) / 2; // note the 2pi's cancel with term1 (see doc)
        }
      term3 = -vget(totals, t-1) * log(vget(var->zeta,t-1));
      
      // printf("time %d contribute: %20f\n", t,
      // -term1/(2.0*var->chain_variance) + term2 + term3 + ent);
      /*	outlog("contributions: %f %f %f %f %f",
	term1,
	term2,
	term3,
	term4,
	term5);
      */
      val += - term1 + term2 + term3 + ent;
    }
  return(val);
}

double compute_bound_fixed(gsl_matrix* word_counts,
			   gsl_vector* totals,
			   sslm_var* var) {
    int t, T = var->T, w, W = var->W;
    double term1 = 0, term2 = 0, term3 = 0, term4 = 0, term5 = 0;
    double val = 0, m, v, prev_m, prev_v;
    double m_update_coeff_g, m_update_coeff_h, exp_i, exp_2i;
    double ent = 0;

    const double chain_variance = var->chain_variance;

    for (w = 0; w < W; w++) {
        // Run the forward-backward algorithm
      compute_post_mean(w, var, chain_variance);
    }
    update_zeta(var);

    for (w = 0; w < W; w++)
    {
        val +=
            (mget(var->variance, w, 0) -
             mget(var->variance, w, T))/
            2 * chain_variance;
    }

    outlog("Computing bound, all times.%s", "");
    for (t = 1; t <= T; t++)
    {
        term1 = term2 = term4 = term5 = ent = 0.0;
        for (w = 0; w < W; w++)
        {
            m = mget(var->mean, w, t);
            prev_m = mget(var->mean, w, t - 1);
	    if (m < kMinLogFrequency) {
	      m = kMinLogFrequency;
	    }
	    if (prev_m < kMinLogFrequency) {
	      prev_m = kMinLogFrequency;
	    }
            v = mget(var->variance, w, t);

	    // Values specifically related to document influence:
	    // Note that our indices are off by 1 here.
            m_update_coeff_g = mget(var->m_update_coeff_g, w, t - 1);
            m_update_coeff_h = mget(var->m_update_coeff, w, t - 1);
	    exp_i = exp(-prev_m + v / 2.0);
	    exp_2i = exp(2.0 * (-prev_m + v));

            term1 +=
	      - pow(m - prev_m, 2) / (2.0 * chain_variance)
	      - v / chain_variance
	      - log(chain_variance);

            term2 += mget(word_counts, w, t-1) * m;

	    term4 +=
	      ((m - prev_m + v) * m_update_coeff_g * exp_i
	       / (chain_variance));

	    term5 += -exp_2i * m_update_coeff_h / (2.0 * chain_variance);

	    ent   += log(v) / 2; // note the 2pi's cancel with term1 (see doc)
        }

        term3 = -vget(totals, t-1) * log(vget(var->zeta,t-1));

        val += term1 + term2 + term3 + term4 + ent;
    }
    return(val);
}

/*
 * update obs
 *
 */

// parameters object
struct opt_params
{
    sslm_var* var;
    gsl_vector* word_counts;
    gsl_vector* totals;
    gsl_matrix* mean_deriv_mtx;
    int word;
};

// objective function
double f_obs(const gsl_vector *x, void *params)
{
    int t, T = x->size;
    double val = 0, term1 = 0, term2 = 0, term3 = 0, term4 = 0;
    struct opt_params * p = (struct opt_params *) params;
    gsl_vector mean, variance, w_phi_l, m_update_coeff;

    msetrow(p->var->obs, p->word, x);

    // Run the forward-backward algorithm
    compute_post_mean(p->word, p->var, p->var->chain_variance);
    mean = gsl_matrix_row(p->var->mean, p->word).vector;
    variance = gsl_matrix_row(p->var->variance, p->word).vector;
    w_phi_l = gsl_matrix_row(p->var->w_phi_l, p->word).vector;
    m_update_coeff = gsl_matrix_row(p->var->m_update_coeff, p->word).vector;

    // Only compute the objective if the chain variance 
    for (t = 1; t <= T; t++) {
        double mean_t = vget(&mean, t);
        double mean_t_prev = vget(&mean, t-1);
	double var_t_prev = vget(&variance, t - 1);
        val = mean_t - mean_t_prev;
        term1 += - (val * val) / (2.0 * p->var->chain_variance);

        // note, badly indexed counts
        term2 += (vget(p->word_counts, t-1) * mean_t -
		  vget(p->totals, t-1) *
		  (exp(mean_t + vget(&variance, t)/2)
		   / vget(p->var->zeta, t-1)));
		  // log(vget(p->var->zeta, t-1)));

	if (FLAGS_model != "fixed") {
	  assert(0);
	}

	if ((FLAGS_model == "dim"
	     || FLAGS_model == "fixed")
	    && t >= 1
	    && p->var->chain_variance > 0.0) {
	  double mean_t_window_prev = vget(&mean, t - 1);
	  double var_t_window_prev = vget(&variance, t - 1);
	  term3 += (exp(- (mean_t_window_prev - 0.0 * vget(p->var->zeta, t-1)) + var_t_window_prev / 2.0)
		    * (mean_t - mean_t_window_prev + var_t_window_prev)
		    * vget(&w_phi_l, t - 1)
		    / p->var->chain_variance);

	  term4 -= (exp( -2.0 * (mean_t_window_prev
				 - 0.0 * vget(p->var->zeta, t-1))
			 + 2.0 * var_t_window_prev)
		    * (vget(&m_update_coeff, t - 1))
		    / (2.0 * p->var->chain_variance));
	}
    }
    // note that we multiply the initial variance by INIT_MULT
    if (p->var->chain_variance > 0.0) {
      term1 = (term1 -
	       vget(&mean, 0) * vget(&mean, 0) /
	       (2 * INIT_MULT * p->var->chain_variance));
    } else {
      term1 = 0.0;
    }

    return(-(term1 + term2 + term3 + term4));
}

// Objective function for fixed model.
double f_obs_fixed(const gsl_vector *x, void *params)
{
    int t, T = x->size;
    double val = 0, term1 = 0, term2 = 0, term3 = 0, term4 = 0, min_penalty = 0;
    struct opt_params * p = (struct opt_params *) params;
    gsl_vector mean, variance, w_phi_l, m_update_coeff_h, m_update_coeff_g;

    msetrow(p->var->obs, p->word, x);

    // Run the forward-backward algorithm
    compute_post_mean(p->word, p->var, p->var->chain_variance);
    mean = gsl_matrix_row(p->var->mean, p->word).vector;
    variance = gsl_matrix_row(p->var->variance, p->word).vector;
    m_update_coeff_h = gsl_matrix_row(
      p->var->m_update_coeff, p->word).vector;
    m_update_coeff_g = gsl_matrix_row(
      p->var->m_update_coeff_g, p->word).vector;

    // Only compute the objective if the chain variance 
    for (t = 1; t <= T; t++) {
        double mean_t = vget(&mean, t);
        double mean_t_prev = vget(&mean, t-1);
	double var_t_prev = vget(&variance, t - 1);

	if (mean_t < kMinLogFrequency) {
	  double x = (kMinLogFrequency - mean_t);
	  min_penalty += -(x * x) * 1000.0 / 2.0;
	}

        val = mean_t - mean_t_prev;
        term1 += -(val * val / (2.0 * p->var->chain_variance));

        // note, badly indexed counts
        term2 += (vget(p->word_counts, t-1) * mean_t -
		  vget(p->totals, t-1) *
		  (exp(mean_t + vget(&variance, t) / 2.0)
		   / vget(p->var->zeta, t-1)));
		  // log(vget(p->var->zeta, t-1)));
	if (FLAGS_model != "fixed") {
	  assert(0);
	}

	if (t >= 1
	    && p->var->chain_variance > 0.0) {
	  double mean_t_window_prev = vget(&mean, t - 1);
	  double var_t_window_prev = vget(&variance, t - 1);
	  term3 += (exp(-mean_t_window_prev + var_t_window_prev / 2.0)
		    * (mean_t - mean_t_window_prev + var_t_window_prev)
		    * vget(&m_update_coeff_g, t - 1)
		    / p->var->chain_variance);
	
	  term4 -= (exp( -2.0 * mean_t_window_prev
			 + 2.0 * var_t_window_prev)
		    * (vget(&m_update_coeff_h, t - 1))
		    / (2.0 * p->var->chain_variance));
	}
    }
    // note that we multiply the initial variance by INIT_MULT
    term1 = (term1 -
	     vget(&mean, 0) * vget(&mean, 0) /
	     (2 * INIT_MULT * p->var->chain_variance));
    // min_penalty = 0.0;
    double m = term1 + term2 + term3 + term4 + min_penalty;
    return(-m);
}

// derivative
void df_obs(const gsl_vector *x, void *params, gsl_vector *df)
{
    struct opt_params * p = (struct opt_params *) params;
    int i;

    msetrow(p->var->obs, p->word, x);

    // Run the forward-backward algorithm
    compute_post_mean(p->word, p->var, p->var->chain_variance);

    if (FLAGS_model == "fixed") {
      compute_obs_deriv_fixed(p->word,
			      p->word_counts,
			      p->totals,
			      p->var,
			      p->mean_deriv_mtx,
			      df);
    } else {
      compute_obs_deriv(p->word,
			p->word_counts,
			p->totals,
			p->var,
			p->mean_deriv_mtx,
			df);
    }

    for (i = 0; i < df->size; i++) {
        vset(df, i, -vget(df, i));
    }
}

// function and derivative
void fdf_obs(const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
    struct opt_params * p = (struct opt_params *) params;
    int i;

    if (FLAGS_model == "fixed") {
      *f = f_obs_fixed(x, params);
      compute_obs_deriv_fixed(p->word,
			      p->word_counts,
			      p->totals,
			      p->var,
			      p->mean_deriv_mtx,
			      df);
    } else {
      *f = f_obs(x, params);
      compute_obs_deriv(p->word,
			p->word_counts,
			p->totals,
			p->var,
			p->mean_deriv_mtx,
			df);
    }

    for (i = 0; i < df->size; i++) {
        vset(df, i, -vget(df, i));
    }
}


// function to perform optimization
void update_obs(gsl_matrix* word_counts,
                gsl_vector* totals,
                sslm_var* var)
{
    int t, w, T = var->T, W = var->W, runs = 0;
    double f_val, conv_val; int niter;
    gsl_vector obs, w_counts, mean_deriv;
    gsl_vector * norm_cutoff_obs = NULL;
    struct opt_params params;
    // Matrix of mean derivatives:
    // Row ~ s
    gsl_matrix* mean_deriv_mtx = gsl_matrix_alloc(T, T+1);

    params.var = var;
    params.totals = totals;

    for (w = 0; w < W; w++)
    {
        if (w % 5000 == 0) {
	  outlog("Updating term %d", w);
	}
        w_counts = gsl_matrix_row(word_counts, w).vector;
        if (((w % 500) == 0) && (w > 0))
        {
            // outlog( "[SSLM] optimized %05d words (%05d cg runs)", w, runs);
        }

        // check norm of observation

        double counts_norm = norm(&w_counts);
        if ((counts_norm < OBS_NORM_CUTOFF) && (norm_cutoff_obs != NULL))
        {
            obs = gsl_matrix_row(var->obs, w).vector;
            gsl_vector_memcpy(&obs, norm_cutoff_obs);
        }
        else
        {
            if (counts_norm < OBS_NORM_CUTOFF)
                gsl_vector_set_all(&w_counts, 0);

            // compute mean deriv
            for (t = 0; t < T; t++) {
                mean_deriv = gsl_matrix_row(mean_deriv_mtx, t).vector;
                compute_mean_deriv(w, t, var, &mean_deriv);
            }
            // set parameters
            params.word_counts = &w_counts;
            params.word = w;
            params.mean_deriv_mtx = mean_deriv_mtx;
            obs = gsl_matrix_row(var->obs, w).vector;
            // optimize
	    if (FLAGS_model == "fixed") {
	      optimize_fdf(T, &obs, &params, &fdf_obs, &df_obs, &f_obs_fixed,
			   &f_val, &conv_val, &niter);
	    } else {
	      optimize_fdf(T, &obs, &params, &fdf_obs, &df_obs, &f_obs,
			   &f_val, &conv_val, &niter);
	    }
            runs++;
            // outlog(
            // "w: %04d  f: %1.15e  c: %5.3e  n: %04d",
            // w, f_val, conv_val, niter);
            if (counts_norm < OBS_NORM_CUTOFF)
            {
                norm_cutoff_obs = gsl_vector_alloc(T);
                // !!! this can be BLASified for speed
                gsl_vector_memcpy(norm_cutoff_obs, &obs);
            }
        }
    }
    update_zeta(var);
    gsl_matrix_free(mean_deriv_mtx);
    if (norm_cutoff_obs != NULL)
    {
        gsl_vector_free(norm_cutoff_obs);
    }
}


/*
 * initialize variational observations
 *
 */

void initialize_obs(sslm_var* var, gsl_matrix* counts)
{
    gsl_matrix_set_zero(var->obs);
#if 0
    int t;

    gsl_matrix_memcpy(var->obs, counts);
    for (t = 0; t < counts->size2; t++)
    {
        gsl_vector time_slice = gsl_matrix_column(var->obs, t).vector;
        gsl_vector_add_constant(&time_slice, 1.0);
        normalize(&time_slice);
        vct_log(&time_slice);
    }
#endif
}


/*
 * fit variational distribution
 *
 */

double fit_sslm(sslm_var* var, gsl_matrix* counts) {
  int iter, w, W = var->W;
  double bound = 0, old_bound = 0;
  double converged = SSLM_FIT_THRESHOLD+1;
  gsl_vector* totals = gsl_vector_alloc(counts->size2);

  // Allocate variational data.
  sslm_inference_alloc(var);
  for (w = 0; w < W; w++) {
    compute_post_variance(w, var, var->chain_variance);
  }

  col_sum(counts, totals);

  iter = 0;
  if (FLAGS_model == "fixed") {
    bound = compute_bound_fixed(counts, totals, var);
    //     bound = compute_bound(counts, totals, var);
  } else {
    bound = compute_bound(counts, totals, var);
  }
  outlog( "initial sslm bound = %10.5f", bound);

  while ((converged > SSLM_FIT_THRESHOLD)
	 && (iter < SSLM_MAX_ITER)) {
    iter++;
    old_bound = bound;
    update_obs(counts, totals, var);
    if (FLAGS_model == "fixed") {
      bound = compute_bound_fixed(counts, totals, var);
    } else {
      bound = compute_bound(counts, totals, var);
    }
    converged = fabs((bound - old_bound) / old_bound);
    outlog( "(%02d) sslm bound = % 10.5f; conv = % 10.5e",
	    iter, bound, converged);
  }

  // compute expected log probability

  compute_expected_log_prob(var);

  // free convenience parameters (there are a lot of them)

  free(totals);
  sslm_inference_free(var);

  return(bound);
}


/*
 * read and write variational distribution
 *
 */

void write_sslm_var(sslm_var* var, const char* out) {
  string filename = StringPrintf("%s-var-obs.dat", out);
  mtx_fprintf(filename.c_str(), var->obs);

  filename = StringPrintf("%s-var-e-log-prob.dat", out);
  mtx_fprintf(filename.c_str(), var->e_log_prob);

    // sprintf(filename, "%s-zeta.dat", out);
    // vct_fprintf(filename, var->zeta);

  filename = StringPrintf("%s-info.dat", out);
  FILE* f = fopen(filename.c_str(), "w");
  params_write_int(f, "SEQ_LENGTH", var->T);
  params_write_int(f, "NUM_TERMS", var->W);
  params_write_double(f, "OBS_VARIANCE", var->obs_variance);
  params_write_double(f, "CHAIN_VARIANCE", var->chain_variance);

  fclose(f);
}


// Read in a set of pre-computed topics from a file whose
// prefix is "in".
sslm_var* read_sslm_var(const char* in) {
    outlog("READING LM FROM %s", in);

    string filename;
    int W, T;

    // read number of words and sequence length; allocate distribution

    filename = StringPrintf("%s-info.dat", in);
    FILE* f = fopen(filename.c_str(), "r");
    if (f == NULL) {
      printf("Error.  Could not open file %s\nFailing.\n", filename.c_str());
      exit(1);
    }

    params_read_int(f, "SEQ_LENGTH", &T);
    params_read_int(f, "NUM_TERMS", &W);
    sslm_var* var = sslm_var_alloc(W, T);

    // read the variance parameters

    params_read_double(f, "OBS_VARIANCE", &(var->obs_variance));
    params_read_double(f, "CHAIN_VARIANCE", &(var->chain_variance));

    // read the variational observations and expected log probabilities

    filename = StringPrintf("%s-var-obs.dat", in);
    mtx_fscanf(filename.c_str(), var->obs);
    filename = StringPrintf("%s-var-e-log-prob.dat", in);
    mtx_fscanf(filename.c_str(), var->e_log_prob);

    // sprintf(filename, "%s-zeta.dat", in);
    // vct_fscanf(filename, var->zeta);

    return(var);
}

}  // namespace dtm
