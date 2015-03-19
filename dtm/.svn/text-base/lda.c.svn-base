// Authors: David Blei (blei@cs.princeton.edu)
//          Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Copyright 2011 Sean Gerrish and David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#include "gflags.h"

#include "lda.h"

int LDA_INFERENCE_MAX_ITER=25;

// Used for the phi update: how small can the smallest phi be?
// We assert around 1e-10, since that is around the smallest beta
// for a term.
static const double kSmallLogNumber = -100.0;
static const double kSmallNumber = exp(kSmallLogNumber);

static double* scaled_influence = NULL;

DEFINE_double(lambda_convergence,
	      0.01,
	      "Specifies the level of convergence required for "
	      "lambda in the phi updates.");

DECLARE_string(normalize_docs);
DECLARE_int32(max_number_time_points);
DECLARE_string(model);
DECLARE_string(mode);
DECLARE_double(sigma_d);
DECLARE_double(sigma_l);
DECLARE_int32(forward_window);

/*
 * posterior inference for lda
 * time and doc_number are only necessary if
 * var is not NULL.
 */

double fit_lda_post(int doc_number, int time,
		    lda_post* p, lda_seq* var,
		    gsl_matrix* g,
		    gsl_matrix* g3_matrix,
		    gsl_matrix* g4_matrix,
		    gsl_matrix* g5_matrix) {
    init_lda_post(p);
    gsl_vector_view topic_view;
    gsl_vector_view renormalized_topic_view;
    if (FLAGS_model == "fixed" && var && var->influence) {
      // Make sure this stays in scope while the posterior is in
      // use!
      topic_view = gsl_matrix_row(
          var->influence->doc_weights[time], doc_number);
      renormalized_topic_view = gsl_matrix_row(
          var->influence->renormalized_doc_weights[time], doc_number);
      p->doc_weight = &topic_view.vector;
      p->renormalized_doc_weight = &renormalized_topic_view.vector;
    }

    double lhood = compute_lda_lhood(p);
    double lhood_old = 0;
    double converged = 0;
    int iter = 0;

    do {
        iter++;
        lhood_old = lhood;
        update_gamma(p);
	if (FLAGS_model == "fixed" && var != NULL) {
	  update_phi_fixed(doc_number,
			   time,
			   p,
			   var,
			   g3_matrix,
			   g4_matrix,
			   g5_matrix);
	} else if (FLAGS_model == "dtm" || var == NULL) {
	  update_phi(doc_number, time, p, var, g);
	} else {
	  printf("Error.  Unhandled model.\n");
	  exit(1);
	}
	// TODO(sgerrish): Remove this.
	// output_phi(p);
        lhood = compute_lda_lhood(p);
        converged = fabs((lhood_old - lhood) /
			 (lhood_old * p->doc->total));
    } while ((converged > LDA_INFERENCE_CONVERGED) &&
	     (iter <= LDA_INFERENCE_MAX_ITER));

    return(lhood);
}


/*
 * initialize variational posterior
 *
 */

void init_lda_post(lda_post* p) {
    int k, n, K = p->model->ntopics, N = p->doc->nterms;

    for (k = 0; k < K; k++)
    {
        vset(p->gamma, k,
             vget(p->model->alpha,k) + ((double) p->doc->total)/K);
        for (n = 0; n < N; n++)
            mset(p->phi, n, k, 1.0/K);
    }
    p->doc_weight = NULL;
}

/*
 * update variational dirichlet parameters
 *
 */

void update_gamma(lda_post* p) {
    int k, n, K = p->model->ntopics, N = p->doc->nterms;

    gsl_vector_memcpy(p->gamma, p->model->alpha);
    for (n = 0; n < N; n++)
    {
        gsl_vector phi_row = gsl_matrix_row(p->phi, n).vector;
        int count = p->doc->count[n];
        for (k = 0; k < K; k++)
            vinc(p->gamma, k, vget(&phi_row, k) * count);
    }
}

/*
 * update variational multinomial parameters
 *
 */

void update_phi(int doc_number, int time,
		lda_post* p, lda_seq* var,
		gsl_matrix* g) {
    int i, k, n, K = p->model->ntopics, N = p->doc->nterms;
    double dig[p->model->ntopics];

    for (k = 0; k < K; k++) {
      dig[k] = gsl_sf_psi(vget(p->gamma, k));
    }

    for (n = 0; n < N; n++) {
      // compute log phi up to a constant

      int w = p->doc->word[n];
      for (k = 0; k < K; k++) {
	mset(p->log_phi, n, k,
	     dig[k] + mget(p->model->topics, w, k));
      }

      // normalize in log space

      gsl_vector log_phi_row = gsl_matrix_row(p->log_phi, n).vector;
      gsl_vector phi_row = gsl_matrix_row(p->phi, n).vector;
      log_normalize(&log_phi_row);
      for (i = 0; i < K; i++) {
	vset(&phi_row, i, exp(vget(&log_phi_row, i)));
      }
    }
}


void update_phi_fixed(int doc_number, int time,
		      lda_post* p, lda_seq* var,
		      gsl_matrix* g3_matrix,
		      gsl_matrix* g4_matrix,
		      gsl_matrix* g5_matrix) {
    // Hate to do this, but I had problems allocating this data
    // structure.
    if (scaled_influence == NULL) {
      scaled_influence = NewScaledInfluence(FLAGS_max_number_time_points);
    }

    int i, k, n, K = p->model->ntopics, N = p->doc->nterms;
    double dig[p->model->ntopics];

    double k_sum = 0.0;
    for (k = 0; k < K; k++) {
        double gamma_k = vget(p->gamma, k);
        dig[k] = gsl_sf_psi(gamma_k);
	k_sum += gamma_k;
    }
    double dig_sum = gsl_sf_psi(k_sum);

    gsl_vector_view document_weights;
    if (var && var->influence) {
      document_weights = gsl_matrix_row(
      var->influence->doc_weights[time], doc_number);
    }

    for (n=0; n < N; ++n) {
      int w = p->doc->word[n];
      // We have info. about the topics. Use them!
      // Try two alternate approaches.  We compare results of the new
      // algorithm with the old to make sure we're not doing anything
      // silly.

      for (k = 0; k < K; ++k) {
	// Find an estimate for log_phi_nk.
	double doc_weight = 0.0;
	sslm_var* topic = var->topic[k];
	const double chain_variance = topic->chain_variance;

	// These matrices are already set up for the correct time.
	double g3 = mget(g3_matrix, w, k);
	double g4 = mget(g4_matrix, w, k);
	double g5 = mget(g5_matrix, w, k);
	double w_phi_sum = gsl_matrix_get(
	    var->topic[k]->w_phi_sum, w, time);

	// Only set these variables if we are within the correct
	// time window.
	if (time < var->nseq) {
	  doc_weight = gsl_vector_get(&document_weights.vector, k);
	}
	
	double term_weight;
	if (FLAGS_normalize_docs == "normalize") {
	  term_weight = ((double) p->doc->count[n]
			 / (double) p->doc->total);
	} else if (FLAGS_normalize_docs == "log") {
	  term_weight = log(p->doc->count[n] + 1.0);
	} else if (FLAGS_normalize_docs == "log_norm") {
	  term_weight = log(p->doc->count[n] / p->doc->total);
	} else if (FLAGS_normalize_docs == "identity") {
	  term_weight = p->doc->count[n];
	} else if (FLAGS_normalize_docs == "occurrence") {
	  term_weight = ((double) p->doc->count[n]
			 / (double) p->doc->total);
	} else {
	  assert(0);
	}

	assert(var != NULL);
	
	double total, term1, term2, term3, term4;
	double phi_last = 0.0;

	// It's unnecessary to always multiply by 1/chain_variance
	// this deep in a loop, but it's likely not a major
	// bottleneck.
	term1 = dig[k] + mget(p->model->topics, w, k);
	term2 = (g3
		 * term_weight
		 * doc_weight
		 / chain_variance);
	term3 = (term_weight
		 * doc_weight
		 * g4
		 / chain_variance);
	term4 = (term_weight * term_weight
		 * (phi_last * (doc_weight * doc_weight)
		    - (doc_weight * doc_weight
		       + FLAGS_sigma_l * FLAGS_sigma_l))
		 * g5
		 / chain_variance);

	// Note: we're ignoring term3.  sgerrish: 18 July 2010:
	// Changing term2 to have a positive coefficient (instead of
	// negative) to be consistent with parallel version.
	// sgerrish: 23 July 2010: changing term2 back to negative,
	// to try to reproduce earlier results.
	total = term1 - term2 - term3 + term4;
	mset(p->log_phi, n, k, total);
      }
      
      // Normalize in log space.
      gsl_vector log_phi_row = gsl_matrix_row(p->log_phi, n).vector;
      gsl_vector phi_row = gsl_matrix_row(p->phi, n).vector;
      log_normalize(&log_phi_row);
      
      for (i = 0; i < K; i++) {
	vset(&phi_row, i, exp(vget(&log_phi_row, i)));
      }
    }
}


/*
 * comput the likelihood bound
 */

double compute_lda_lhood(lda_post* p) {
  int k, n;
  int K = p->model->ntopics, N = p->doc->nterms;

  double gamma_sum = sum(p->gamma);
  double lhood =
    gsl_sf_lngamma(sum(p->model->alpha)) -
    gsl_sf_lngamma(gamma_sum);
  vset(p->lhood, K, lhood);
  
  double influence_term = 0.0;
  double digsum = gsl_sf_psi(gamma_sum);
  for (k = 0; k < K; k++) {
    if (p->doc_weight != NULL) {
      //	  outlog("doc weight size: %d", p->doc_weight->size);
      assert (K == p->doc_weight->size);
      double influence_topic = gsl_vector_get(p->doc_weight, k);
      if (FLAGS_model == "dim"
	  || FLAGS_model == "fixed") {
	influence_term = - ((influence_topic * influence_topic
			     + FLAGS_sigma_l * FLAGS_sigma_l)
			    / 2.0 / (FLAGS_sigma_d * FLAGS_sigma_d));
	// Note that these cancel with the entropy.
	//     - (log(2 * PI) + log(FLAGS_sigma_d)) / 2.0);
      }
    }
    double e_log_theta_k = gsl_sf_psi(vget(p->gamma, k)) - digsum;
    double lhood_term =
      (vget(p->model->alpha, k)-vget(p->gamma, k)) * e_log_theta_k +
      gsl_sf_lngamma(vget(p->gamma, k)) -
      gsl_sf_lngamma(vget(p->model->alpha, k));
    
    for (n = 0; n < N; n++) {
      if (mget(p->phi, n, k) > 0) {
	lhood_term +=
	  p->doc->count[n]*
	  mget(p->phi, n, k) *
	  (e_log_theta_k
	   + mget(p->model->topics, p->doc->word[n], k)
	   - mget(p->log_phi, n, k));
      }
    }
    vset(p->lhood, k, lhood_term);
    lhood += lhood_term;
    lhood += influence_term;
  }
  
  return(lhood);
}



/*
 * compute expected sufficient statistics for a corpus
 *
 */

double lda_e_step(lda* model,
		  corpus_t* data,
		  lda_suff_stats* ss) {
    int d, k, n;

    if (ss != NULL) reset_lda_suff_stats(ss);
    lda_post* p = new_lda_post(model->ntopics, data->max_unique);
    p->model = model;
    double lhood = 0;

    // for each document

    for (d = 0; d < data->ndocs; d++)
    {
        if (((d % 1000) == 0) && (d > 0))
        {
            outlog( "e-step: processing doc %d", d);
        }

        // fit posterior

        p->doc = data->doc[d];
        lhood += fit_lda_post(d, 0, p, NULL, NULL, NULL, NULL, NULL);

        // update expected sufficient statistics

        if (ss != NULL)
            for (k = 0; k < model->ntopics; k++)
                for (n = 0; n < p->doc->nterms; n++)
                    minc(ss->topics_ss,
                         p->doc->word[n], k,
                         mget(p->phi, n, k) * p->doc->count[n]);
    }

    // !!! FREE POSTERIOR

    return(lhood);
}


/*
 * compute MLE topics from sufficient statistics
 *
 */

double lda_m_step(lda* model, lda_suff_stats* ss) {
    int k, w;
    double lhood = 0;
    for (k = 0; k < model->ntopics; k++)
    {
        gsl_vector ss_k = gsl_matrix_column(ss->topics_ss, k).vector;
        gsl_vector log_p = gsl_matrix_column(model->topics, k).vector;
        if (LDA_USE_VAR_BAYES == 0)
        {
            gsl_blas_dcopy(&ss_k, &log_p);
            normalize(&log_p);
            vct_log(&log_p);
        }
        else
        {
            double digsum = sum(&ss_k)+model->nterms*LDA_TOPIC_DIR_PARAM;
            digsum = gsl_sf_psi(digsum);
            double param_sum = 0;
            for (w = 0; w < model->nterms; w++)
            {
                double param = vget(&ss_k, w) + LDA_TOPIC_DIR_PARAM;
                param_sum += param;
                double elogprob = gsl_sf_psi(param) - digsum;
                vset(&log_p, w, elogprob);
                lhood += (LDA_TOPIC_DIR_PARAM - param) * elogprob + gsl_sf_lngamma(param);
            }
            lhood -= gsl_sf_lngamma(param_sum);
        }
    }
    return(lhood);
}


/*
 * read sufficient statistics
 *
 */

void write_lda_suff_stats(lda_suff_stats* ss, char* name) {
    mtx_fprintf(name, ss->topics_ss);
}

lda_suff_stats* read_lda_suff_stats(char* filename, int ntopics, int nterms) {
    lda_suff_stats* ss = (lda_suff_stats*) malloc(sizeof(lda_suff_stats));
    ss->topics_ss = gsl_matrix_alloc(nterms, ntopics);
    mtx_fscanf(filename, ss->topics_ss);
    return(ss);
}

/*
 * new lda model and sufficient statistics
 *
 */

lda* new_lda_model(int ntopics, int nterms) {
    lda* m = (lda*) malloc(sizeof(lda));
    m->ntopics = ntopics;
    m->nterms = nterms;
    m->topics = gsl_matrix_calloc(nterms, ntopics);
    m->alpha  = gsl_vector_calloc(ntopics);

    return(m);
}

void free_lda_model(lda* m) {
    gsl_matrix_free(m->topics);
    gsl_vector_free(m->alpha);
    free(m);
}

lda_suff_stats* new_lda_suff_stats(lda* model) {
    lda_suff_stats* ss = (lda_suff_stats*) malloc(sizeof(lda_suff_stats));
    ss->topics_ss = gsl_matrix_calloc(model->nterms, model->ntopics);

    return(ss);
}

void reset_lda_suff_stats(lda_suff_stats* ss) {
    gsl_matrix_set_all(ss->topics_ss, 0.0);
}

lda_post* new_lda_post(int ntopics, int max_length) {
    lda_post* p = (lda_post*) malloc(sizeof(lda_post));
    p->phi = gsl_matrix_calloc(max_length, ntopics);
    p->log_phi = gsl_matrix_calloc(max_length, ntopics);
    p->gamma = gsl_vector_calloc(ntopics);
    p->lhood = gsl_vector_calloc(ntopics + 1);

    return(p);
}

void free_lda_post(lda_post* p) {
    gsl_matrix_free(p->phi);
    gsl_matrix_free(p->log_phi);
    gsl_vector_free(p->gamma);
    gsl_vector_free(p->lhood);
    free(p);
}

/*
 * initalize LDA SS from random
 *
 */

void initialize_lda_ss_from_random(corpus_t* data, lda_suff_stats* ss) {
    int k, n;
    gsl_rng * r = new_random_number_generator();
    for (k = 0; k < ss->topics_ss->size2; k++)
    {
        gsl_vector topic = gsl_matrix_column(ss->topics_ss, k).vector;
        gsl_vector_set_all(&topic, 0);
        for (n = 0; n < topic.size; n++)
        {
	  vset(&topic, n, gsl_rng_uniform(r) + 0.5 / data->ndocs + 4.0);
        }
    }
}


/*
 * initialize sufficient statistics from a document collection
 *
 */

void initialize_lda_ss_from_data(corpus_t* data, lda_suff_stats* ss) {
    int k, n, i, w;
    gsl_rng * r = new_random_number_generator();

    for (k = 0; k < ss->topics_ss->size2; k++)
    {
        gsl_vector topic = gsl_matrix_column(ss->topics_ss, k).vector;
        for (n = 0; n < LDA_SEED_INIT; n++)
        {
            int d = floor(gsl_rng_uniform(r) * data->ndocs);
            doc_t* doc = data->doc[d];
            for (i = 0; i < doc->nterms; i++)
            {
                vinc(&topic, doc->word[n], doc->count[n]);
            }
        }
        for (w = 0; w < topic.size; w++)
        {
            vinc(&topic, w, LDA_INIT_SMOOTH + gsl_rng_uniform(r));
        }
    }
}

/*
 * write LDA model
 *
 */

void write_lda(lda* model, char* name) {
    char filename[400];
    sprintf(filename, "%s.beta", name);
    mtx_fprintf(filename, model->topics);
    sprintf(filename, "%s.alpha", name);
    vct_fprintf(filename, model->alpha);
}

/*
 * read LDA
 *
 */

lda* read_lda(int ntopics, int nterms, char* name) {
    char filename[400];

    lda* model = new_lda_model(ntopics, nterms);
    sprintf(filename, "%s.beta", name);
    mtx_fscanf(filename, model->topics);
    sprintf(filename, "%s.alpha", name);
    vct_fscanf(filename, model->alpha);

    return(model);
}

void lda_em(lda* model,
	    lda_suff_stats* ss,
	    corpus_t* data,
            int max_iter,
	    char* outname) {
    int iter = 0;
    double lhood = lda_e_step(model, data, ss);
    double old_lhood = 0;
    double converged = 0;
    double m_lhood = lda_m_step(model, ss);
    outlog( "initial likelihood = %10.3f\n", lhood);
    do
    {
        iter++;
        old_lhood = lhood;
	double e_lhood = lda_e_step(model, data, ss);
	
        m_lhood = lda_m_step(model, ss);
        lhood = e_lhood + m_lhood;
        converged = (old_lhood - lhood) / (old_lhood);
        outlog("iter   = %d", iter);
        outlog("lhood  = % 10.3f", lhood);
        outlog("m, e lhood  = % 10.3f, % 10.3f", m_lhood, e_lhood);
        outlog("conv   = % 5.3e\n", converged);
	outlog("max_iter: %d\n", max_iter);
    }
    while (((converged > LDA_EM_CONVERGED) || (iter <= 5))
	   && (iter < max_iter));
    write_lda(model, outname);
}
