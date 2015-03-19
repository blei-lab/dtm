#ifndef DATA_H
#define DATA_H

#include "gsl-wrappers.h"
#include "param.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <hash_map>

#define OFFSET 0
//class string;

using namespace std;

namespace dtm {


// Create// Create the scaled beta distribution, which describes
// how much weight documents have after n years.
const int kScaledInfluenceMax = 200;
// This mean and variance are relative to the interval [0, 1.0].
const double kScaledInfluenceMean = 10.0 / kScaledInfluenceMax;
const double kScaledInfluenceVariance = ((10.0 / kScaledInfluenceMax)
					 * (10.0 / kScaledInfluenceMax));

/*
 * a document is a collection of counts and terms
 *
 */

typedef struct doc_t
{
    int total;
    int nterms;
    int* word;
    int* count;
    // A parameter for finding phi.
    double* lambda;

    // Used for measuring perplexity.
    double log_likelihood;
    double* log_likelihoods;
} doc_t;


/*
 * a corpus is a collection of documents
 *
 */

typedef struct corpus_t
{
    doc_t** doc;
    int ndocs;
    int nterms;
    int max_unique;  // maximum number of unique terms in a document
} corpus_t;


/*
 * a sequence is a sequence of corpora
 *
 */

typedef struct corpus_seq_t
{
    corpus_t** corpus;
    int nterms;
    int max_nterms;
    int len;
    int ndocs;
} corpus_seq_t;


typedef struct inf_var
{
  gsl_matrix** doc_weights;   // T matrices of document weights.
                              // each matrix is d_t x K.
  gsl_matrix** renormalized_doc_weights;   // T matrices of document weights.
                              // each matrix is d_t x K.
  int ntime;
} inf_var;

/*
 * variational posterior structure
 *
 */


typedef struct sslm_var {
    // properties

    int W; // vocabulary size
    int T; // sequence length

    // variational parameters

    gsl_matrix* obs;             // observations, W x T

    // biproducts of the variational parameters

    double obs_variance;         // observation variance
    double chain_variance;       // chain variance
    gsl_vector* zeta;            // extra variational parameter, T
    gsl_matrix* e_log_prob;      // E log prob(w | t), W x T

    // convenient quantities for inference

    gsl_matrix* fwd_mean;       // forward posterior mean, W x T
    gsl_matrix* fwd_variance;   // forward posterior variance, W x T
    gsl_matrix* mean;           // posterior mean, W x T
    gsl_matrix* variance;       // posterior variance, W x T

    gsl_matrix* mean_t;         // W x T
    gsl_matrix* variance_t;

    gsl_matrix* influence_sum_lgl;  // The sum exp * w_phi_l

    // Recent copy of w_phi_l.
    gsl_matrix* w_phi_l;         // W x T
    // gsl_matrix* w_phi_sum;      // W x T
    // gsl_matrix* w_phi_l_sq;      // Square term involving various
    gsl_matrix* m_update_coeff;  // Terms involving squares of
                                 // W, l, and phi.  W x T.
    gsl_matrix* m_update_coeff_g;  // \sum_i=0..t phi_l(t) r(i-t)
                                   // W x T.

    // useful temporary vector
    gsl_vector* T_vct;
} sslm_var;


typedef struct lda_seq
{
    int ntopics;             // number of topics
    int nterms;              // number of terms
    int nseq;                // length of sequence
    gsl_vector* alpha;       // dirichlet parameters

    sslm_var** topic;        // topic chains.

    inf_var* influence;      // document weights

    gsl_matrix** influence_sum_lgl;  // Sum of document weights at time t (see g in the regression formula)

  //    gsl_vector** influence_sum_g;  // Sum of document weights at time t.
  // gsl_vector** influence_sum_h;  // Sum of document weights at time t.

    inf_var* renormalized_influence;      // document weights

  //    gsl_matrix** w_phi_l;        // Product term for the \beta update.
  //  gsl_matrix** w_phi_l_sq;     // Square term involving various
                                // coefficients for the \beta update.

  pair<int, float>**** top_doc_phis;        // T x D_t x n of document phis.
} lda_seq;

/*
 * functions
 *
 */

corpus_t* read_corpus(const char* name);
void free_corpus(corpus_t* corpus);
corpus_seq_t* read_corpus_seq(const char* name);
int compute_max_nterms(const corpus_seq_t* c);
gsl_matrix * compute_total_counts(const corpus_seq_t* c);
corpus_seq_t* make_seq_corpus_subset(corpus_seq_t* all, int start, int end);
void write_corpus(corpus_t* c, char* filename);
void write_corpus_seq(corpus_seq_t* c, char* name);
corpus_seq_t* make_corpus_seq_subset(corpus_seq_t* all, int start, int end);
corpus_t* collapse_corpus_seq(corpus_seq_t* c);
double* NewScaledInfluence(int size);

/*
 * Reading and writing.
 *
 */

// read and write an lda sequence
void write_lda_seq_docs(const string& model_type,
			const string& root_directory,
			int min_time,
			int max_time,
			const lda_seq* m);



}  // namespace dtm

#endif
