#include "c2_lib.h"
#include "lda-seq.h"

#include "ss-lm.h"

#include "gflags.h"
#include "strutil.h"

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector_double.h>
#include <string.h>
#include <vector>
#include <hash_set>
#include <hash_map>

DEFINE_double(phi_squared_factor,
	      1.0,
	      "Scale factor for squared phi terms.");

DEFINE_int32(lda_sequence_max_em_iter,
	     30,
	     "The maximum number of iterations.");
DEFINE_int32(lda_sequence_min_iter,
	     1,
	     "The maximum number of iterations.");

DEFINE_string(seq_topic_filename,
	      "",
	      "The sequential topics filename.");

DECLARE_string(normalize_docs);

DEFINE_string(time_boundaries,
	      "",
	      "Comma-separated list of integers describing time ranges.");
DECLARE_int32(lda_max_em_iter);
DECLARE_string(doc_boundaries);
DECLARE_string(topic_boundaries);
DECLARE_int64(rng_seed);

DEFINE_int32(save_time,
	     -1,
	     "Save a specific time.  If -1, save all times.");

DECLARE_double(alpha);
DECLARE_int32(ntopics);

DEFINE_int32(max_time, 0, "");
DEFINE_int32(min_time, 0, "");

DEFINE_double(top_obs_var, -1.0, "");
DEFINE_double(top_chain_var, -1.0, "");

// DEFINE_int32(start, -1, "");
// DEFINE_int32(end, -1, "");

DEFINE_string(model,
	      "fixed",
              "The function to perform. "
	      "Can be dtm or fixed.");
DEFINE_string(resume_stage,
	      "lda",
              "Resume from this stage.  Options are lda, doc, topic.");
DECLARE_string(corpus_prefix);
DECLARE_int32(number_tasks);

DECLARE_int32(influence_flat_years);

DECLARE_int32(max_doc);
DECLARE_int32(min_doc);
DECLARE_int32(min_topic);
DECLARE_int32(max_topic);

DECLARE_int32(fix_topics);
DECLARE_string(root_directory);
DEFINE_string(outname, "", "");
DECLARE_int32(max_number_time_points);
DECLARE_double(sigma_d);
DECLARE_double(sigma_l);
DECLARE_double(time_resolution);

namespace dtm {

extern int LDA_INFERENCE_MAX_ITER;
static double* scaled_influence = NULL;
  // using namespace std;
// const int TIME = 61;
const int TIME = -3;
const double PI = 3.141592654;


/*
 * populate an LDA model at a particular time point
 *
 */

static void write_topic_suffstats(const int min_time,
				  const int max_time,
				  gsl_matrix** ss,
				  int nterms,
				  int k) {
  string filename = StringPrintf(
    "%s/lda-seq-ss-topic_%d-time_%d_%d.dat",
    FLAGS_root_directory.c_str(),
    k, min_time, max_time);

  gsl_matrix_view topics_time_ss = gsl_matrix_submatrix(
      ss[k],
      0, min_time,
      nterms,
      max_time - min_time);

  mtx_fprintf(filename.c_str(), &topics_time_ss.matrix);
}

static gsl_matrix* read_topic_suffstats(
  const vector<int>& time_boundaries,
  int nterms,
  int k) {
  // Note that these sufficient statistics are by time, and stored in
  // disjoint subsets of the ss matrix.  Therefore we don't need to
  // aggregate them.
  const int kMaxTime = time_boundaries[time_boundaries.size() - 1];
  gsl_matrix* topics_ss = gsl_matrix_alloc(
      nterms,
      kMaxTime);
  for (int i=0; i + 1 < time_boundaries.size(); ++i) {
    const int kMinTime = time_boundaries[i];
    const int kMaxTime = time_boundaries[i + 1];
    string filename = StringPrintf("%s/lda-seq-ss-topic_%d-time_%d_%d.dat",
				   FLAGS_root_directory.c_str(),
				   k, kMinTime, kMaxTime);
    gsl_matrix_view topics_time_ss = gsl_matrix_submatrix(
      topics_ss,
      0, kMinTime,
      nterms,
      kMaxTime - kMinTime);
    mtx_fscanf(filename.c_str(), &topics_time_ss.matrix);
  }
  return topics_ss;
}

static void RewriteTopicsAsSequential(
    const vector<int>& time_boundaries,
    const vector<int>& topic_boundaries,
    lda_seq* model) {
  // model is of the form W x Time

  // Model is already initialized by new_lda_seq from before.
  // model->topic = new sslm_var*[model->ntopics];
  string kSSFilename = StringPrintf("%s/initial-lda-ss.dat",
				    FLAGS_root_directory.c_str());
  lda_suff_stats* init_suffstats = read_lda_suff_stats(
    kSSFilename.c_str(),
    model->ntopics,
    model->nterms);

  lda* flat_model = new_lda_model(model->ntopics,
				  model->nterms);
					
  read_lda_topics(topic_boundaries, flat_model);

  const int W = model->nterms;
  const int T = model->nseq;
  const int K = model->ntopics;
  for (int k=0; k < model->ntopics; ++k) {
    // Is this necessary here?
    model->topic[k]->w_phi_l = gsl_matrix_calloc(W, T);
    model->topic[k]->m_update_coeff_g = NULL;

    model->topic[k]->obs_variance = FLAGS_top_obs_var;
    model->topic[k]->chain_variance = FLAGS_top_chain_var;

    gsl_vector_view from = gsl_matrix_column(
      flat_model->topics, k);
    for (int t=0; t < model->nseq; ++t) {
      gsl_vector_view to =
        gsl_matrix_column(model->topic[k]->e_log_prob, t);
      gsl_vector_memcpy(&to.vector, &from.vector);
    }

    gsl_vector slice = gsl_matrix_column(
      init_suffstats->topics_ss, k).vector;

    gsl_vector* log_norm_counts = gsl_vector_alloc(slice.size);

    // Populate only a subset of fields.
    gsl_vector_memcpy(log_norm_counts, &slice);
    gsl_vector_add_constant(log_norm_counts, 1.0 / model->nterms);
    normalize(log_norm_counts);
    vct_log(log_norm_counts);

    // set variational observations to transformed counts

    for (int t = 0; t < model->nseq; t++) {
      msetcol(model->topic[k]->obs, t, log_norm_counts);
    }
    gsl_vector_free(log_norm_counts);

    const string kTopicsPrefix =
      StringPrintf("%s/sslm-topic_%d",
		   FLAGS_root_directory.c_str(),
		   k);

    write_sslm_var(model->topic[k], kTopicsPrefix.c_str());
  }

  for (int i=0; i + 1 < time_boundaries.size(); ++i) {
    const int kMinTime = time_boundaries[i];
    const int kMaxTime = time_boundaries[i + 1];

    // Re-allocate m_update_coeff to have the correct size.
    for (int k=0; k < K; ++k) {
      if (i > 0) {
	gsl_matrix_free(model->topic[k]->m_update_coeff);
      }
      model->topic[k]->m_update_coeff = gsl_matrix_calloc(W, kMaxTime - kMinTime);
    }

    write_lda_seq_docs(FLAGS_model,
		       FLAGS_root_directory,
		       kMinTime,
		       kMaxTime,
		       model);
  }

  for (int k=0; k < K; ++k) {
    gsl_matrix_free(model->topic[k]->m_update_coeff);
    gsl_matrix_free(model->topic[k]->w_phi_l);
    model->topic[k]->m_update_coeff = NULL;
    model->topic[k]->w_phi_l = NULL;
  }

  gsl_matrix_free(init_suffstats->topics_ss);
  free_lda_model(flat_model);
  
}

inf_var* inf_var_alloc(int number_topics,
		       int* ndocs,
		       int len) {

  // Hate to do this, but I had trouble using it before.  This should
  // be the first place we use it; otherwise we'll get a sigsev nil.
  if (scaled_influence == NULL) {
    scaled_influence =
      NewScaledInfluence(FLAGS_max_number_time_points);

    printf("New scaled influence: \n");
    for (int i=0; i < FLAGS_max_number_time_points; ++i) {
      printf(" %lf", scaled_influence[i]);
    }
    printf("\n");
  }

  inf_var* inf_var_ptr = (inf_var*) malloc(sizeof(inf_var));
  inf_var_ptr->doc_weights = (gsl_matrix**) malloc(sizeof(gsl_matrix*)
						   * len);
  inf_var_ptr->renormalized_doc_weights = (gsl_matrix**) malloc(
    sizeof(gsl_matrix*)
    * len);
  inf_var_ptr->ntime = len;
  int i=0;
  for (i=0; i < len; ++i) {
    outlog("creating matrix. %d %d", ndocs[i], number_topics);
    if (ndocs[i] == 0) {
      inf_var_ptr->doc_weights[i] = (gsl_matrix*) malloc(sizeof(gsl_matrix));
      inf_var_ptr->doc_weights[i]->size1 = 0;
      inf_var_ptr->doc_weights[i]->size2 = number_topics;
      inf_var_ptr->renormalized_doc_weights[i] = (gsl_matrix*) malloc(sizeof(gsl_matrix));
      inf_var_ptr->renormalized_doc_weights[i]->size1 = 0;
      inf_var_ptr->renormalized_doc_weights[i]->size2 = number_topics;
    } else {
      inf_var_ptr->doc_weights[i] = gsl_matrix_calloc(ndocs[i],
						      number_topics);
      inf_var_ptr->renormalized_doc_weights[i] = gsl_matrix_calloc(ndocs[i],
								   number_topics);
    }
  }
  return inf_var_ptr;
}

void inf_var_free(inf_var* ptr) {
  // TODO.
}

// Solves the linear system Ax = b for x.
// Assumes that x is already allocated.
void Solve(gsl_matrix* A,
	   const gsl_vector* b,
	   gsl_vector* x) {
    int permutation_sign;
    gsl_permutation* permutation = gsl_permutation_alloc(b->size);
    gsl_linalg_LU_decomp(A, permutation, &permutation_sign);
    gsl_linalg_LU_solve(A, permutation, b, x);
    gsl_permutation_free(permutation);
}

// Find the sums of influence of all documents in advance.
// g has scores for everything *up to but not including* g.
void InfluenceTotalFixed(lda_seq* seq,
			 const corpus_seq_t* data) {
  gsl_vector* exp_tmp = gsl_vector_alloc(seq->nterms);

  for (int k=0; k < seq->ntopics; ++k) {

    for (int s=0; s < seq->nseq; ++s) {
      // Pull out elements of g, and make sure to set them to 0!
      gsl_vector_view g = gsl_matrix_column(
	  seq->influence_sum_lgl[k], s);
      gsl_vector_set_zero(&g.vector);

      for (int t=0; t <= s; ++t) {
	gsl_vector_view w_phi_l =
	  gsl_matrix_column(seq->topic[k]->w_phi_l, t);
	gsl_vector_memcpy(exp_tmp, &w_phi_l.vector);
	gsl_vector_scale(exp_tmp, scaled_influence[s - t]);
	gsl_vector_add(&g.vector, exp_tmp);
      }
    }
  }
  gsl_vector_free(exp_tmp);
}

void DumpTimeDocTopicStats(const char* root,
			   size_t t,
			   corpus_t* corpus,
			   gsl_matrix** phi) {
  // For all documents, dump the top topics.

  string filename = StringPrintf("%s/%d_doc_term_topics.dat",
				 FLAGS_root_directory.c_str(), t);
  //  string filename = FLAGS_root_directory
  // Dump the top topics for each word.
  FILE* f = fopen(filename.c_str(), "w");
  for (unsigned int d=0; d < corpus->ndocs; ++d) {
    gsl_matrix* phi_d = phi[d];
    doc_t* doc = corpus->doc[d];
    for (unsigned int n=0; n < doc->nterms; ++n) {
      
      unsigned int w = doc->word[n];

      // First, find the max topic weight.
      unsigned int max_topic_index = 0;
      double max_topic_weight = gsl_matrix_get(phi_d, n, 0);
      for (unsigned int k=0; k < phi_d->size2; ++k) {
	double phi_d_n_k = gsl_matrix_get(phi_d, n, k);
	if (phi_d_n_k > max_topic_weight) {
	  max_topic_weight = phi_d_n_k;
	  max_topic_index = k;
	}
      }

      fprintf(f, "%d:%d:%.3f", w, max_topic_index, max_topic_weight);
      
      if (n < doc->nterms - 1) {
	fprintf(f, " ");
      }
    }
    fprintf(f, "\n");
  }
  fclose(f);
}


void PrepareRegressionComponents(
    corpus_t* corpus,
    lda_seq* seq,
    unsigned int k,
    gsl_matrix* W_phi,
    gsl_matrix* W_phi_var,
    gsl_vector* d_phi_var_tmp,
    gsl_vector* response,
    gsl_vector* d_tmp,
    gsl_matrix* dd_tmp,
    gsl_vector* document_weights,
    gsl_matrix* W_phi_tmp,
    gsl_vector* exp_h_tmp) {

  // Then left-multiply this by W_phi:
  gsl_blas_dgemv(CblasTrans,
		 1.0,
		 W_phi,
		 response,
		 0.0,
		 d_tmp);

  // Set up the transformation matrix.
  // First, set up W_phi^T \Lambda W_phi.
  gsl_matrix_memcpy(W_phi_tmp, W_phi);
  for (unsigned int d=0; d < corpus->ndocs; ++d) {
    gsl_vector_view col = gsl_matrix_column(W_phi_tmp, d);
    gsl_vector_mul(&col.vector, exp_h_tmp);
  }

  // Yuck.  Maybe we should do this sparsely?
  // Probably won't be too bad, at least for now.
  gsl_blas_dgemm(CblasTrans,
		 CblasNoTrans,
		 1.0,
		 W_phi_tmp,
		 W_phi,
		 0.0,
		 dd_tmp);

  gsl_blas_dgemv(CblasTrans,
		 1.0,
		 W_phi_var,
		 exp_h_tmp,
		 0.0,
		 d_phi_var_tmp);

  // Next, add elements to the diagonal of dd_tmp.
  for (unsigned int d=0; d < corpus->ndocs; ++d) {
    double value = gsl_matrix_get(dd_tmp, d, d);
    value += (seq->topic[k]->chain_variance
	      / (FLAGS_sigma_d * FLAGS_sigma_d));
    // sgerrish: Is this supposed to be multiplied by anything?
    value += gsl_vector_get(d_phi_var_tmp, d);
    gsl_matrix_set(dd_tmp, d, d, value);
  }  
}

void SetExpHTmp(lda_seq* seq,
		const corpus_seq_t* data,
		unsigned int t,
		unsigned int k,
		unsigned int n,
		gsl_vector* exp_h_tmp,
		gsl_vector* zw_tmp,
		gsl_vector** exp_i_tmp) {
  gsl_vector_set_zero(exp_h_tmp);
  for (int i = t; i < seq->nseq; ++i) {
    gsl_vector_view mean_i_current =
      gsl_matrix_column(seq->topic[k]->e_log_prob, i);
    gsl_vector_view var_i_current =
      gsl_matrix_column(seq->topic[k]->variance, i + 1);

    // Set up exp_h_tmp.
    gsl_vector_memcpy(zw_tmp, &var_i_current.vector);
    gsl_vector_sub(zw_tmp, &mean_i_current.vector);
    gsl_vector_scale(zw_tmp, 2.0);
    for (n=0; n < data->nterms; ++n) {
      gsl_vector_set(zw_tmp, n,
		     exp(gsl_vector_get(zw_tmp, n)));
    }
    gsl_vector_scale(zw_tmp,
		     scaled_influence[i - t]
		     * scaled_influence[i - t]);
    gsl_vector_add(exp_h_tmp, zw_tmp);
    
    // Set up exp_i_tmp.
    gsl_vector_memcpy(exp_i_tmp[i], &var_i_current.vector);
    gsl_vector_scale(exp_i_tmp[i], 0.5);
    gsl_vector_sub(exp_i_tmp[i], &mean_i_current.vector);
    for (n=0; n < data->nterms; ++n) {
      gsl_vector_set(exp_i_tmp[i], n,
		     exp(gsl_vector_get(exp_i_tmp[i], n)));
    }
  }
}

double update_inf_var_fixed(lda_seq* seq,
			    const corpus_seq_t* data,
			    gsl_matrix** phi,
			    size_t t,
			    const char* root) {
  double lhood = 0.0;
  // Note that we're missing a suspicious factor of -2 on the document
  // weights currently.  We won't worry about that for now (since
  // we're still experimenting), but it should soon be fixed.
  corpus_t* corpus = data->corpus[t];
  if (!corpus->ndocs) {
    return lhood;
  }
  inf_var* influence = seq->influence;
  if (t != TIME && 0) {
    return lhood;
  }

  // We need access to the following:
  gsl_matrix* documents_topics_t = influence->doc_weights[t];
  gsl_matrix* renormalized_documents_topics_t = influence->renormalized_doc_weights[t];

  size_t k;

  gsl_matrix* W_phi = gsl_matrix_calloc(
    seq->nterms, corpus->ndocs);
  gsl_matrix* W2_phi2 = gsl_matrix_calloc(
    seq->nterms, corpus->ndocs);
  gsl_matrix* W_phi_var =
    gsl_matrix_calloc(seq->nterms, corpus->ndocs);
  gsl_matrix* W_phi_tmp = gsl_matrix_alloc(
    seq->nterms, corpus->ndocs);
  gsl_matrix* dd_tmp = gsl_matrix_alloc(
    corpus->ndocs, corpus->ndocs);
  gsl_vector* xd_tmp = gsl_vector_calloc(
    corpus->ndocs);
  gsl_vector* yw_tmp = gsl_vector_calloc(
    seq->nterms);
  gsl_vector* zw_tmp = gsl_vector_calloc(
    seq->nterms);
  gsl_vector* terms_inc_tmp = gsl_vector_calloc(
    seq->nterms);
  gsl_vector** exp_i_tmp = new gsl_vector*[seq->nseq];
  for (int i=0; i < seq->nseq; ++i) {
    exp_i_tmp[i] = gsl_vector_calloc(seq->nterms);
  }

  int n;

  gsl_vector* response = gsl_vector_calloc(seq->nterms);
  gsl_vector* exp_h_tmp = gsl_vector_calloc(seq->nterms);
  gsl_vector* d_tmp = gsl_vector_calloc(corpus->ndocs);
  gsl_vector* d_phi_var_tmp = gsl_vector_calloc(corpus->ndocs);

  //  assert(post->phi->size2 == documents_topics_t->size2);
  double* total_terms = new double[corpus->ndocs];
  double* renormalization_totals = new double[corpus->ndocs];
  for (k=0; k < documents_topics_t->size2; ++k) {
    // Set up W_phi and W_phi_var.
    for (int d=0; d < corpus->ndocs; ++d) {
      doc_t* doc = corpus->doc[d];
      total_terms[d] = 0.0;
      renormalization_totals[d] = 0.0;
      double log_norm_sum = 0.0;
      for (n=0; n < doc->nterms; ++n) {
	total_terms[d] += doc->count[n];
	if (FLAGS_normalize_docs == "normalize") {
	  renormalization_totals[d] = 1.0;
	} else if (FLAGS_normalize_docs == "log") {
	  renormalization_totals[d] += log(doc->count[n] + 1);
	} else if (FLAGS_normalize_docs == "log_norm") {
	  double weight = log(doc->count[n] / doc->total);
	  renormalization_totals[d] += weight;
	} else if (FLAGS_normalize_docs == "identity") {
	  renormalization_totals[d] += doc->count[n];
	} else if (FLAGS_normalize_docs == "occurrence") {
	  renormalization_totals[d] = 1.0;
	}
      }
      assert(doc->total == total_terms[d]);

      for (n=0; n < doc->nterms; ++n) {
	// Phi, for the doc's term n and topic k.
	double phi_d_n_k = gsl_matrix_get(phi[d], n, k);

	// This cast should happen automatically because total_terms is a double,
	// but we make the cast here to be explicit and to avoid bugs later.
	double number_terms;
	if (FLAGS_normalize_docs == "normalize") {
	  number_terms = ((double) doc->count[n]
			  / (double) total_terms[d]);
	} else if (FLAGS_normalize_docs == "log") {
	  number_terms = log(doc->count[n] + 1.0);
	} else if (FLAGS_normalize_docs == "log_norm") {
	  number_terms = log(doc->count[n] / total_terms[d]);
	  renormalization_totals[d] += log(doc->count[n] / total_terms[d]);
	} else if (FLAGS_normalize_docs == "identity") {
	  number_terms = doc->count[n];
	} else if (FLAGS_normalize_docs == "occurrence") {
	  number_terms = ((double) doc->count[n] / (double) total_terms[d]);
	  assert(doc->count[n] == 1);
	} else {
	  assert(0);
	}

	gsl_matrix_set(W_phi, doc->word[n], d,
		       number_terms * phi_d_n_k);
	gsl_matrix_set(W2_phi2, doc->word[n], d,
		       number_terms * number_terms
		       * phi_d_n_k * phi_d_n_k);
	gsl_matrix_set(W_phi_var, doc->word[n], d,
		       number_terms * number_terms
		       * (phi_d_n_k - phi_d_n_k * phi_d_n_k));
      }
    }

    gsl_vector_view document_weights = gsl_matrix_column(
      documents_topics_t,
      k);
    gsl_vector_view renormalized_document_weights = gsl_matrix_column(
      renormalized_documents_topics_t,
      k);
    assert(seq->topic[k]->e_log_prob->size2 == data->len);

    // Now, with w_phi_var, etc. set, determine
    // \sum_{i=t}^{T-1} r(...) h(t, i)
    SetExpHTmp(seq,
	       data,
	       t,
	       k,
	       n,
	       exp_h_tmp,
	       zw_tmp,
	       exp_i_tmp);

    // Next, set up the weighted response,
    // \exp_{-m + v / 2) \circ (m_{t+1} - m_t + v_t}).
    // Here we also subtract the current l's contribution to influence_sum_lgl.
    gsl_vector_view w_phi_l_t = gsl_matrix_column(seq->topic[k]->w_phi_l, t);
    gsl_vector_set_zero(response);
    for (int i = t; i < seq->nseq - 1; ++i) {
      gsl_vector_view total_influence_time_i =
	gsl_matrix_column(seq->influence_sum_lgl[k], i);
      gsl_vector_memcpy(zw_tmp, &w_phi_l_t.vector);
      gsl_vector_scale(zw_tmp, scaled_influence[i - t]);
      gsl_vector_sub(&total_influence_time_i.vector, zw_tmp);

      // Now, copy this total influence at time i back into zw_tmp.
      gsl_vector_memcpy(zw_tmp, &total_influence_time_i.vector);
      gsl_vector_mul(zw_tmp, exp_i_tmp[i]);

      gsl_vector_view mean_i_current =
	gsl_matrix_column(seq->topic[k]->e_log_prob, i);
      gsl_vector_view mean_i_next =
	gsl_matrix_column(seq->topic[k]->e_log_prob, i + 1);
      gsl_vector_view var_i_current =
	gsl_matrix_column(seq->topic[k]->variance, i + 1);
      gsl_vector_memcpy(terms_inc_tmp, &mean_i_next.vector);
      gsl_vector_sub(terms_inc_tmp, &mean_i_current.vector);
      gsl_vector_add(terms_inc_tmp, &var_i_current.vector);
      gsl_vector_sub(terms_inc_tmp, zw_tmp);
      assert(data->nterms == terms_inc_tmp->size);
      gsl_vector_mul(terms_inc_tmp, exp_i_tmp[i]);
      gsl_vector_scale(terms_inc_tmp, scaled_influence[i - t]);

      gsl_vector_add(response, terms_inc_tmp);
    }

    PrepareRegressionComponents(corpus,
				seq,
				k,
				W_phi,
				W_phi_var,
				d_phi_var_tmp,
				response,
				d_tmp,
				dd_tmp,
				&document_weights.vector,
				W_phi_tmp,
				exp_h_tmp);
    // Finally, solve for the document weights d!
    Solve(dd_tmp,
	  d_tmp,
	  &document_weights.vector);
    
    // Keep track of the iteration so we can dump certain stats
    // occasionally (but not always).
    static int dump_count = 0;
    ++dump_count;

    if (FLAGS_save_time == -1) {
      // First, dump phi's for the top topics.
      DumpTimeDocTopicStats(root, t, corpus, phi);
    }
    outlog("Done updating topic %ld, time %ld.", k, t);

    for (int d = 0; d < document_weights.vector.size; ++d) {
      gsl_vector_set(&renormalized_document_weights.vector, d,
		     vget(&document_weights.vector, d)
		     * renormalization_totals[d]);
    }

    //     outlog("influence, topic %d", );

    // Now copy this and several products to 
    // the sslm_var object.

    gsl_vector_view w_phi_l =
      gsl_matrix_column(seq->topic[k]->w_phi_l, t);
    gsl_blas_dgemv(CblasNoTrans,
		   1.0,
		   W_phi,
		   &document_weights.vector,
		   0.0,
		   &w_phi_l.vector);

    // Copy this value back into lgl.
    for (int i=t; i < seq->nseq - 1; ++i) {
      gsl_vector_view total_influence_time_i = 
	gsl_matrix_column(seq->influence_sum_lgl[k], i);
      gsl_vector_memcpy(zw_tmp, &w_phi_l.vector);
      gsl_vector_scale(zw_tmp, scaled_influence[i - t]);
      gsl_vector_add(&total_influence_time_i.vector, zw_tmp);
    }

    // Keep track of the term we need to add to m_update_coeff.
    gsl_vector_memcpy(terms_inc_tmp, &w_phi_l.vector);
    gsl_vector_mul(terms_inc_tmp, &w_phi_l.vector);

    // Copy and square the document weights vector.
    for (int i = 0; i < xd_tmp->size; ++i) {
      double value = gsl_vector_get(&document_weights.vector, i);
      value = value * value + FLAGS_sigma_l * FLAGS_sigma_l;
      gsl_vector_set(xd_tmp, i, value);
    }
    gsl_blas_dgemv(CblasNoTrans,
		   1.0,
		   W_phi_var,
		   xd_tmp,
		   0.0,
		   yw_tmp);
    gsl_vector_add(terms_inc_tmp, yw_tmp);
    
    for (int i = 0; i < xd_tmp->size; ++i) {
      gsl_vector_set(xd_tmp, i, FLAGS_sigma_l * FLAGS_sigma_l);
    }
    gsl_blas_dgemv(CblasNoTrans,
		   1.0,
		   W2_phi2,
		   xd_tmp,
		   0.0,
		   yw_tmp);
    gsl_vector_add(terms_inc_tmp, yw_tmp);
    
    // Copy terms_inc_tmp to m_update_coeff_h.
    // This later is merged with m_udpate_coeff from all other
    // times.
    gsl_vector_view m_update_coeff_h =
      gsl_matrix_column(seq->topic[k]->m_update_coeff,
			t - FLAGS_min_time);
    gsl_vector_memcpy(&m_update_coeff_h.vector, terms_inc_tmp);

    /*
    // Store an update coefficient for the beta updates.
    for (int i = t; i < seq->nseq; ++i) {
      gsl_vector_view m_update_coeff_h =
	gsl_matrix_column(seq->topic[k]->m_update_coeff, i);
      if (t == 0) {
	gsl_vector_set_zero(&m_update_coeff_h.vector);
      }
      gsl_vector_memcpy(yw_tmp, terms_inc_tmp);
      gsl_vector_scale(yw_tmp, scaled_influence[i - t]);
      gsl_vector_add(&m_update_coeff_h.vector, yw_tmp);
    }

    for (int i = t; i < seq->nseq; ++i) {
      gsl_vector_view m_update_coeff_g = 
	gsl_matrix_column(seq->topic[k]->m_update_coeff_g, i);
      if (t == 0) {
	gsl_vector_set_zero(&m_update_coeff_g.vector);
      }
      gsl_vector_memcpy(yw_tmp, &w_phi_l.vector);
      gsl_vector_scale(yw_tmp, scaled_influence[i - t]);
      gsl_vector_add(&m_update_coeff_g.vector, yw_tmp);
    }
    */

    for (int i = 0; i < corpus->ndocs; ++i) {
      double value = gsl_vector_get(&document_weights.vector, i);
      
      // While we're here, increment the likelihood.
      lhood += (-(value * value + FLAGS_sigma_l * FLAGS_sigma_l)
		/ (2.0 * FLAGS_sigma_d * FLAGS_sigma_d)
		- 0.5 * log(2 * PI)
		- log(FLAGS_sigma_d * FLAGS_sigma_d));
    }
  }
  delete[] total_terms;
  delete[] renormalization_totals;
  gsl_matrix_free(W_phi);
  gsl_matrix_free(W_phi_tmp);
  gsl_matrix_free(W2_phi2);
  gsl_matrix_free(W_phi_var);
  gsl_matrix_free(dd_tmp);
  gsl_vector_free(exp_h_tmp);
  gsl_vector_free(response);
  gsl_vector_free(terms_inc_tmp);
  for (int i=0; i < seq->nseq; ++i) {
    gsl_vector_free(exp_i_tmp[i]);
  }
  delete[] exp_i_tmp;
  gsl_vector_free(d_tmp);
  gsl_vector_free(d_phi_var_tmp);
  gsl_vector_free(xd_tmp);
  gsl_vector_free(yw_tmp);
  gsl_vector_free(zw_tmp);
  
  return lhood;
}


void make_lda_from_seq_slice(lda* lda_m,
			     lda_seq* lda_seq_m,
			     int time) {
  // set lda model topics
  // !!! note: we should be able to point to the view...

  int k;
  for (k = 0; k < lda_seq_m->ntopics; k++) {
      // get topic
      gsl_vector s =
	gsl_matrix_column(lda_seq_m->topic[k]->e_log_prob,
			  time).vector;
      gsl_vector d =
	gsl_matrix_column(lda_m->topics, k).vector;
      gsl_blas_dcopy(&s, &d);
    }
  gsl_blas_dcopy(lda_seq_m->alpha, lda_m->alpha);
}

static gsl_matrix* g_alloc(lda_seq* model,
			   const corpus_seq_t* data,
			   int time) {
     gsl_matrix* g = gsl_matrix_calloc(model->nterms,
				       model->ntopics);
     double exp_m, m, m_next;
     for (int k = 0; k < model->ntopics; ++k) {
       for (int w=0; w < model->nterms; ++w) {
	 double variance_first = mget(model->topic[k]->variance, w, time);
	 double m = mget(model->topic[k]->e_log_prob, w, time);
	 double m_next;
	 exp_m = exp(-m + variance_first / 2.0);
	 gsl_matrix_set(g, w, k,
			(scaled_influence[0]
			 * -variance_first
			 * exp_m));

	 for (int i=time; i < model->nseq - 1; ++i) {
	   // This loop is kind of going overboard, but at least we
	   // do this once per E-M iteration.
	   double influence_other_times = 0.0;
	   for (int j = 0; j < i; ++j) {
	     exp_m = exp(-mget(model->topic[k]->e_log_prob, w, j)
			 + mget(model->topic[k]->variance, w, j) / 2.0);
	     // Note that we skip the other docs in this time period.
	     // Those get special treatment below. 
	     if (j != time) {
	       influence_other_times += (
	          mget(model->topic[k]->w_phi_l, w, j)
		  * scaled_influence[i - j]
		  * exp_m);
	      }
	   }

	   m = mget(model->topic[k]->e_log_prob, w, i);
	   m_next = mget(model->topic[k]->e_log_prob, w, i + 1);
	   // Increment the current count by this value.
	   gsl_matrix_set(g, w, k,
			  mget(g, w, k) +
			  (scaled_influence[i - time]
			   * (m_next - m - influence_other_times)));
	 }
	 exp_m = exp(-m
		     + mget(model->topic[k]->variance, w, time) / 2.0);
	 int i=5;
	 gsl_matrix_set(g, w, k, mget(g, w, k) * exp_m);
       }
     }
     return g;
}

static gsl_matrix* g3_alloc(lda_seq* model,
			    const corpus_seq_t* data,
			    int time) {
  gsl_matrix* g = gsl_matrix_calloc(model->nterms,
				    model->ntopics);
  double exp_m, m, m_next, variance, total;
  for (int k = 0; k < model->ntopics; ++k) {
    for (int w=0; w < model->nterms; ++w) {
      total = 0.0;
      for (int i=time; i < model->nseq - 1; ++i) {
	// This loop is kind of going overboard, but at least we
	// do this once per E-M iteration.
	variance = mget(model->topic[k]->variance, w, i + 1);
	m = mget(model->topic[k]->e_log_prob, w, i);
	m_next = mget(model->topic[k]->e_log_prob, w, i + 1);
	exp_m = exp(-m + variance / 2.0);
	total += (scaled_influence[i - time]
		  * exp_m
		  * (m_next - m + variance));
      }
      gsl_matrix_set(g, w, k, total);
    }
  }
  return g;
}

static gsl_matrix* g4_alloc(lda_seq* model,
			    const corpus_seq_t* data,
			    int time) {
  gsl_matrix* g = gsl_matrix_calloc(model->nterms,
				    model->ntopics);
  double exp_m, exp_m_scaled, m, total, variance, w_phi_l;
  for (int k=0; k < model->ntopics; ++k) {
    for (int w=0; w < model->nterms; ++w) {
      total = 0.0;
      for (int i=time; i < model->nseq - 1; ++i) {
	// This loop is kind of going overboard, but at least we
	// do this once per E-M iteration.
	variance = mget(model->topic[k]->variance, w, i + 1);
	m = mget(model->topic[k]->e_log_prob, w, i);
	exp_m = exp(-2.0 * m + 2.0 * variance);
	exp_m_scaled = exp_m * scaled_influence[i - time];
	for (int j=0; j <= i; ++j) {
	  w_phi_l = mget(model->topic[k]->w_phi_l, w, j);
	  total += exp_m_scaled * w_phi_l * scaled_influence[i - j];
	}
      }
      gsl_matrix_set(g, w, k, total);
    }
  }
  return g;
}

static gsl_matrix* g5_alloc(lda_seq* model,
			    const corpus_seq_t* data,
			    int time) {
  gsl_matrix* g = gsl_matrix_calloc(model->nterms,
				    model->ntopics);
  double exp_m, m, total, variance;
  for (int k=0; k < model->ntopics; ++k) {
    for (int w=0; w < model->nterms; ++w) {
      total = 0.0;
      for (int i=time; i < model->nseq - 1; ++i) {
	// This loop is kind of going overboard, but at least we
	// do this once per E-M iteration.
	variance = mget(model->topic[k]->variance, w, i + 1);
	m = mget(model->topic[k]->e_log_prob, w, i);
	exp_m = exp(-2.0 * m + 2.0 * variance);
	total += exp_m * (scaled_influence[i - time]
			  * scaled_influence[i - time]);
      }
      gsl_matrix_set(g, w, k, total);
    }
  }
  return g;
}

 /*
  * compute the likelihood of a sequential corpus under an LDA seq
  * model.  return the likelihood bound.
  *
  */

double lda_seq_infer(lda_seq* model,
		      const corpus_seq_t* data,
		      gsl_matrix** suffstats,
		      gsl_matrix* lhoods,
		      const char* file_root) {
   int K = model->ntopics; int W = model->nterms;
   double bound = 0.0;
   lda* lda_model = new_lda_model(K, W);
   lda_post post;
   post.phi = gsl_matrix_calloc(data->max_nterms, K);
   post.log_phi = gsl_matrix_calloc(data->max_nterms, K);
   post.model = lda_model;
   int t, d, doc_idx = 0;
   
   if (FLAGS_model == "fixed") {
     // First, pre-compute the functions f and g.
     InfluenceTotalFixed(model, data);
   } else if (FLAGS_model == "dtm") {
     // Do nothing here.
   } else {
     printf("Error.  Unknown model.\n");
     exit(1);
   }

   for (t = FLAGS_min_time; t < FLAGS_max_time && t < data->len; ++t) {
     // Prepare coefficients for the phi updates.  This change is
     // relatively painless.
     gsl_matrix* g = g_alloc(model, data, t);
     
     gsl_matrix* g3_matrix = g3_alloc(model, data, t);
     gsl_matrix* g4_matrix = g4_alloc(model, data, t);
     gsl_matrix* g5_matrix = g5_alloc(model, data, t);
     
     make_lda_from_seq_slice(lda_model, model, t);
     int ndocs = data->corpus[t]->ndocs;
     gsl_matrix* gammas = gsl_matrix_calloc(ndocs, K);
     gsl_matrix** phi_t = (gsl_matrix**) malloc(ndocs
						* sizeof(gsl_matrix*));
     for (d = 0; d < ndocs; d++) {
       gsl_vector gam   = gsl_matrix_row(gammas, d).vector;
       gsl_vector lhood = gsl_matrix_row(lhoods, d).vector;
       post.gamma = &gam;
       post.doc   = data->corpus[t]->doc[d];
       post.lhood = &lhood;
       double doc_lhood;
       // For now, only do the standard, phi-based update.
       doc_lhood = fit_lda_post(d, t, &post, model, g,
				g3_matrix, g4_matrix, g5_matrix);
       if (suffstats == NULL) {
	 printf("Sufficient statistics are NULL.\nFailing.\n");
	 exit(1);
       }
       update_lda_seq_ss(t,
			 data->corpus[t]->doc[d],
			 &post,
			 suffstats);
       phi_t[d] = gsl_matrix_alloc(post.doc->nterms, K);
       gsl_matrix_view phi_view = gsl_matrix_submatrix(
         post.phi,
	 0, 0, post.doc->nterms, K);
       gsl_matrix_memcpy(phi_t[d], &phi_view.matrix);
       bound += doc_lhood;
       doc_idx++;
     }
     
     if (t < data->len - 1) {
       if (FLAGS_model == "fixed") {
	 double l_bound = update_inf_var_fixed(
	       model,
	       data,
	       // Also want a copy of phi for each doc.
	       // Can keep this as a vector for now.
	       phi_t,
	       t,
	       file_root);
	 bound += l_bound;
       }
     }
     
     for (int d=0; d < ndocs; ++d) {
       gsl_matrix_free(phi_t[d]);
     }
     free(phi_t);

     // Write out the gammas and likelihoods.
     string name = StringPrintf("%s/gam-time-%d.dat",
				FLAGS_root_directory.c_str(),
				t);
     mtx_fprintf(name.c_str(), gammas);
     gsl_matrix_free(gammas);

     gsl_matrix_free(g);
     gsl_matrix_free(g3_matrix);
     gsl_matrix_free(g4_matrix);
     gsl_matrix_free(g5_matrix);
   }
   gsl_matrix_free(post.phi);
   gsl_matrix_free(post.log_phi);
   free_lda_model(lda_model);
   return(bound);
}

double fit_lda_seq_st() {
  corpus_seq_t* data =
    read_corpus_seq(FLAGS_corpus_prefix.c_str());  
  lda_seq* model_seq = new_lda_seq(data,
				   NULL,
				   data->nterms,
				   data->len,
				   FLAGS_ntopics);

  double topic_bound = 0.0;
  vector<int> time_boundaries;
  vector<string> time_boundaries_str;
  SplitStringUsing(FLAGS_time_boundaries, ",", &time_boundaries_str);
  for (int i=0; i < time_boundaries_str.size(); ++i) {
    time_boundaries.push_back(atoi(time_boundaries_str[i].c_str()));
  }

  // Read the corpus as a sequence.
  int* ndocs = new int[data->len];
  for (int i=0; i < data->len; ++i) {
    ndocs[i] = data->corpus[i]->ndocs;
  }

  // Initialize the sequence from sufficient statistics.
  init_lda_seq(time_boundaries,
	       FLAGS_root_directory,
	       FLAGS_model,
	       model_seq,
	       ndocs,
	       data->len,
	       FLAGS_top_chain_var,
	       FLAGS_top_obs_var,
	       FLAGS_alpha,
	       FLAGS_min_topic,
	       FLAGS_max_topic,
	       /*seq_docs=*/false);

  for (int k=FLAGS_min_topic; k < FLAGS_max_topic; ++k) {
    // initialize sufficient statistics
    gsl_matrix* topic_suffstats = read_topic_suffstats(
      time_boundaries,
      data->nterms,
      k);

    // fit the variational distribution
    topic_bound += fit_lda_seq_topic(k, model_seq, topic_suffstats);

    const string kFilePrefix = StringPrintf(
      "%s/sslm-topic_%d",
      FLAGS_root_directory.c_str(), k);
    write_sslm_var(model_seq->topic[k], kFilePrefix.c_str());

    gsl_matrix_free(topic_suffstats);
  }

  delete[] ndocs;
  return(topic_bound);
}


static bool Converged(double current, double last) {
  if (fabs((last - current) / (last)) < 5e-6) {
    printf("%f\n",
	   fabs((last - current)));
    printf("%f\n",
	   fabs(last + 1.0));
    printf("%f\n",
	   fabs((last - current) / (last + 1.0)));
    return true;
  } else {
    return false;
  }
}

  /*
static void WriteTopics(const string& root_directory,
			lda_seq* seq,
			int k) {
  gsl_matrix* mean_t = seq->topic[k];
  const string kMeanFilename =
    StringPrintf("%s/topic_mean_%d.dat",
		 root_directory.c_str(),
		 k);
  mtx_fprintf(kMeanFilename.c_str(), mean_t);

  gsl_matrix* variance_t = seq->topic[k];
  const string kVarianceFilename =
    StringPrintf("%s/topic_variance_%d.dat",
		 root_directory.c_str(),
		 k);
  mtx_fprintf(kVarianceFilename.c_str(), mean_t);
}
  */
			

static double ParallelFitTopics(int corpus_length,
				const string& time_boundaries,
				string* topic_boundaries_str,
				double* lhood) {
  // Create a task and set up appropriate flags.
  hash_map<string, string> flags;
  flags["ntopics"] = StringPrintf("%d", FLAGS_ntopics);
  flags["sigma_d"] = StringPrintf("%lf", FLAGS_sigma_d);
  flags["sigma_l"] = StringPrintf("%lf", FLAGS_sigma_l);
  flags["time_resolution"] = StringPrintf("%lf",
					  FLAGS_time_resolution);
  flags["alpha"] = StringPrintf("%lf", FLAGS_alpha);
  flags["model"] = FLAGS_model;

  flags["rng_seed"] = StringPrintf("%d", FLAGS_rng_seed);

  flags["lda_max_em_iter"] = StringPrintf("%d", FLAGS_lda_max_em_iter);
  flags["lda_sequence_min_iter"] = StringPrintf("%d", FLAGS_lda_sequence_min_iter);
  flags["lda_sequence_max_em_iter"] = StringPrintf("%d", FLAGS_lda_sequence_max_em_iter);

  flags["top_chain_var"] = StringPrintf("%lf", FLAGS_top_chain_var);
  flags["top_obs_var"] = StringPrintf("%lf", FLAGS_top_obs_var);
  flags["fix_topics"] = StringPrintf("%d", FLAGS_fix_topics);
  flags["root_directory"] = FLAGS_root_directory;
  flags["influence_flat_years"] = StringPrintf("%d", FLAGS_influence_flat_years);
  flags["time_boundaries"] = time_boundaries;
  // flags["nterms"] = StringPrintf("%d", nterms);
  flags["corpus_prefix"] = FLAGS_corpus_prefix;
  flags["normalize_docs"] = FLAGS_normalize_docs;
  flags["max_number_time_points"] = StringPrintf("%d",
						 FLAGS_max_number_time_points);

  const int kNumberTasks = min(FLAGS_number_tasks,
			       FLAGS_ntopics);

  const int kTopicsPerTask = ((FLAGS_ntopics
			       + kNumberTasks - 1)
			      / kNumberTasks);
  const int kNumberBigTasks = (kNumberTasks
			       - (kTopicsPerTask * kNumberTasks
				  - FLAGS_ntopics));
  vector<int> topic_boundaries;
  topic_boundaries.push_back(0);
  *topic_boundaries_str = "0";
  int i=1;
  for (int t=kTopicsPerTask;
       t < FLAGS_ntopics;
       ) {
    *topic_boundaries_str += StringPrintf(",%d", t);
    topic_boundaries.push_back(t);
    if (i < kNumberBigTasks) {
      t += kTopicsPerTask;
    } else {
      t += kTopicsPerTask - 1;
    }
    ++i;
  }
  topic_boundaries.push_back(FLAGS_ntopics);;
  *topic_boundaries_str += StringPrintf(",%d", FLAGS_ntopics);
  flags["topic_boundaries"] = *topic_boundaries_str;

  vector<string> done_sentinels;
  vector<Resource*> resources;
  vector<Task*> tasks;
  for (int i=0; i < kNumberTasks; ++i) {
    const int kMinTopic = topic_boundaries[i];
    const int kMaxTopic = topic_boundaries[i + 1];

    flags["min_topic"] = StringPrintf(
      "%d", kMinTopic);
    flags["max_topic"] = StringPrintf(
      "%d", kMaxTopic);
    flags["sentinel_filename"] = StringPrintf(
	  "%s/lda-fit-topics_%d_%d.done",
	  FLAGS_root_directory.c_str(),
	  kMinTopic,
	  kMaxTopic);
    done_sentinels.push_back(flags["sentinel_filename"]);
  
    if (FLAGS_resume_stage == "topic"
	&& c2_lib::FileExists(flags["sentinel_filename"])) {
      continue;
    }

    Task* task = TaskFactory::NewTopicsFitTask(
      i,
      FLAGS_root_directory.c_str(),
      flags,
      resources,
      flags["sentinel_filename"]);
    tasks.push_back(task);
    
  }

  if (FLAGS_resume_stage == "topic") {
    FLAGS_resume_stage = "";
  }

  // Run.
  for (int t=0; t < tasks.size(); ++t) {
    //    printf("Starting task.\n");
    tasks[t]->Start();
  }

  c2_lib::WaitOnTasks(tasks);
  
  *lhood = ReadLikelihoods(done_sentinels);
}

// Compares objects, to sort them from highest priority to lowest.
bool PriorityCmp(Task* l, Task* r) {
  return l->Priority() > r->Priority();
}

static void ParallelFitDocs(corpus_seq_t* data,
			    int corpus_length,
			    int number_tasks,
			    const vector<int>& time_boundaries,
			    const string& time_boundaries_str,
			    double* lhood) {
  // Create a task and set up appropriate flags.
  hash_map<string, string> flags;
  flags["ntopics"] = StringPrintf("%d", FLAGS_ntopics);
  flags["sigma_d"] = StringPrintf("%lf", FLAGS_sigma_d);
  flags["sigma_l"] = StringPrintf("%lf", FLAGS_sigma_l);
  flags["time_resolution"] = StringPrintf("%lf",
					  FLAGS_time_resolution);
  flags["alpha"] = StringPrintf("%lf", FLAGS_alpha);
  flags["model"] = FLAGS_model;

  flags["rng_seed"] = StringPrintf("%d", FLAGS_rng_seed);

  flags["lda_max_em_iter"] = StringPrintf("%d", FLAGS_lda_max_em_iter);
  flags["lda_sequence_min_iter"] = StringPrintf("%d", FLAGS_lda_sequence_min_iter);
  flags["lda_sequence_max_em_iter"] = StringPrintf("%d", FLAGS_lda_sequence_max_em_iter);

  flags["top_chain_var"] = StringPrintf("%lf", FLAGS_top_chain_var);
  flags["top_obs_var"] = StringPrintf("%lf", FLAGS_top_obs_var);
  flags["fix_topics"] = StringPrintf("%d", FLAGS_fix_topics);
  flags["root_directory"] = FLAGS_root_directory;
  flags["influence_flat_years"] = StringPrintf("%d", FLAGS_influence_flat_years);
  flags["time_boundaries"] = time_boundaries_str;
  flags["max_number_time_points"] = StringPrintf("%d",
						 FLAGS_max_number_time_points);
  // flags["nterms"] = StringPrintf("%d", nterms);
  flags["corpus_prefix"] = FLAGS_corpus_prefix;
  flags["normalize_docs"] = FLAGS_normalize_docs;

  vector<string> done_sentinels;
  vector<Resource*> resources;
  vector<Task*> tasks;
  for (int i=0; i < number_tasks; ++i) {
  //  for (int i=0; i < 1; ++i) {
    const int kMinTime = time_boundaries[i];
    const int kMaxTime = time_boundaries[i + 1];

    flags["min_time"] = StringPrintf(
      "%d", kMinTime);
    flags["max_time"] = StringPrintf(
      "%d", kMaxTime);
    flags["sentinel_filename"] = StringPrintf(
	  "%s/lda-fit-times_%d_%d.done",
	  FLAGS_root_directory.c_str(),
	  kMinTime,
	  kMaxTime);
    done_sentinels.push_back(flags["sentinel_filename"]);
    
    if (FLAGS_resume_stage == "doc"
	&& c2_lib::FileExists(flags["sentinel_filename"])) {
      continue;
    }
    QSubTask* task = (QSubTask*) TaskFactory::NewDocsFitTask(
      i,
      FLAGS_root_directory.c_str(),
      flags,
      resources,
      flags["sentinel_filename"]);

    int number_docs_sq = 0;
    int number_docs = 0;
    for (int j=kMinTime; j < kMaxTime; ++j) {
      corpus_t* corpus = data->corpus[j];
      number_docs_sq += corpus->ndocs * corpus->ndocs;
      number_docs += corpus->ndocs;
    }

    printf("number_docs is %d.  number_docs_sq is %d.\n",
	   number_docs, number_docs_sq);
    int walltime = number_docs_sq / 1000000 / 6 + 5;
    printf("Setting walltime to %d.\n", walltime);
    task->set_walltime_hours(walltime);

    int memory_mb = number_docs / 10 + 11800;
    // 10959 is too low for 90-93.
    // 11xxx also seemed too low.
    printf("Setting memory_mb to %d.\n", memory_mb);
    // task->set_memory_mb(memory_mb);
    // We only need to store half of this information, hopefully..
    task->set_memory_mb((int) (memory_mb * 0.32));
    task->set_memory_mb(4000);
    tasks.push_back(task);
  }
  if (FLAGS_resume_stage == "doc") {
    FLAGS_resume_stage = "";
  }

  std::sort(tasks.begin(), tasks.end(), PriorityCmp);

  // Run.
  for (int t=0; t < tasks.size(); ++t) {
    //    printf("Starting task.\n");
    tasks[t]->Start();
  }

  c2_lib::WaitOnTasks(tasks);

  for (vector<Task*>::iterator it=tasks.begin();
       it != tasks.end();
       ++it) {
    delete *it;
  }
  
  *lhood = ReadLikelihoods(done_sentinels);
}

static void CreateTimeBoundaries(int corpus_length,
				 vector<int>* time_boundaries,
				 string* time_boundaries_str,
				 int* number_tasks) {
  *number_tasks = min(FLAGS_number_tasks,
		      corpus_length);

  const int kTimesPerTask = ((corpus_length
			      + (*number_tasks) - 1)
			     / (*number_tasks));
  const int kNumberBigTasks = (*number_tasks
			       - (kTimesPerTask * (*number_tasks)
				  - corpus_length));

  time_boundaries->push_back(0);
  *time_boundaries_str = "0";
  int i=1;
  for (int t=kTimesPerTask;
       t < corpus_length;
       ) {
    *time_boundaries_str += StringPrintf(",%d", t);
    time_boundaries->push_back(t);
    if (i < kNumberBigTasks) {
      t += kTimesPerTask;
    } else {
      t += kTimesPerTask - 1;
    }
    ++i;
  }
  time_boundaries->push_back(corpus_length);
  *time_boundaries_str += StringPrintf(",%d", corpus_length);
}

bool FitParallelLDASeq(const vector<int>& topic_boundaries) {
  // !!! make this an option
  // Read in the corpus so we know how many terms.

  printf("Preparing model.\n");

  corpus_seq_t* data_full = read_corpus_seq(
    FLAGS_corpus_prefix.c_str());
  lda_seq* model_seq = new_lda_seq(data_full,
				   NULL,
				   data_full->nterms,
				   data_full->len,
				   FLAGS_ntopics);

  vector<int> time_boundaries;
  string time_boundaries_str;
  int number_doc_tasks;
  CreateTimeBoundaries(data_full->len,
		       &time_boundaries,
		       &time_boundaries_str,
		       &number_doc_tasks);
  
  if (FLAGS_resume_stage != "doc"
      && FLAGS_resume_stage != "topic") {
    RewriteTopicsAsSequential(time_boundaries,
			      topic_boundaries,
			      model_seq);
  }

  // Sleep for a minute to give the machine a chance to rest before
  // continuing.

  printf("fitting.. \n");
  // estimate dynamic topic model

  outlog("\n%s\n","### FITTING DYNAMIC TOPIC MODEL ###");

  bool converged = false;
  double last_lhood = -1.0;
  int iteration = 0;
  while (!converged && iteration < FLAGS_lda_sequence_max_em_iter) {
    ++iteration;
    double d_lhood = 0.0;
    outlog("\n%s\n", "Fitting documents.");
    printf("Fitting documents.\n");
    if (FLAGS_resume_stage != "topic") {
      ParallelFitDocs(data_full,
		      data_full->len,
		      number_doc_tasks,
		      time_boundaries,
		      time_boundaries_str,
		      &d_lhood);
    }

    outlog("\n%s\n", "Fitting topics.");
    double t_lhood;
    string topic_boundaries;
    printf("Fitting topics.\n");
    ParallelFitTopics(data_full->len,
		      time_boundaries_str,
		      &topic_boundaries,
		      &t_lhood);

    converged = Converged(t_lhood + d_lhood,
			  last_lhood);
    last_lhood = t_lhood + d_lhood;
    printf("Log likelihood bound: %lf\n", last_lhood);
  }

  return true;
}

double fit_lda_seq_topic(int topic,
			 lda_seq* model,
			 gsl_matrix* topic_ss) {
  double lhood = 0, lhood_term = 0;
  outlog( "\nFitting topic %d of %d", topic, model->ntopics);
  lhood_term = fit_sslm(model->topic[topic], topic_ss);
  lhood += lhood_term;
  return(lhood);
}


static void ReadSSLMAllTopics(lda_seq* var,
			      int ntopics) {
  for (int k=0; k < ntopics; ++k) {
    const string kFilePrefix = StringPrintf(
      "%s/sslm-topic_%d",
      FLAGS_root_directory.c_str(), k);
    // We've already allocated a topic.  Delete it and reallocate it.
    free(var->topic[k]);
    var->topic[k] = read_sslm_var(kFilePrefix.c_str());
  }
}

 /*
  * fit an lda sequence model:
  *
  * . for each time period
  * .     set up lda model with E[log p(w|z)] and \alpha
  * .     for each document
  * .         perform posterior inference
  * .         update sufficient statistics/likelihood
  * .
  * . maximize topics
  *
  */

double fit_lda_seq_sd() {
  corpus_seq_t* data =
    read_corpus_seq(FLAGS_corpus_prefix.c_str());
  // We delete corpus data for the minimum and maximum times to
  // save memory.
  int* ndocs = new int[data->len];
  int max_ndocs = 0;
  for (int i=0; i < data->len; ++i) {
    ndocs[i] = data->corpus[i]->ndocs;
    if (ndocs[i] > max_ndocs) {
      max_ndocs = ndocs[i];
    }
    if (i >= FLAGS_min_time && i < FLAGS_max_time) {
      continue;
    }
    free_corpus(data->corpus[i]);
    data->corpus[i] = NULL;
  }
  lda_seq* m = new_lda_seq(NULL,
			   ndocs,
			   data->nterms,
			   data->len,
			   FLAGS_ntopics);

  const int K = m->ntopics, W = m->nterms;
  
  vector<string> time_boundaries_str;
  vector<int> time_boundaries;
  SplitStringUsing(FLAGS_time_boundaries, ",",
		   &time_boundaries_str);
  for (int i=0; i < time_boundaries_str.size(); ++i) {
    time_boundaries.push_back(atoi(time_boundaries_str[i].c_str()));
  }

  // Initialize topic sufficient statistics and set them all to zero.
  gsl_matrix* topic_suffstats[K];
  for (int k = 0; k < K; ++k) {
    topic_suffstats[k] = gsl_matrix_calloc(W, data->len);
  }

  init_lda_seq(time_boundaries,
	       FLAGS_root_directory,
	       FLAGS_model,
	       m,
	       ndocs,
	       data->len,
	       FLAGS_top_chain_var,
	       FLAGS_top_obs_var,
	       FLAGS_alpha,
	       /* min_topic */ 0,
	       /* max_topic */ FLAGS_ntopics,
	       /*seq_docs=*/true);

  delete[] ndocs;

  // set up variables

  gsl_matrix* lhoods = gsl_matrix_calloc(max_ndocs, K+1);

  double bound = 0;
  double convergence = LDA_SEQ_EM_THRESH + 1;
  
  // LDA_INFERENCE_MAX_ITER = 1;

  gsl_matrix_set_zero(lhoods);

  // compute the likelihood of a sequential corpus under an LDA
  // seq model and find the evidence lower bound.
  bound = lda_seq_infer(m,
			data,
			topic_suffstats,
			lhoods,
			FLAGS_root_directory.c_str());

  // Write out the variational distribution.

  // Write out intermediate matrices.  Note that we are taking
  // submatrices because we populated only a small fraction of these
  // matrices.
  write_lda_seq_docs(FLAGS_model,
		     FLAGS_root_directory,
		     FLAGS_min_time,
		     FLAGS_max_time,
		     m);

  // Delete topic sufficient statistics.
  for (int k = 0; k < K; k++) {
    write_topic_suffstats(FLAGS_min_time,
			  FLAGS_max_time,
			  topic_suffstats,
			  data->nterms,
			  k);
    // Note that we don't need to write the sslm_var,
    // since it's not changed by this task.

    gsl_matrix_free(topic_suffstats[k]);
  }


  return(bound);
}

/*
 * update lda sequence sufficient statistics from an lda posterior
 *
 */
void update_lda_seq_ss(int time,
                       const doc_t* doc,
                       const lda_post* post,
                       gsl_matrix** ss) {
  int K = post->phi->size2, N = doc->nterms;
  int k, n;

  for (k = 0; k < K; k++) {
    gsl_matrix* topic_ss = ss[k];
    for (n = 0; n < N; n++) {
      int w = doc->word[n];
      int c = doc->count[n];
      minc(topic_ss, w, time, c * mget(post->phi, n, k));
    }
  }
}

/*
 * initialize from sufficient statistics (expected counts).
 *
 */

void init_lda_seq(
    const vector<int> time_boundaries,
    const string& root_directory,
    const string& model_type,
    lda_seq* model,
    int* ndocs,
    int len,
    double topic_chain_variance,
    double topic_obs_variance,
    double alpha,
    int min_topic,
    int max_topic,
    bool seq_docs) {
  gsl_vector_set_all(model->alpha, alpha);

  gsl_matrix* tmp_terms_time = gsl_matrix_calloc(model->nterms,
						 model->nseq);
  gsl_vector* tmp_terms = gsl_vector_calloc(model->nterms);
  for (int k = min_topic; k < max_topic; k++) {
    const string kFilePrefix =
      StringPrintf("%s/sslm-topic_%d",
		   FLAGS_root_directory.c_str(),
		   k);
    model->topic[k] = read_sslm_var(kFilePrefix.c_str());
    model->topic[k]->chain_variance = topic_chain_variance;
    model->topic[k]->obs_variance = topic_obs_variance;

    // Allocate sequential data.
    if (seq_docs) {
      model->topic[k]->w_phi_l = gsl_matrix_calloc(model->nterms,
						   model->nseq);
      model->topic[k]->m_update_coeff = gsl_matrix_calloc(model->nterms,
							  FLAGS_max_time - FLAGS_min_time);
    } else {
      model->topic[k]->w_phi_l = gsl_matrix_calloc(model->nterms,
						   model->nseq);
      model->topic[k]->m_update_coeff = gsl_matrix_calloc(model->nterms,
							  model->nseq);
      model->topic[k]->m_update_coeff_g = gsl_matrix_calloc(model->nterms,
							    model->nseq);
    }
    // Load sequential data.
    const int kMaxTime = time_boundaries[time_boundaries.size() - 1];
    for (int i=0; i + 1 < time_boundaries.size(); ++i) {
      int min_time = time_boundaries[i];
      int max_time = time_boundaries[i + 1];
      string filename = StringPrintf("%s/w_phi_l-topic_%d-time_%d_%d.dat",
		root_directory.c_str(),
		k, min_time, max_time);
      gsl_matrix_view view = gsl_matrix_submatrix(
              model->topic[k]->w_phi_l,
	      0, min_time, model->nterms, max_time - min_time);
      mtx_fscanf(filename.c_str(), &view.matrix);

      view = gsl_matrix_submatrix(
	  tmp_terms_time,
	  0, min_time, model->nterms, max_time - min_time);
      filename = StringPrintf("%s/m_update_coeff-topic_%d-time_%d_%d.dat",
			      root_directory.c_str(),
			      k, min_time, max_time);      
      mtx_fscanf(filename.c_str(), &view.matrix);
    }

    // Populate the m_update_coeff matrices.
    if (!seq_docs) {
      for (int t=0; t < kMaxTime; ++t) {
	gsl_vector_view m_update_coeff =
	  gsl_matrix_column(tmp_terms_time, t);
	for (int i=t; i < kMaxTime; ++i) {
	  gsl_vector_view muc_i =
	    gsl_matrix_column(model->topic[k]->m_update_coeff, i);
	  gsl_vector_memcpy(tmp_terms, &m_update_coeff.vector);
	  gsl_vector_scale(tmp_terms, scaled_influence[i - t]);
	  gsl_vector_add(&muc_i.vector, tmp_terms);
	}

	gsl_vector_view w_phi_l_i =
	  gsl_matrix_column(model->topic[k]->w_phi_l, t);
	for (int i=t; i < kMaxTime; ++i) {
	  gsl_vector_view muc_i =
	    gsl_matrix_column(model->topic[k]->m_update_coeff_g, i);
	  gsl_vector_memcpy(tmp_terms, &w_phi_l_i.vector);
	  gsl_vector_scale(tmp_terms, scaled_influence[i - t]);
	  gsl_vector_add(&muc_i.vector, tmp_terms);
	}
      }
    }
  }

  if (model_type == "fixed") {
    model->influence = (inf_var*) malloc(sizeof(inf_var));
    model->influence->doc_weights =
      new gsl_matrix*[model->nseq];
    model->influence->renormalized_doc_weights =
      new gsl_matrix*[model->nseq];
    model->influence->ntime = model->nseq;

    // Read in both normalized and unnormalized doc weights.
    for (int t=0; t < model->nseq; ++t) {
      string filename = StringPrintf("%s/influence_time-%03d",
	      root_directory.c_str(), t);
      outlog("\n reading influence weights for time %d from %s",
	     t, filename.c_str());
      model->influence->doc_weights[t] = gsl_matrix_alloc(
        ndocs[t],
	model->ntopics);
      mtx_fscanf(filename.c_str(),
		 model->influence->doc_weights[t]);

      filename = StringPrintf("%s/renormalized_influence_time-%03d",
	      root_directory.c_str(), t);
      outlog("\n reading influence weights for time %d from %s",
	     t, filename.c_str());
      model->influence->renormalized_doc_weights[t] = gsl_matrix_alloc(
        ndocs[t],
	model->ntopics);
      mtx_fscanf(filename.c_str(),
		 model->influence->renormalized_doc_weights[t]);
    }
  } else {
    model->influence = NULL;
  }
  gsl_vector_free(tmp_terms);
  gsl_matrix_free(tmp_terms_time);
}


/*
 * allocate lda seq
 *
 */

lda_seq* new_lda_seq(corpus_seq_t* data, int* ndocs,
		     int W, int T, int K) {
    lda_seq* model = (lda_seq*) malloc(sizeof(lda_seq));

    model->ntopics = K;
    model->nterms = W;
    model->nseq = T;

    model->alpha = gsl_vector_alloc(K);
    model->topic = (sslm_var**) malloc(sizeof(sslm_var*) * K);

    // Create the vectors of total counts for each time.
    model->influence_sum_lgl = (gsl_matrix**) malloc(sizeof(gsl_matrix*) * K);

    for (int k = 0; k < K; k++) {
      model->influence_sum_lgl[k] = gsl_matrix_calloc(W, T);

      model->topic[k] = sslm_var_alloc(W, T);
      if (k < FLAGS_fix_topics) {
	model->topic[k]->chain_variance = 1e-10;
      }
      model->topic[k]->m_update_coeff = NULL;
      model->topic[k]->m_update_coeff_g = NULL;
      model->topic[k]->w_phi_l = NULL;
    }

    int* ndocs_tmp;
    if (data == NULL) {
      ndocs_tmp = ndocs;
    } else {
      ndocs_tmp = new int[data->len];
      for (int i=0; i < data->len; ++i) {
	ndocs_tmp[i] = data->corpus[i]->ndocs;
      }
    }
    model->influence = inf_var_alloc(K, ndocs_tmp, T);
    if (data != NULL) {
      delete[] ndocs_tmp;
    }
    return model;
}

}  // namespace dtm
