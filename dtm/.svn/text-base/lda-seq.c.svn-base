// Authors: David Blei (blei@cs.princeton.edu)
//          Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Copyright 2011 Sean Gerrish and David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#include "lda-seq.h"

#include "gflags.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector_double.h>

extern int LDA_INFERENCE_MAX_ITER;

static double* scaled_influence = NULL;

// const int TIME = 61;
const int TIME = -3;
const double PI = 3.141592654;

DEFINE_int32(lda_sequence_max_iter,
	     20,
	     "The maximum number of iterations.");
DEFINE_int32(lda_sequence_min_iter,
	     1,
	     "The maximum number of iterations.");

DEFINE_int32(forward_window,
	     1,
	     "The forward window for deltas. "
	     "If negative, we use a beta with mean "
	     "5.");

DEFINE_string(normalize_docs,
	      "normalize",
	      "Describes how documents's wordcounts "
	      "are considered for finding influence. "
	      "Options are \"normalize\", \"none\", "
	      "\"occurrence\", \"log\", or \"log_norm\".");

DEFINE_int32(save_time,
	     1e20,
	     "Save a specific time.  If -1, save all times.");

DEFINE_int32(fix_topics,
	     0,
	     "Fix a set of this many topics. This amounts "
	     "to fixing these topics' variance at 1e-10.");


DECLARE_string(model);
DECLARE_int32(max_number_time_points);
DECLARE_double(sigma_d);
DECLARE_double(sigma_l);
DECLARE_double(sigma_c);
DECLARE_double(sigma_cv);

/*
 * populate an LDA model at a particular time point
 *
 */

inf_var* inf_var_alloc(int number_topics,
		       corpus_seq_t* corpus_seq) {
  // Hate to do this, but I had trouble using it before.  This should
  // be the first place we use it; otherwise we'll get a sigsev nil.
  if (scaled_influence == NULL) {
    scaled_influence = NewScaledInfluence(FLAGS_max_number_time_points);
  }

  inf_var* inf_var_ptr = (inf_var*) malloc(sizeof(inf_var));
  inf_var_ptr->doc_weights = (gsl_matrix**) malloc(sizeof(gsl_matrix*)
						   * corpus_seq->len);
  inf_var_ptr->renormalized_doc_weights = (gsl_matrix**) malloc(
    sizeof(gsl_matrix*)
    * corpus_seq->len);
  inf_var_ptr->ntime = corpus_seq->len;
  int i=0;
  for (i=0; i < corpus_seq->len; ++i) {
    corpus_t* corpus = corpus_seq->corpus[i];
    outlog("creating matrix. %d %d", corpus->ndocs, number_topics);
    if (corpus->ndocs == 0) {
      inf_var_ptr->doc_weights[i] = (gsl_matrix*) malloc(sizeof(gsl_matrix));
      inf_var_ptr->doc_weights[i]->size1 = 0;
      inf_var_ptr->doc_weights[i]->size2 = number_topics;
      inf_var_ptr->renormalized_doc_weights[i] = (gsl_matrix*) malloc(sizeof(gsl_matrix));
      inf_var_ptr->renormalized_doc_weights[i]->size1 = 0;
      inf_var_ptr->renormalized_doc_weights[i]->size2 = number_topics;
    } else {
      inf_var_ptr->doc_weights[i] = gsl_matrix_calloc(corpus->ndocs,
						      number_topics);
      inf_var_ptr->renormalized_doc_weights[i] = gsl_matrix_calloc(corpus->ndocs,
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

  char name[400];
  // Dump the top topics for each word.
  sprintf(name, "%s%ld_doc_term_topics.dat", root, t);
  FILE* f = fopen(name, "w");
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
			    const char* root,
			    int dump_doc_stats) {
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
  double* total_terms = (double*) malloc(sizeof(double) * corpus->ndocs);
  double* renormalization_totals = (double*) malloc(sizeof(double) * corpus->ndocs);
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

    if (FLAGS_save_time == t) {
      outlog("Updating topic %ld, time %ld.", k, t);
      char name[400];
      sprintf(name, "%s%ld_%ld_weighted_document_terms.dat", root, k, t);
      FILE* f = fopen(name, "w");
      params_write_sparse_gsl_matrix(f, "W_phi", W_phi);
      fclose(f);
      
      sprintf(name, "%s%ld_%ld_weighted_document_terms_var.dat", root, k, t);
      f = fopen(name, "w");
      params_write_sparse_gsl_matrix(f, "W_phi_var", W_phi_var);
      fclose(f);

      sprintf(name, "%s%ld_%ld_phi.dat", root, k, t);
      f = fopen(name, "w");
      params_write_gsl_matrix(f, "phi 0", phi[0]);
      //     params_write_gsl_matrix(f, "phi 2", phi[1]);
      // params_write_gsl_matrix(f, "phi 2", phi[7]);
      fclose(f);

      sprintf(name, "%s%ld_%ld_weighted_document_terms_sq.dat", root, k, t);
      f = fopen(name, "w");
      params_write_sparse_gsl_matrix(f, "W2_phi2", W2_phi2);
      fclose(f);

      sprintf(name, "%s%ld_%ld_weighted_response.dat", root, k, t);
      f = fopen(name, "w");
      params_write_gsl_vector_multiline(f, "weighted_response", d_tmp);
      fclose(f);

      sprintf(name, "%s%ld_%ld_response.dat", root, k, t);
      f = fopen(name, "w");
      params_write_gsl_vector_multiline(f, "response", response);
      fclose(f);

      sprintf(name, "%s%ld_%ld_exp_h.dat", root, k, t);
      f = fopen(name, "w");
      params_write_gsl_vector_multiline(f, "exp_h", exp_h_tmp);
      fclose(f);

      if (dump_doc_stats || dump_count % 4 == 0) {
	sprintf(name, "%s%ld_%ld_document_document_matrix.dat", root, k, t);
	f = fopen(name, "w");
	params_write_gsl_matrix(f, "document_document_matrix", dd_tmp);
	fclose(f);
	
	sprintf(name, "%s%ld_%ld_exp_h.dat", root, k, t);
	f = fopen(name, "w");
	params_write_gsl_vector_multiline(f, "exp_h", exp_h_tmp);
	fclose(f);
      }
    }

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

     for (int i = 0; i < corpus->ndocs; ++i) {
       double value = gsl_vector_get(&document_weights.vector, i);

       // While we're here, increment the likelihood.
       lhood += (-(value * value + FLAGS_sigma_l * FLAGS_sigma_l)
		 / (2.0 * FLAGS_sigma_d * FLAGS_sigma_d)
		 - 0.5 * log(2 * PI)
		 - log(FLAGS_sigma_d * FLAGS_sigma_d));
     }
  }
  free(total_terms);
  free(renormalization_totals);
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
     for (k = 0; k < lda_seq_m->ntopics; k++)
     {
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
     for (int k = 0; k < model->ntopics; ++k) {
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
     for (int k = 0; k < model->ntopics; ++k) {
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

static void InferDTMSeq(const int K,
			unsigned int iter,
			unsigned int last_iter,
			const corpus_seq_t* data,
			gsl_matrix* gammas,
			gsl_matrix* lhoods,
			lda* lda_model,
			lda_post* post,
			lda_seq* model,
			gsl_matrix** suffstats,
			double* bound) {
  int doc_index = 0;
  for (int t = 0; t < data->len; t++) {
    // Prepare coefficients for the phi updates.  This change is
    // relatively painless.
    make_lda_from_seq_slice(lda_model, model, t);
    int ndocs = data->corpus[t]->ndocs;
    for (int d = 0; d < ndocs; d++) {
      gsl_vector gam   = gsl_matrix_row(gammas, doc_index).vector;
      gsl_vector lhood = gsl_matrix_row(lhoods, doc_index).vector;
      post->gamma = &gam;
      post->doc   = data->corpus[t]->doc[d];
      post->lhood = &lhood;
      double doc_lhood;
      // For now, only do the standard, phi-based update.
      if (iter == 0) {
	doc_lhood = fit_lda_post(d, t,
				 post, NULL,
				 NULL,
				 NULL,
				 NULL,
				 NULL);
      } else {
	doc_lhood = fit_lda_post(d, t, post, model, NULL,
				 NULL, NULL, NULL);
      }
      if (suffstats != NULL) {
	update_lda_seq_ss(t,
			  data->corpus[t]->doc[d],
			  post,
			  suffstats);
      }
      *bound += doc_lhood;
      doc_index++;
    }
  }
}

static void InferDIMSeq(const int K,
			unsigned int iter,
			unsigned int last_iter,
			const char* file_root,
			const corpus_seq_t* data,
			gsl_matrix* gammas,
			gsl_matrix* lhoods,
			lda* lda_model,
			lda_post* post,
			lda_seq* model,
			gsl_matrix** suffstats,
			double* bound) {
  int doc_index = 0;
  for (int t = 0; t < data->len; t++) {
    // Prepare coefficients for the phi updates.  This change is
    // relatively painless.
    gsl_matrix* g = g_alloc(model, data, t);

    gsl_matrix* g3_matrix = g3_alloc(model, data, t);
    gsl_matrix* g4_matrix = g4_alloc(model, data, t);
    gsl_matrix* g5_matrix = g5_alloc(model, data, t);
    
    make_lda_from_seq_slice(lda_model, model, t);
    int ndocs = data->corpus[t]->ndocs;
    gsl_matrix** phi_t = (gsl_matrix**) malloc(ndocs
					       * sizeof(gsl_matrix*));
    for (int d = 0; d < ndocs; d++) {
      gsl_vector gam   = gsl_matrix_row(gammas, doc_index).vector;
      gsl_vector lhood = gsl_matrix_row(lhoods, doc_index).vector;
      post->gamma = &gam;
      post->doc   = data->corpus[t]->doc[d];
      post->lhood = &lhood;
      double doc_lhood;
      // For now, only do the standard, phi-based update.
      if (iter == 0) {
	doc_lhood = fit_lda_post(d, t,
				 post, NULL,
				 NULL,
				 NULL,
				 NULL,
				 NULL);
      } else {
	doc_lhood = fit_lda_post(d, t, post, model, g,
				 g3_matrix, g4_matrix, g5_matrix);
      }
      if (suffstats != NULL) {
	update_lda_seq_ss(t,
			  data->corpus[t]->doc[d],
			  post,
			  suffstats);
      }
      phi_t[d] = gsl_matrix_alloc(post->doc->nterms, K);
      gsl_matrix_view phi_view = gsl_matrix_submatrix(
        post->phi,
	0, 0, post->doc->nterms, K);
      gsl_matrix_memcpy(phi_t[d], &phi_view.matrix);
      *bound += doc_lhood;
      doc_index++;
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
	  file_root,
	  last_iter || iter >= FLAGS_lda_sequence_min_iter);
	*bound += l_bound;
      }
    }

    for (int d=0; d < ndocs; ++d) {
      gsl_matrix_free(phi_t[d]);
    }
    free(phi_t);
    gsl_matrix_free(g);
    gsl_matrix_free(g3_matrix);
    gsl_matrix_free(g4_matrix);
    gsl_matrix_free(g5_matrix);
  }
}

double lda_seq_infer(lda_seq* model,
		     const corpus_seq_t* data,
		     gsl_matrix** suffstats,
		     gsl_matrix* gammas,
		     gsl_matrix* lhoods,
		     unsigned int iter,
		     unsigned int last_iter,
		     const char* file_root) {
  int K = model->ntopics; int W = model->nterms;
  double bound = 0.0;
  lda* lda_model = new_lda_model(K, W);
  lda_post post;
  post.phi = gsl_matrix_calloc(data->max_nterms, K);
  post.log_phi = gsl_matrix_calloc(data->max_nterms, K);
  post.model = lda_model;
  
  if (FLAGS_model == "fixed") {
    // First, pre-compute the functions f and g.
    InfluenceTotalFixed(model, data);
    InferDIMSeq(K,
		iter,
		last_iter,
		file_root,
		data,
		gammas,
		lhoods,
		lda_model,
		&post,
		model,
		suffstats,
		&bound);
  } else if (FLAGS_model == "dtm") {
    InferDTMSeq(K,
		iter,
		last_iter,
		data,
		gammas,
		lhoods,
		lda_model,
		&post,
		model,
		suffstats,
		&bound);
  } else {
    printf("Error.  Unknown model.\n");
    exit(1);
  }

  gsl_matrix_free(post.phi);
  gsl_matrix_free(post.log_phi);
  free_lda_model(lda_model);
  return(bound);
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

double fit_lda_seq(lda_seq* m, const corpus_seq_t* data,
		   const corpus_seq_t* heldout, const char* file_root) {
     int K = m->ntopics, W = m->nterms;
     int k;

     // initialize sufficient statistics

     gsl_matrix* topic_suffstats[K];
     for (k = 0; k < K; k++) {
	 topic_suffstats[k] = gsl_matrix_calloc(W, data->len);
     }

     // set up variables

     char name[400];
     gsl_matrix* gammas = gsl_matrix_calloc(data->ndocs, K);
     gsl_matrix* lhoods = gsl_matrix_calloc(data->ndocs, K+1);
     gsl_matrix* heldout_gammas = NULL;
     gsl_matrix* heldout_lhoods = NULL;
     if (heldout != NULL)
     {
	 heldout_gammas = gsl_matrix_calloc(heldout->ndocs, K);
	 heldout_lhoods = gsl_matrix_calloc(heldout->ndocs, K+1);
     }

     double bound = 0, heldout_bound = 0, old_bound;
     double convergence = LDA_SEQ_EM_THRESH + 1;

     char root[400];
     sprintf(root, "%s/lda-seq/", file_root);
     make_directory(root);

     char em_log_filename[400];
     sprintf(em_log_filename, "%s/em_log.dat", file_root);
     FILE* em_log = fopen(em_log_filename, "w");

     // run EM

     int iter = 0;
     // LDA_INFERENCE_MAX_ITER = 1;
     short final_iters_flag = 0;
     unsigned int last_iter = 0;
     while (iter < FLAGS_lda_sequence_min_iter ||
	    ((final_iters_flag == 0 || convergence > LDA_SEQ_EM_THRESH)
	     && iter <= FLAGS_lda_sequence_max_iter)
	    && !last_iter) {

	 if (!(iter < FLAGS_lda_sequence_min_iter ||
	       ((final_iters_flag == 0 || convergence > LDA_SEQ_EM_THRESH)
		&& iter <= FLAGS_lda_sequence_max_iter))) {
	   last_iter = 1;
	 }
	 outlog("\nEM iter %3d\n", iter);
	 outlog("%s", "E step\n");
	 fprintf(em_log, "%17.14e %5.3e\n", bound, convergence);

	 old_bound = bound;
	 gsl_matrix_set_zero(gammas);
	 gsl_matrix_set_zero(lhoods);
	 if (heldout != NULL) {
	   gsl_matrix_set_zero(heldout_gammas);
	   gsl_matrix_set_zero(heldout_lhoods);
	 }

	 for (k = 0; k < K; k++) {
	     gsl_matrix_set_zero(topic_suffstats[k]);
	 }

	 // compute the likelihood of a sequential corpus under an LDA
	 // seq model and find the evidence lower bound.
	 bound = lda_seq_infer(m,
			       data,
			       topic_suffstats,
			       gammas,
			       lhoods,
			       iter,
			       last_iter,
			       file_root);
	 if (heldout != NULL) {
	   heldout_bound = lda_seq_infer(m,
					 heldout,
					 NULL,
					 heldout_gammas,
					 heldout_lhoods,
					 iter,
					 last_iter,
					 file_root);
	 }

	// print out the gammas and likelihoods.
        sprintf(name, "%s/gam.dat", root);
        mtx_fprintf(name, gammas);
        sprintf(name, "%s/lhoods.dat", root);
        mtx_fprintf(name, lhoods);
        if (heldout != NULL)
        {
            sprintf(name, "%s/heldout_lhoods.dat", root);
            mtx_fprintf(name, heldout_lhoods);
            sprintf(name, "%s/heldout_gam.dat", root);
            mtx_fprintf(name, heldout_gammas);
        }
        outlog("%s", "\nM step");

	// fit the variational distribution
        double topic_bnd = fit_lda_seq_topics(m, topic_suffstats);
        bound += topic_bnd;

        write_lda_seq(m, root);
        if ((bound - old_bound) < 0)
        {
            if (LDA_INFERENCE_MAX_ITER == 1)
                LDA_INFERENCE_MAX_ITER = 2;
            if (LDA_INFERENCE_MAX_ITER == 2)
                LDA_INFERENCE_MAX_ITER = 5;
            if (LDA_INFERENCE_MAX_ITER == 5)
                LDA_INFERENCE_MAX_ITER = 10;
            if (LDA_INFERENCE_MAX_ITER == 10)
                LDA_INFERENCE_MAX_ITER = 20;

            outlog( "\nWARNING: bound went down %18.14f; "
		    "increasing var iter to %d\n",
                    bound-old_bound, LDA_INFERENCE_MAX_ITER);
        }

	// check for convergence
        convergence = fabs((bound - old_bound) / old_bound);
        if (convergence < LDA_SEQ_EM_THRESH) {
            final_iters_flag = 1;
            LDA_INFERENCE_MAX_ITER = 500;
            outlog("starting final iterations : max iter = %d\n",
                   LDA_INFERENCE_MAX_ITER);
            convergence = 1.0;
        }
        outlog("\n(%02d) lda seq bound=% 15.7f; "
	       "heldout bound=% 15.7f, conv=% 15.7e\n",
               iter, bound, heldout_bound, convergence);
        iter++;
    }
    return(bound);
}


/*
 * read and write lda sequence variational distribution
 *
 */

void write_lda_seq(const lda_seq* model, const char* root) {
  char name[400];
  sprintf(name, "%sinfo.dat", root);
  FILE* f = fopen(name, "w");

  params_write_int(f, "NUM_TOPICS", model->ntopics);
  params_write_int(f, "NUM_TERMS", model->nterms);
  params_write_int(f, "SEQ_LENGTH", model->nseq);
  params_write_gsl_vector(f, "ALPHA", model->alpha);

  fclose(f);

  int k;
  for (k = 0; k < model->ntopics; k++) {
    const int tmp = k;
    outlog("\nwriting topic %03d", tmp);
    sprintf(name, "%stopic-%03d", root, tmp);
    write_sslm_var(model->topic[tmp], name);
  }
  
  if (FLAGS_model == "fixed") {
    for (int t=0; t < model->influence->ntime; ++t) {
      sprintf(name, "%sinfluence_time-%03d", root, t);
      outlog("\nwriting influence weights for time %d to %s", t, name);
      gsl_matrix* influence_t = model->influence->doc_weights[t];
      assert(model->ntopics == influence_t->size2);
      mtx_fprintf(name, influence_t);

      sprintf(name, "%srenormalized_influence_time-%03d", root, t);
      outlog("\nwriting influence weights for time %d to %s", t, name);
      influence_t = model->influence->renormalized_doc_weights[t];
      assert(model->ntopics == influence_t->size2);
      mtx_fprintf(name, influence_t);
    }
  }
}

// Read information about a particular model.
// This model should be named "{root}info.dat"
// and should contain the following rows:
// number_topics
// number_times
// alpha, as a gsl vector
lda_seq* read_lda_seq(const char* root, corpus_seq_t* data) {
  char name[400];
  lda_seq* model = (lda_seq*) malloc(sizeof(lda_seq));

  sprintf(name, "%sinfo.dat", root);
  FILE* f = fopen(name, "r");
  if (f == NULL) {
    outlog("Unable to open file %s.  Failing.", name);
    exit(1);
  }
  params_read_int(f, "NUM_TOPICS", &(model->ntopics));
  params_read_int(f, "NUM_TERMS", &(model->nterms));
  params_read_int(f, "SEQ_LENGTH", &(model->nseq));
  params_read_gsl_vector(f, "ALPHA", &(model->alpha));
  fclose(f);

  model->topic = (sslm_var**) malloc(sizeof(sslm_var*) * model->ntopics);

  for (int k = 0; k < model->ntopics; k++) {
    outlog( "reading topic %d", k);
    sprintf(name, "%stopic-%03d", root, k);
    model->topic[k] = read_sslm_var(name);
    
    model->topic[k]->w_phi_l =
      gsl_matrix_alloc(model->nterms, model->nseq);
    model->topic[k]->w_phi_sum =
      gsl_matrix_alloc(model->nterms, model->nseq);
    model->topic[k]->w_phi_l_sq =
      gsl_matrix_alloc(model->nterms, model->nseq);
    
    if (FLAGS_model == "dim"
	|| FLAGS_model == "regression") {
      sprintf(name, "%sw_phi_l-%d", root, k);
      mtx_fscanf(name, model->topic[k]->w_phi_l);
      
      sprintf(name, "%sw_phi_sum-%d", root, k);
      mtx_fscanf(name, model->topic[k]->w_phi_sum);
      
      sprintf(name, "%sw_phi_l_sq-%d", root, k);
      mtx_fscanf(name, model->topic[k]->w_phi_l_sq);
    }
  }
  
  if (FLAGS_model == "dim"
      || FLAGS_model == "regression"
      && data != NULL) {
    model->influence = (inf_var*) malloc(sizeof(inf_var));
    model->influence->doc_weights =
      (gsl_matrix**) malloc(sizeof(gsl_matrix*));    
    int t;
    model->influence->ntime = model->nseq;
    for (t=0; t < model->nseq; ++t) {
      //	outlog("%d %d", t, model->influence->ntime);
      sprintf(name, "%sinfluence_time-%03d", root, t);
      outlog("\n reading influence weights for time %d from %s", t, name);
      model->influence->doc_weights[t] =
	gsl_matrix_alloc(data->corpus[t]->ndocs,
			 model->ntopics);
      mtx_fscanf(name, model->influence->doc_weights[t]);
    }
  } else {
    model->influence = NULL;
  }
  
  return(model);
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

    for (k = 0; k < K; k++)
    {
        gsl_matrix* topic_ss = ss[k];
        for (n = 0; n < N; n++)
        {
            int w = doc->word[n];
            int c = doc->count[n];
            minc(topic_ss, w, time, c * mget(post->phi, n, k));
        }
    }
}


/*
 * fit lda sequence
 *
 */

double fit_lda_seq_topics(lda_seq* model,
                          gsl_matrix** ss) {
    double lhood = 0, lhood_term = 0;
    int k;

    for (k = 0; k < model->ntopics; k++)
    {
        outlog( "\nfitting topic %02d", k);
        lhood_term = fit_sslm(model->topic[k], ss[k]);
        lhood += lhood_term;
    }
    return(lhood);
}


/*
 * allocate lda seq
 *
 */

lda_seq* new_lda_seq(corpus_seq_t* data,
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
      //      model->w_phi_l = (gsl_matrix*) malloc(sizeof(gsl_matrix));
      // model->w_phi_l_sq = (gsl_matrix*) malloc(sizeof(gsl_matrix*));
      model->influence_sum_lgl[k] = gsl_matrix_calloc(W, T);

      model->topic[k] = sslm_var_alloc(W, T);
      if (k < FLAGS_fix_topics) {
	model->topic[k]->chain_variance = 1e-10;
      }
      model->topic[k]->w_phi_l = gsl_matrix_calloc(W, T);
      model->topic[k]->w_phi_sum = gsl_matrix_calloc(W, T);
      model->topic[k]->w_phi_l_sq = gsl_matrix_calloc(W, T);
    }
    model->influence = inf_var_alloc(K, data);

    return(model);
}


/*
 * initialize from sufficient statistics (expected counts).
 *
 */

void init_lda_seq_from_ss(lda_seq* model,
                          double topic_chain_variance,
                          double topic_obs_variance,
                          double alpha,
                          gsl_matrix* init_suffstats) {
    gsl_vector_set_all(model->alpha, alpha);

    for (int k = 0; k < model->ntopics; k++)
    {
        gsl_vector slice = gsl_matrix_column(init_suffstats, k).vector;
        sslm_counts_init(model->topic[k],
                         topic_obs_variance,
                         topic_chain_variance,
                         &slice);
	if (k < FLAGS_fix_topics) {
	  model->topic[k]->chain_variance = 1e-10;
	}
	model->topic[k]->w_phi_l = gsl_matrix_calloc(model->nterms,
						     model->nseq);
	model->topic[k]->w_phi_sum = gsl_matrix_calloc(model->nterms,
						       model->nseq);
	model->topic[k]->w_phi_l_sq = gsl_matrix_calloc(model->nterms,
							model->nseq);
    }

}
