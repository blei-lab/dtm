// Authors: David Blei (blei@cs.princeton.edu)
//          Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Copyright 2011 Sean Gerrish and David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#ifndef LDA_H
#define LDA_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_lambert.h>
#include <gsl/gsl_rng.h>

#include "param.h"
#include "data.h"
#include "gsl-wrappers.h"

/*
 * functions for posterior inference in the latent dirichlet
 * allocation model.
 *
 */

#define LDA_INFERENCE_CONVERGED 1e-8
#define LDA_SEED_INIT 1
#define LDA_INIT_SMOOTH 1.0
#define LDA_EM_CONVERGED 5e-5
#define LDA_USE_VAR_BAYES 0
#define LDA_TOPIC_DIR_PARAM 0.001

// lda model

typedef struct lda {
    int ntopics;         // number of topics
    int nterms;          // vocabulary size
    gsl_matrix* topics;  // each column is a topic (V X K)
    gsl_vector* alpha;   // dirichlet parameters
} lda;

// lda posterior

typedef struct lda_post {
    doc_t* doc;          // document associated to this posterior
    lda* model;          // lda model
    gsl_matrix* phi;     // variational mult parameters (nterms x K)
    gsl_matrix* log_phi; // convenient for computation (nterms x K)
    gsl_vector* gamma;   // variational dirichlet parameters (K)
    gsl_vector* lhood;   // a K+1 vector, sums to the lhood bound
    gsl_vector* doc_weight;  // Not owned by this structure.
    gsl_vector* renormalized_doc_weight;  // Not owned by this structure.
} lda_post;

// lda sufficient statistics

typedef struct lda_suff_stats {
  gsl_matrix* topics_ss;
} lda_suff_stats;


// new lda model and suff stats

lda* new_lda_model(int ntopics, int nterms);
void free_lda_model(lda* m);
lda_suff_stats* new_lda_suff_stats(lda* model);
void reset_lda_suff_stats(lda_suff_stats* ss);
lda_post* new_lda_post(int ntopics, int max_length);
void free_lda_post(lda_post* p);
void initialize_lda_ss_from_data(corpus_t* data, lda_suff_stats* ss);

// posterior inference

double fit_lda_post(int doc_number, int time,
		    lda_post* p, lda_seq* var,
		    gsl_matrix* g,
		    gsl_matrix* g3,
		    gsl_matrix* g4,
		    gsl_matrix* g5);
void init_lda_post(lda_post* p);
void update_gamma(lda_post* p);
void update_phi(int doc_number, int time,
		lda_post* p, lda_seq* var,
		gsl_matrix* g);
void update_phi_dim(int doc_number, int time,
		    lda_post* p, lda_seq* var,
		    gsl_matrix* g);
void update_phi_fixed(int doc_number, int time,
		      lda_post* p, lda_seq* var,
		      gsl_matrix* g3_matrix,
		      gsl_matrix* g4_matrix,
		      gsl_matrix* g5_matrix);
void update_phi_multiple(int doc_number, int time,
			 lda_post* p, lda_seq* var,
			 gsl_matrix* g);

// compute the likelihood bound

double compute_lda_lhood(lda_post* p);

// EM algorithm

double lda_e_step(lda* model, corpus_t* data, lda_suff_stats* ss);
double lda_m_step(lda* model, lda_suff_stats* ss);
void lda_em(lda* model,
	    lda_suff_stats* ss,
	    corpus_t* data,
	    int max_iter,
	    char* outname);

// reading and writing

lda_suff_stats* read_lda_suff_stats(char* filename, int ntopics, int nterms);
void write_lda(lda* model, char* name);
void write_lda_suff_stats(lda_suff_stats* ss, char* name);
lda* read_lda(int ntopics, int nterms, char* name);


void initialize_lda_ss_from_random(corpus_t* data, lda_suff_stats* ss);

#endif
