// Authors: David Blei (blei@cs.princeton.edu)
//          Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Copyright 2011 Sean Gerrish and David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#ifndef LDASEQ_H
#define LDASEQ_H

#include <sys/stat.h>
#include <sys/types.h>

#include "gsl-wrappers.h"
#include "lda.h"

#define LDA_SEQ_EM_THRESH 1e-4
#define SAVE_LAG 10

/*
 * an lda sequence is a collection of simplex sequences for K topics
 * and an alpha vector
 *
 */

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <stdlib.h>
#include <assert.h>

#include "param.h"
#include "ss-lm.h"
#include "data.h"
#include "lda.h"

#define LDA_SEQ_EM_THRESHOLD 1e-5;

// lda sequence variational posterior distribution


// === allocation and initialization ===

inf_var* inf_var_alloc(int number_topics,
		       corpus_seq_t* corpus_seq);
void inf_var_free(inf_var* ptr);

// initialize lda sequence from lda model topics

void init_lda_seq_from_ss(lda_seq* model,
                          double topic_chain_variance,
                          double topic_obs_variance,
                          double alpha,
                          gsl_matrix* init_suffstats);

// === fitting ===


// infer a corpus with an lda-seq

double update_inf_var(lda_seq* seq,
		      const corpus_seq_t* data,
		      gsl_matrix** phi,
		      size_t t,
		      const char* root);
double update_inf_var_multiple(lda_seq* seq,
			       const corpus_seq_t* data,
			       gsl_matrix** phi,
			       size_t t,
			       const char* root);
void update_inf_reg(lda_seq* seq,
		    const corpus_seq_t* data,
		    gsl_matrix** phi,
		    size_t t,
		    const char* root);

double lda_seq_infer(lda_seq* model,
                     const corpus_seq_t* data,
                     gsl_matrix** suffstats,
                     gsl_matrix* gammas,
                     gsl_matrix* lhoods,
		     int iter,
		     const char* file_root);

// fit lda sequence from sufficient statistics

double fit_lda_seq(lda_seq* m,
                   const corpus_seq_t* data,
                   const corpus_seq_t* heldout,
                   const char* file_root);

void update_lda_seq_ss(int time,
                       const doc_t* doc,
                       const lda_post* post,
                       gsl_matrix** ss);

double fit_lda_seq_topics(lda_seq* model,
                          gsl_matrix** ss);


// === reading and writing ===


// read and write a lda sequence

void write_lda_seq(const lda_seq* m, const char* root);

lda_seq* read_lda_seq(const char* root, corpus_seq_t* data);

// write lda sequence sufficient statistics

void write_lda_seq_suffstats(lda_seq* m,
                             gsl_matrix** topic_ss,
                             const char* root);

// new lda sequence

lda_seq* new_lda_seq(corpus_seq_t* data,
		     int W,
		     int T,
		     int K);

void make_lda_from_seq_slice(lda* lda_m,
                             lda_seq* lda_seq_m,
                             int time);

#endif
