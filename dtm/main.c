// Authors: David Blei (blei@cs.princeton.edu)
//          Sean Gerrish (sgerrish@cs.princeton.edu)
//
// Copyright 2011 Sean Gerrish and David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#include "gflags.h"

#include "main.h"

DEFINE_string(mode,
	      "fit",
              "The function to perform. "
	      "Can be fit, est, or time.");
DEFINE_string(model,
	      "dtm",
              "The function to perform. "
	      "Can be dtm or dim.");
DEFINE_string(corpus_prefix,
	      "",
              "The function to perform. "
	      "Can be dtm or dim.");
DEFINE_string(lda_model_prefix,
	      "",
              "The name of a fit model to be "
	      "used for testing likelihood.  Appending \"info.dat\" "
	      "to this should give the name of the file.");
DEFINE_int32(heldout_time,
	     -1,
	     "A time up to (but not including) which we wish to train, "
	     "and at which we wish to test.");
DEFINE_string(output_table, "", "");
DEFINE_string(params_file,
	      "settings.txt",
              "A file containing parameters for this run.");
DEFINE_bool(initialize_lda,
	    0,
	    "If true, initialize the model with lda.");

DEFINE_string(outname, "", "");
DEFINE_double(top_obs_var, 0.5, "");
DEFINE_double(top_chain_var, 0.005, "");
DEFINE_double(alpha, -10.0, "");
DEFINE_double(ntopics, -1.0, "");
DEFINE_int32(lda_max_em_iter, 20, "");
DEFINE_string(heldout_corpus_prefix, "", "");
DEFINE_int32(start, -1, "");
DEFINE_int32(end, -1, "");

extern int LDA_INFERENCE_MAX_ITER;

/*
 * read the parameters
 *
 * !!! use the cleaner functions in params.h
 *
 */


/*
 * fit a model from data
 *
 */

void fit_dtm(int min_time, int max_time)
{
    char name[400];

    // make the directory for this fit
    char run_dir[400];
    sprintf(run_dir, "%s/", FLAGS_outname.c_str());
    if (!directory_exist(run_dir)) {
      make_directory(run_dir);
    }

    // initialize (a few iterations of LDA fitting)
    outlog("%s","### INITIALIZING MODEL FROM LDA ###\n");

    printf("data file: %s\n", FLAGS_corpus_prefix.c_str());
    corpus_t* initial_lda_data = read_corpus(FLAGS_corpus_prefix.c_str());

    gsl_matrix* topics_ss;
    // !!! make this an option
    if (FLAGS_initialize_lda) {
      lda* lda_model = new_lda_model(FLAGS_ntopics, initial_lda_data->nterms);
      gsl_vector_set_all(lda_model->alpha, FLAGS_alpha);
      
      lda_suff_stats* lda_ss = new_lda_suff_stats(lda_model);
      // initialize_lda_ss_from_data(initial_lda_data, lda_ss);
      initialize_lda_ss_from_random(initial_lda_data, lda_ss);
      // sgerrish: Why do we only define the topics once?
      lda_m_step(lda_model, lda_ss);
      
      sprintf(name, "%s/initial-lda", run_dir);
      // TODO(sgerrish): Fix this.  This was originally hardcoded to 1.
      LDA_INFERENCE_MAX_ITER = 25;
      lda_em(lda_model, lda_ss, initial_lda_data, FLAGS_lda_max_em_iter, name);
      sprintf(name, "%s/initial-lda-ss.dat", run_dir);
      
      write_lda_suff_stats(lda_ss, name);
      topics_ss = lda_ss->topics_ss;
    } else {
      printf("loading %d terms..\n", initial_lda_data->nterms);
      topics_ss = gsl_matrix_calloc(initial_lda_data->nterms, FLAGS_ntopics);
      sprintf(name, "%s/initial-lda-ss.dat", FLAGS_outname.c_str());
      mtx_fscanf(name, topics_ss);
    }

    printf("fitting.. \n");
    // estimate dynamic topic model

    outlog("\n%s\n","### FITTING DYNAMIC TOPIC MODEL ###");

    corpus_seq_t* data_full = read_corpus_seq(FLAGS_corpus_prefix.c_str());

    corpus_seq_t* data_subset;
    if (max_time >= 0) {
      // We are training on a subset of the data.
      assert(max_time > min_time
	     && min_time >= 0
	     && max_time < data_full->len);
      data_subset = (corpus_seq_t*) malloc(sizeof(corpus_seq_t));
      data_subset->len = max_time - min_time + 1;
      data_subset->nterms = data_full->nterms;
      data_subset->corpus = (corpus_t**) malloc(
        sizeof(corpus_t*) * data_subset->len);
      int max_nterms = 0;
      int ndocs = 0;
      for (int i=min_time; i < max_time; ++i) {
	corpus_t* corpus = data_full->corpus[i];
	max_nterms = max_nterms > corpus->nterms ? max_nterms : corpus->nterms;
	data_subset->corpus[i - min_time] = corpus;
	ndocs += corpus->ndocs;
      }
      data_subset->max_nterms = max_nterms;
      data_subset->ndocs = ndocs;
    } else {
      // Use the entire dataset.
      data_subset = data_full;
    }
    
    lda_seq* model_seq = new_lda_seq(data_subset,
				     data_subset->nterms,
				     data_subset->len,
				     FLAGS_ntopics);
    init_lda_seq_from_ss(model_seq,
                         FLAGS_top_chain_var,
                         FLAGS_top_obs_var,
                         FLAGS_alpha,
                         topics_ss);

    fit_lda_seq(model_seq, data_subset, NULL, run_dir);

    if (max_time < 0) {
      return;
    }

    // Now find the posterior likelihood of the next time slice
    // using the most-recently-known time slice.
    lda* lda_model = new_lda_model(model_seq->ntopics, model_seq->nterms);
    make_lda_from_seq_slice(lda_model, model_seq, max_time - 1);

    lda_post post;
    int max_nterms = compute_max_nterms(data_full);
    post.phi = gsl_matrix_calloc(max_nterms, model_seq->ntopics);
    post.log_phi = gsl_matrix_calloc(max_nterms, model_seq->ntopics);
    post.gamma = gsl_vector_calloc(model_seq->ntopics);
    post.lhood = gsl_vector_calloc(model_seq->ntopics);
    post.model = lda_model;
    post.doc_weight = NULL;

    int d;
    double* table = (double*) malloc(sizeof(double) * data_full->corpus[max_time]->ndocs);

    for (d = 0; d < data_full->corpus[max_time]->ndocs; d++)
      {
	post.doc = data_full->corpus[max_time]->doc[d];
	table[d] = fit_lda_post(d, max_time, &post, NULL, NULL,
				NULL, NULL, NULL);
      }
    char tmp_string[400];
    sprintf(tmp_string, "%s-heldout_post_%d.dat", FLAGS_outname.c_str(),
	    max_time);
    FILE* post_file = fopen(tmp_string, "w");
    for (int d = 0; d < data_full->corpus[max_time]->ndocs; ++d)
      {
	fprintf(post_file, "%f\n", table[d]);
      }
}

/*
 * main function
 *
 * supports fitting a dynamic topic model
 *
 */

int main(int argc, char* argv[])
{
  // Initialize the flag objects.
  //    InitFlags(argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, 0);

    // usage: main (sums corpus_sequence|fit param|time params)

    // mode for spitting out document sums
    if (FLAGS_mode == "sums")
    {
        corpus_seq_t* c = read_corpus_seq(FLAGS_corpus_prefix.c_str());
	outlog("Tried to read corpus %s", FLAGS_corpus_prefix.c_str());
        int d, t;
        for (t = 0; t < c->len; t++)
        {
            int sum = 0;
            for (d = 0; d < c->corpus[t]->ndocs; d++)
            {
                sum += c->corpus[t]->doc[d]->total;
            }
            printf("%d\n\n", sum);
        }
    }

    // mode for fitting a dynamic topic model

    if (FLAGS_mode == "fit") {
      fit_dtm(0, FLAGS_heldout_time - 1);
    }

    // mode for analyzing documents through time according to a DTM

    if (FLAGS_mode == "time")
    {
        // read parameters

        // load corpus and model based on information from params

        corpus_seq_t* data = read_corpus_seq(FLAGS_heldout_corpus_prefix.c_str());
        lda_seq* model = read_lda_seq(FLAGS_lda_model_prefix.c_str(),
				      data);

        // initialize the table (D X OFFSETS)

        int d;
        double** table = (double**) malloc(sizeof(double*) * data->len);

        for (int t = 0; t < data->len; t++)
	{
  	    table[t] = (double*) malloc(sizeof(double) * data->corpus[t]->ndocs);
            for (d = 0; d < data->corpus[t]->ndocs; d++)
            {
	      table[t][d] = -1;  // this should be NAN
            }
        }

        // set up the LDA model to be populated

        lda* lda_model = new_lda_model(model->ntopics, model->nterms);

        lda_post post;
        int max_nterms = compute_max_nterms(data);
        post.phi = gsl_matrix_calloc(max_nterms, model->ntopics);
        post.log_phi = gsl_matrix_calloc(max_nterms, model->ntopics);
        post.gamma = gsl_vector_calloc(model->ntopics);
        post.lhood = gsl_vector_calloc(model->ntopics);
        post.model = lda_model;

        // compute likelihoods for each model

        for (int t = 0; t < data->len; t++) {
            make_lda_from_seq_slice(lda_model, model, t);
	    for (d = 0; d < data->corpus[t]->ndocs; d++) {
		post.doc = data->corpus[t]->doc[d];
		double likelihood = fit_lda_post(d, t, &post, model,
						 NULL,
						 NULL, NULL, NULL);
		table[t][d] = post.doc->log_likelihood;
	      }
	}
	char tmp_string[400];
	sprintf(tmp_string, "%s-heldout_post.dat", FLAGS_outname.c_str());
	FILE* post_file = fopen(tmp_string, "w");
	for (int t=0; t < data->len; ++t)
	{
	  if (data->corpus[t]->ndocs >= 0) {
	    fprintf(post_file, "%f", table[t][0]);
	  }
	  for (int d = 1; d < data->corpus[t]->ndocs; ++d)
          {
	    fprintf(post_file, ",%f", table[t][d]);
	  }
	  fprintf(post_file, "\n");
	}
        // !!! write out table
    }

    return(0);
}



