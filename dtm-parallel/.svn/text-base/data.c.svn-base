#include <string>

#include "gflags.h"
#include "params.h"
#include "strutil.h"
#include "ss-lm-fit_seq_topics.h"

#include "data.h"

DEFINE_double(sigma_l,
	      0.05,
	      "If true, use the new phi calculation.");
DEFINE_double(sigma_d,
	      0.05,
	      "If true, use the new phi calculation.");
DEFINE_double(sigma_c,
	      0.05,
	      "c stdev.");
DEFINE_double(resolution,
	      1,
	      "The resolution.  Used to determine how far out the beta mean should be.");
DEFINE_int32(max_number_time_points,
	     200,
	     "Used for the influence window.");
DEFINE_double(time_resolution,
	      0.5,
	      "This is the number of years per time slice.");
DEFINE_double(influence_mean_years,
	      20.0,
	      "How many years is the mean number of citations?");
DEFINE_double(influence_stdev_years,
	      15.0,
	      "How many years is the stdev number of citations?");
DEFINE_int32(influence_flat_years,
	      -1,
	     "How many years is the influence nonzero?"
	     "If nonpositive, a lognormal distribution is used.");
DEFINE_int32(fix_topics,
	     0,
	     "Fix all topics below this number.");

DECLARE_string(normalize_docs);

using namespace std;

namespace dtm {

/*
 * seq corpus range: [start, end)
 *
 * creates a subset of time slices
 *
 */

corpus_seq_t* make_corpus_seq_subset(corpus_seq_t* all,
				     int start,
				     int end) {
    int n;
    corpus_seq_t* subset_corpus = (corpus_seq_t*) malloc(sizeof(corpus_seq_t));
    subset_corpus->nterms = all->nterms;
    subset_corpus->len    = end - start;
    subset_corpus->ndocs  = 0;
    subset_corpus->corpus = (corpus_t**) malloc(sizeof(corpus_t*) * subset_corpus->len);
    for (n = start; n < end; n++)
    {
        subset_corpus->corpus[n - start] = all->corpus[n];
        subset_corpus->ndocs += all->corpus[n]->ndocs;
    }
    return(subset_corpus);
}


/*
 * collapse a sequential corpus to a flat corpus
 *
 */

corpus_t* collapse_corpus_seq(corpus_seq_t* c) {
    corpus_t* collapsed = (corpus_t*) malloc(sizeof(corpus_t));
    collapsed->ndocs  = c->ndocs;
    collapsed->nterms = c->nterms;
    collapsed->doc    = (doc_t**) malloc(sizeof(doc_t*) * c->ndocs);
    collapsed->max_unique = 0;
    int t, n, doc_idx = 0;
    for (t = 0; t < c->len; t++)
    {
        for (n = 0; n < c->corpus[t]->ndocs; n++)
        {
            collapsed->doc[doc_idx] = c->corpus[t]->doc[n];
            if (collapsed->doc[doc_idx]->nterms > collapsed->max_unique)
                collapsed->max_unique = collapsed->doc[doc_idx]->nterms;
            doc_idx++;
        }
    }
    assert(doc_idx == collapsed->ndocs);
    return(collapsed);
}

/*
 * read corpus
 *
 */

corpus_t* read_corpus(const char* name)
{
    int length, count, word, n;
    corpus_t* c;
    char filename[400];
    sprintf(filename, "%s-mult.dat", name);
    outlog("reading corpus from %s", filename);
    c = (corpus_t*) malloc(sizeof(corpus_t));
    c->max_unique = 0;
    FILE* fileptr = fopen(filename, "r");
    if (fileptr == NULL) {
      outlog("Error reading corpus prefix %s. Failing.",
	     filename);
      exit(1);
    }
    c->ndocs = 0; c->nterms = 0;
    c->doc = (doc_t**) malloc(sizeof(doc_t*));
    int grand_total = 0;
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
        if (length > c->max_unique) c->max_unique = length;
        c->doc = (doc_t**) realloc(c->doc, sizeof(doc_t*)*(c->ndocs+1));
        c->doc[c->ndocs] = (doc_t*) malloc(sizeof(doc_t));
        c->doc[c->ndocs]->nterms = length;
        c->doc[c->ndocs]->total = 0;
	c->doc[c->ndocs]->log_likelihood = 0.0;

        c->doc[c->ndocs]->word = (int*) malloc(sizeof(int)*length);
        c->doc[c->ndocs]->count = (int*) malloc(sizeof(int)*length);
        c->doc[c->ndocs]->lambda = (double*) malloc(sizeof(double)*length);
        c->doc[c->ndocs]->log_likelihoods = (double*) malloc(sizeof(double)*length);
        for (n = 0; n < length; n++)
        {
            fscanf(fileptr, "%10d:%10d", &word, &count);
            word = word - OFFSET;
	    if (FLAGS_normalize_docs == "occurrence") {
	      count = 1;
	    }
            c->doc[c->ndocs]->word[n] = word;
            c->doc[c->ndocs]->count[n] = count;
            c->doc[c->ndocs]->total += count;
	    // Is there a better value for initializing lambda?
	    c->doc[c->ndocs]->lambda[n] = 0.0;
	    c->doc[c->ndocs]->log_likelihoods[n] = 0.0;
            if (word >= c->nterms) { c->nterms = word + 1; }
        }
        grand_total += c->doc[c->ndocs]->total;
        c->ndocs = c->ndocs + 1;
    }
    fclose(fileptr);
    outlog("read corpus (ndocs = %d; nterms = %d; nwords = %d)\n",
           c->ndocs, c->nterms, grand_total);
    return(c);
}

void free_corpus(corpus_t* corpus) {
  for (int i=0; i < corpus->ndocs; ++i) {
    delete corpus->doc[i]->word;
    delete corpus->doc[i]->count;
    delete corpus->doc[i]->lambda;
    delete[] corpus->doc[i]->log_likelihoods;
  }
  delete corpus->doc;
  delete corpus;
}

/*
 * read corpus sequence
 *
 */

corpus_seq_t* read_corpus_seq(const char* name)
{
    char filename[400];
    corpus_seq_t* corpus_seq = (corpus_seq_t*) malloc(sizeof(corpus_seq_t));

    // read corpus
    corpus_t* raw_corpus = read_corpus(name);
    corpus_seq->nterms = raw_corpus->nterms;
    // read sequence information
    sprintf(filename, "%s-seq.dat", name);
    outlog("Reading corpus sequence %s.", filename);
    FILE* fileptr = fopen(filename, "r");
    if (!fileptr) {
      outlog("Error opening dtm sequence file %s.\n",
	     filename);
      exit(1);
    }
    fscanf(fileptr, "%d", &(corpus_seq->len));
    corpus_seq->corpus = (corpus_t**) malloc(sizeof(corpus_t*) * corpus_seq->len);
    // allocate corpora
    int doc_idx = 0;
    int ndocs, i, j;
    corpus_seq->ndocs = 0;
    for (i = 0; i < corpus_seq->len; ++i)
    {
        fscanf(fileptr, "%d", &ndocs);
        corpus_seq->ndocs += ndocs;
        corpus_seq->corpus[i] = (corpus_t*) malloc(sizeof(corpus_t));
        corpus_seq->corpus[i]->ndocs = ndocs;
        corpus_seq->corpus[i]->doc = (doc_t**) malloc(sizeof(doc_t*) * ndocs);
        for (j = 0; j < ndocs; j++)
        {
	  if (doc_idx >= raw_corpus->ndocs) {
	    outlog("Error: too few documents listed in dtm sequence file %s.\n"
		   "Current  line: %d %d %d.\n",
		   filename,
		   doc_idx,
		   ndocs,
		   j);
	    exit(1);
	  }
	  //	  outlog("%d %d %d %d\n", i, j, doc_idx, raw_corpus->ndocs);
	  corpus_seq->corpus[i]->doc[j] = raw_corpus->doc[doc_idx];
	  doc_idx++;
        }
    }
    corpus_seq->max_nterms = compute_max_nterms(corpus_seq);
    outlog("read corpus of length %d\n", corpus_seq->len);
    return(corpus_seq);
}


/*
 * write sequential corpus
 *
 */

void write_corpus_seq(corpus_seq_t* c, char* name)
{
    char tmp_string[400];
    int n;

    outlog("writing %d slices to %s (%d total docs)", c->len, name, c->ndocs);
    sprintf(tmp_string, "%s-seq.dat", name);
    FILE* seq_file = fopen(tmp_string, "w");
    fprintf(seq_file, "%d", c->len);
    for (n = 0; n < c->len; n++)
        fprintf(seq_file, " %d", c->corpus[n]->ndocs);
    fclose(seq_file);

    corpus_t* flat = collapse_corpus_seq(c);
    sprintf(tmp_string, "%s-mult.dat", name);
    write_corpus(flat, tmp_string);
}

/*
 * write corpus
 *
 */

void write_corpus(corpus_t* c, char* filename)
{
    int i, j;
    FILE * fileptr;
    doc_t * d;
    outlog("writing %d docs to %s\n", c->ndocs, filename);
    fileptr = fopen(filename, "w");
    for (i = 0; i < c->ndocs; i++)
    {
        d = c->doc[i];
        fprintf(fileptr, "%d", d->nterms);
        for (j = 0; j < d->nterms; j++)
        {
            fprintf(fileptr, " %d:%d", d->word[j], d->count[j]);
        }
        fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}

/*
 * read and write lda sequence variational distribution
 *
 */

void write_lda_seq_docs(const string& model_type,
			const string& root_directory,
			int min_time,
			int max_time,
			const lda_seq* model) {
  string filename = StringPrintf("%s/times_%d_%d_info.dat",
				 root_directory.c_str(),
				 min_time,
				 max_time);
  FILE* f = fopen(filename.c_str(), "w");

  params_write_int(f, "NUM_TOPICS", model->ntopics);
  params_write_int(f, "NUM_TERMS", model->nterms);
  params_write_int(f, "SEQ_LENGTH", model->nseq);
  params_write_gsl_vector(f, "ALPHA", model->alpha);
  
  fclose(f);

  for (int k = 0; k < model->ntopics; k++) {
    if (model_type == "fixed") {
      gsl_matrix_view view = gsl_matrix_submatrix(
        model->topic[k]->w_phi_l,
	0, min_time, model->nterms, max_time - min_time);
      filename = StringPrintf(
        "%s/w_phi_l-topic_%d-time_%d_%d.dat",
	root_directory.c_str(), k, min_time, max_time); 
      mtx_fprintf(filename.c_str(), &view.matrix);
      
      // Note that in this case, we write the entire sequence.
      view = gsl_matrix_submatrix(
        model->topic[k]->m_update_coeff,
	0, 0, model->nterms, max_time - min_time);
      filename = StringPrintf("%s/m_update_coeff-topic_%d-time_%d_%d.dat",
			      root_directory.c_str(),
			      k, min_time, max_time);
      mtx_fprintf(filename.c_str(), &view.matrix);
    }
  }

  for (int t=min_time; t < max_time; ++t) {
    outlog("\nwriting influence weights for time %d to %s",
	   t, filename.c_str());
    filename = StringPrintf(
      "%s/influence_time-%03d",
      root_directory.c_str(), t);
    gsl_matrix* influence_t =
      model->influence->doc_weights[t];
    assert(model->ntopics == influence_t->size2);
    mtx_fprintf(filename.c_str(), influence_t);

    filename = StringPrintf("%s/renormalized_influence_time-%03d",
			    root_directory.c_str(), t);
    outlog("\nwriting influence weights for time %d to %s",
	   t, filename.c_str());
    influence_t =
      model->influence->renormalized_doc_weights[t];
    assert(model->ntopics == influence_t->size2);
    mtx_fprintf(filename.c_str(), influence_t);
  }
}


/*
 * compute the maximum nterms in a corpus sequence
 *
 */

int compute_max_nterms(const corpus_seq_t* c)
{
    int i,j;
    int max = 0;
    for (i = 0; i < c->len; i++)
    {
        corpus_t* corpus = c->corpus[i];
        for (j = 0; j < corpus->ndocs; j++)
            if (corpus->doc[j]->nterms > max)
                max = corpus->doc[j]->nterms;
    }
    return(max);
}


/*
 * compute the total matrix of counts (W x T)
 *
 */

gsl_matrix* compute_total_counts(const corpus_seq_t* c)
{
    int t, d, n;
    gsl_matrix* ret = gsl_matrix_alloc(c->nterms, c->len);

    for (t = 0; t < c->len; t++)
    {
        corpus_t* corpus = c->corpus[t];
        for (d = 0; d < corpus->ndocs; d++)
        {
            doc_t* doc = corpus->doc[d];
            for (n = 0; n < doc->nterms; n++)
            {
                minc(ret, doc->word[n], t, (double) doc->count[n]);
            }
        }
    }
    return(ret);
}

// Creates a new array of doubles with kScaledBetaMax
// elements.
double* NewScaledInfluence(int size) {
  double* scaled_influence = new double[size];

  if (FLAGS_influence_flat_years > 0) {
    // Note that we round up, to make sure we have at least one epoch.
    int number_epochs = FLAGS_influence_flat_years * FLAGS_time_resolution;
    double epoch_weight = 1.0 / number_epochs;
    for (int i=0; i < number_epochs; ++i) {
      scaled_influence[i] = epoch_weight;
    }
    for (int i=number_epochs; i < size; ++i) {
      scaled_influence[i] = 0.0;
    }
    return scaled_influence;
  }


  /*
  // Use the simple distribution: 1 at [0], 0 everywhere else.
  for (int i=0; i < size; ++i) {
    scaled_influence[i] = 0.0;
  }
  scaled_influence[0] = 1.0;
  //  scaled_beta[1] = 0;
  return scaled_influence;
  */

  /*
  // Simulate a beta distribution with specified mean and variance.
  double total = 0.0;
  double tmp = (scaled_beta_mean
		* (1 - scaled_beta_mean)
		/ scaled_beta_variance) - 1.0;
  double beta_alpha = scaled_beta_mean * tmp;
  double beta_beta = (1 - scaled_beta_mean) * tmp;
  for (int i=0; i < scaled_beta_max; ++i) {
    // Offset tmp by 0.5 so we get a centered distribution
    // and don't run into degeneracy issues.
    tmp = (i + 0.5) / (scaled_beta_max);
    scaled_beta[i] = (pow(tmp, beta_alpha - 1.0)
		      * pow(1 - tmp, beta_beta - 1.0));
    total += scaled_beta[i];
  }
  */


  // Handle the log-normal distribution.
  double total = 0.0;

  // Here, we're interested more in the median.
  // So we treat the variable mean as median and note this in
  // our paper.
  double scaled_influence_mean = FLAGS_influence_mean_years;
  double scaled_influence_variance = (FLAGS_influence_stdev_years
				      * FLAGS_influence_stdev_years);
  double tmp = (1.0
		+ (scaled_influence_variance
		   / (scaled_influence_mean
		      * scaled_influence_mean)));
  double lognormal_sigma_squared = log(tmp);
  double lognormal_mu = (log(scaled_influence_mean)
			 - 0.5 * lognormal_sigma_squared);
  printf("Median: %.2f\n", exp(lognormal_mu));
  for (int i = 0; i < size; ++i) {
    // Shift right by half a timeframe to avoid corner cases.
    double x = (i / FLAGS_time_resolution) + (1.0 / FLAGS_time_resolution) / 2;
    double tmp2 = (log(x) - lognormal_mu);
    scaled_influence[i] = (1.0
		      / (x * sqrt(lognormal_sigma_squared * 2 * 3.1415926))
		      * exp(-tmp2 * tmp2
			    / (2.0
			       * lognormal_sigma_squared)));
    total += scaled_influence[i];
  }
  
  for (int i = 0; i < kScaledInfluenceMax; ++i) {
    scaled_influence[i] /= total;
  }

  return scaled_influence;
  
}

}  // namespace dtm
