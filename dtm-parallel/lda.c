#include "gflags.h"

#include "c2_lib.h"
#include "data.h"
#include "lda.h"
#include "strutil.h"

#include <hash_map.h>
#include <vector.h>

DEFINE_bool(new_phi,
	    true,
	    "If true, use the new phi calculation.");

DEFINE_int32(nterms,
	     0,
	     "Number of terms.");
DEFINE_int32(ntopics,
	     0,
	     "Number of topics.");

DEFINE_int32(number_tasks,
	     5,
	     "Number of tasks.");

DEFINE_string(doc_boundaries,
	      "",
	      "Comma-separated list of integers describing epochs.");
DEFINE_int32(max_doc, 0, "");
DEFINE_int32(min_doc, 0, "");

DEFINE_string(topic_boundaries,
	      "",
	      "Comma-separated list of integers describing epochs.");
DEFINE_int32(max_topic, 0, "");
DEFINE_int32(min_topic, 0, "");

DEFINE_string(corpus_prefix,
	      "",
              "The function to perform. "
	      "Can be dtm or dim.");

DEFINE_string(root_directory,
	      "",
              "A directory in which all data is read and written.");

DEFINE_string(normalize_docs,
	      "normalize",
	      "How should we transform docs' word counts? "
	      "Options: take the log (log), normalize so they sum to 1 (normalize), "
	      "take the count (identity), set positive counts to 1 (occurrence).");

DEFINE_int32(lda_max_em_iter, -1, "");

DECLARE_int32(max_number_time_points);
DECLARE_string(model);
DECLARE_double(sigma_d);
DECLARE_double(sigma_l);
DEFINE_double(alpha, -10.0, "");

using c2_lib::Task;
using c2_lib::Resource;

namespace dtm {

const int LDA_INFERENCE_MAX_ITER=15;
const int LDA_INFERENCE_INIT_ITER=12;

// Used for the phi update: how small can the smallest phi be?
// We assert around 1e-10, since that is around the smallest beta
// for a term.
static const double kSmallLogNumber = -100.0;
static const double kSmallNumber = exp(kSmallLogNumber);
static double* scaled_influence = NULL;

/*
 * posterior inference for lda
 * time and doc_number are only necessary if
 * var is not NULL.
 */

static void SaveTopicsPartial(const char* name, lda* model) {
  gsl_matrix_view topics_view = gsl_matrix_submatrix(
    model->topics,
    0, 
    FLAGS_min_topic,
    FLAGS_nterms,
    FLAGS_max_topic - FLAGS_min_topic);
  mtx_fprintf(name, &topics_view.matrix);
}

double RunEStep() {
  double l_hood;

  // Load the topics and run inference on select documents.
  lda* model = new_lda_model(FLAGS_ntopics,
			     FLAGS_nterms);
  gsl_vector_set_all(model->alpha, FLAGS_alpha);

  corpus_t* data = read_corpus(FLAGS_corpus_prefix.c_str());

  read_lda_topics(model);
  lda_suff_stats* ss_docs = new_lda_suff_stats(model);
  
  l_hood = lda_e_step(model, data, ss_docs);

  // Write topic sufficient statistics for these docs (to be
  // aggregated later).
  string name = StringPrintf("%s/lda-ss-docs_%d_%d.dat",
			     FLAGS_root_directory.c_str(),
			     FLAGS_min_doc,
			     FLAGS_max_doc);
  write_lda_suff_stats(ss_docs, name.c_str());

  return l_hood;
}

double RunMStep() {
  double l_hood;
  lda* model = new_lda_model(FLAGS_ntopics,
			     FLAGS_nterms);
  gsl_vector_set_all(model->alpha, FLAGS_alpha);
  
  string name = StringPrintf("%s/initial-lda-ss.dat",
			     FLAGS_root_directory.c_str());
  outlog("Reading file %s.\n", name.c_str());
  lda_suff_stats* ss = read_lda_suff_stats(name.c_str(),
					   FLAGS_ntopics,
					   FLAGS_nterms);

  l_hood = lda_m_step(model, ss);

  outlog("Writing file %s.\n", name.c_str());
  name = StringPrintf("%s/lda-ss-topics_%d_%d.dat",
		      FLAGS_root_directory.c_str(),
		      FLAGS_min_topic,
		      FLAGS_max_topic);
  outlog("Writing file %s.\n", name.c_str());
  SaveTopicsPartial(name.c_str(), model);

  return l_hood;
}

void AggregateSuffStats() {
  string doc_boundaries_str = FLAGS_doc_boundaries;
  int nterms = FLAGS_nterms;
  int ntopics = FLAGS_ntopics;
  string name;

  vector<string> doc_boundaries;
  SplitStringUsing(doc_boundaries_str,
		   ",",
		   &doc_boundaries);
  gsl_matrix* ss = gsl_matrix_calloc(nterms,
				     ntopics);

  // TODO(sgerrish): Fix this part.
  for (int i=0; i + 1 < doc_boundaries.size(); ++i) {
    const string kMinDoc = doc_boundaries[i];
    const string kMaxDoc = doc_boundaries[i + 1];
    name = StringPrintf("%s/lda-ss-docs_%s_%s.dat",
			FLAGS_root_directory.c_str(),
			kMinDoc.c_str(),
			kMaxDoc.c_str());
    lda_suff_stats* ss_docs = read_lda_suff_stats(name.c_str(),
						  ntopics,
						  nterms);
    gsl_matrix_add(ss, ss_docs->topics_ss);
    gsl_matrix_free(ss_docs->topics_ss);
    delete ss_docs;
  }
  name = StringPrintf("%s/initial-lda-ss.dat",
		      FLAGS_root_directory.c_str());
  mtx_fprintf(name.c_str(), ss);

  gsl_matrix_free(ss);
}

static bool LDAConverged(double current, double last) {
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

static void RunMStepInParallel(int nterms,
			       double* l_hood,
			       string* topic_boundaries_str,
			       const vector<int>& topic_boundaries) {
  // Create a task and set up appropriate flags.
  hash_map<string, string> flags;
  flags["ntopics"] = StringPrintf("%d", FLAGS_ntopics);
  flags["root_directory"] = FLAGS_root_directory;
  flags["nterms"] = StringPrintf("%d", nterms);
  flags["corpus_prefix"] = FLAGS_corpus_prefix;

  *topic_boundaries_str = "0";
  for (int i=1; i < topic_boundaries.size(); ++i) {
    *topic_boundaries_str += StringPrintf(",%d", topic_boundaries[i]);
  }
  flags["topic_boundaries"] = *topic_boundaries_str;

  vector<string> done_sentinels;
  vector<Resource*> resources;
  vector<Task*> tasks;
  for (int i=0; i < FLAGS_number_tasks; ++i) {
    const int kMinTopic = topic_boundaries[i];
    const int kMaxTopic = topic_boundaries[i + 1];

    flags["min_topic"] = StringPrintf(
      "%d", kMinTopic);
    flags["max_topic"] = StringPrintf(
      "%d", kMaxTopic);
    flags["sentinel_filename"] = StringPrintf(
	  "%s/lda-topics_%d_%d.done",
	  FLAGS_root_directory.c_str(),
	  kMinTopic,
	  kMaxTopic);
    done_sentinels.push_back(flags["sentinel_filename"]);
  
    Task* task = TaskFactory::NewMStepTask(
      i,
      FLAGS_root_directory.c_str(),
      flags,
      resources,
      flags["sentinel_filename"]);
    tasks.push_back(task);
    
  }

  // Run.
  for (int t=0; t < tasks.size(); ++t) {
    //    printf("Starting task.\n");
    tasks[t]->Start();
  }

  c2_lib::WaitOnTasks(tasks);

  // Read the log likelihoods.
  *l_hood = ReadLikelihoods(done_sentinels);
}

static void RunEStepInParallel(int nterms,
			       int ndocs,
			       const string& topic_boundaries,
			       string* doc_boundaries_str,
			       double* l_hood) {

  hash_map<string, string> flags;
  //  flags["ndocs"] = StringPrintf("%d", ndocs);
  flags["alpha"] = StringPrintf("%f", FLAGS_alpha);

  flags["ntopics"] = StringPrintf("%d", FLAGS_ntopics);
  flags["topic_boundaries"] = topic_boundaries;
  flags["root_directory"] = FLAGS_root_directory;
  flags["nterms"] = StringPrintf("%d", nterms);
  flags["corpus_prefix"] = FLAGS_corpus_prefix;

  const int kDocsPerTask = ((ndocs
			     + FLAGS_number_tasks - 1)
			    / FLAGS_number_tasks);
  const int kNumberBigTasks = (FLAGS_number_tasks
			       - (kDocsPerTask * FLAGS_number_tasks
				  - ndocs));
  vector<int> doc_boundaries;
  doc_boundaries.push_back(0);
  *doc_boundaries_str = "0";
  int i=1;
  for (int t=kDocsPerTask;
       t < ndocs;
       ) {
    *doc_boundaries_str += StringPrintf(",%d", t);
    doc_boundaries.push_back(t);
    if (i < kNumberBigTasks) {
      t += kDocsPerTask;
    } else {
      t += kDocsPerTask - 1;
    }
    ++i;
  }
  doc_boundaries.push_back(ndocs);
  *doc_boundaries_str += StringPrintf(",%d", ndocs);
  flags["doc_boundaries"] = *doc_boundaries_str;

  vector<string> done_sentinels;
  vector<Resource*> resources;
  vector<Task*> tasks;
  for (int i=0; i < FLAGS_number_tasks; ++i) {
    const int kMinDoc = doc_boundaries[i];
    const int kMaxDoc = doc_boundaries[i + 1];
    
    flags["min_doc"] = StringPrintf(
      "%d", kMinDoc);
    flags["max_doc"] = StringPrintf(
      "%d", kMaxDoc);
    flags["sentinel_filename"] = StringPrintf(
	  "%s/lda-ss-docs_%d_%d.done",
	  FLAGS_root_directory.c_str(),
	  kMinDoc,
	  kMaxDoc);
    done_sentinels.push_back(flags["sentinel_filename"]);
  
    Task* task = TaskFactory::NewEStepTask(
      i,
      FLAGS_root_directory.c_str(),
      flags,
      resources,
      flags["sentinel_filename"]);
    tasks.push_back(task);
    
  }

  // Run.
  for (int t=0; t < tasks.size(); ++t) {
    tasks[t]->Start();
  }

  c2_lib::WaitOnTasks(tasks);

  // Read the log likelihoods.
  printf("Reading likelihoods.\n");
  *l_hood = ReadLikelihoods(done_sentinels);
  printf("Read likelihood %.2lf.\n", l_hood);
}

 
void AggregateSuffStatsInParallel(const string& doc_boundaries,
			     int nterms,
			     int ntopics) {
  printf("assip a\n");
  hash_map<string, string> flags;
  flags["ntopics"] = StringPrintf("%d", ntopics);
  flags["nterms"] = StringPrintf("%d", nterms);
  flags["doc_boundaries"] = doc_boundaries;
  flags["root_directory"] = FLAGS_root_directory;
  printf("assip c\n");
  flags["sentinel_filename"] = StringPrintf("%s/aggregate_suff_stats.done",
					    FLAGS_root_directory.c_str());
  
  vector<Resource*> resources;
  printf("assip d\n");
  Task* task = TaskFactory::NewTask(0,
				    "aggregate_suff_stats",
				    FLAGS_root_directory.c_str(),
				    flags,
				    resources,
				    flags["sentinel_filename"],
				    "main-aggregate-ss");
  task->Start();
  WaitOnTask(*task);
  delete task;
  printf("assip e\n");
}


bool RunParallelLDA(const vector<int>& topic_boundaries) {
  // !!!
  bool converged = false;

  corpus_t* initial_lda_data = read_corpus(
      FLAGS_corpus_prefix.c_str());
  const int kNTerms = initial_lda_data->nterms;

  lda* lda_model = new_lda_model(FLAGS_ntopics,
				 initial_lda_data->nterms);
  gsl_vector_set_all(lda_model->alpha, FLAGS_alpha);
  lda_suff_stats* lda_ss = new_lda_suff_stats(lda_model);

  initialize_lda_ss_from_random(lda_ss);

  string name = StringPrintf("%s/initial-lda-ss.dat",
			     FLAGS_root_directory.c_str());
  write_lda_suff_stats(lda_ss, name.c_str());

  int iterations = 0;

  if (FLAGS_number_tasks > initial_lda_data->ndocs) {
    FLAGS_number_tasks = initial_lda_data->ndocs;
  }
  if (FLAGS_number_tasks > FLAGS_ntopics) {
    FLAGS_number_tasks = FLAGS_ntopics;
  }

  double m_lhood, e_lhood, last = 1.0;
  string topic_boundaries_str;
  
  RunMStepInParallel(initial_lda_data->nterms,
		     &m_lhood,
		     &topic_boundaries_str,
		     topic_boundaries);

  while (!converged
	 && iterations++ < FLAGS_lda_max_em_iter) {

    string doc_boundaries;
    RunEStepInParallel(initial_lda_data->nterms,
		       initial_lda_data->ndocs,
		       topic_boundaries_str,
		       &doc_boundaries,
		       &e_lhood);

    // Aggregate sufficient statistics across all documents.
    AggregateSuffStatsInParallel(doc_boundaries,
				 initial_lda_data->nterms,
				 FLAGS_ntopics);
    RunMStepInParallel(initial_lda_data->nterms,
		       &m_lhood,
		       &topic_boundaries_str,
		       topic_boundaries);

    if (!(converged = LDAConverged(m_lhood + e_lhood,
				   last))) {
      last = m_lhood + e_lhood;
    }

    printf("LDA Log likelihood: %lf\n",
	   last);
  }
  
  free_corpus(initial_lda_data);

  return true;
}

double fit_lda_post(int doc_number, int time,
		    lda_post* p, lda_seq* var,
		    gsl_matrix* g,
		    gsl_matrix* g3_matrix,
		    gsl_matrix* g4_matrix,
		    gsl_matrix* g5_matrix) {
    init_lda_post(p);
    gsl_vector_view topic_view;
    gsl_vector_view renormalized_topic_view;
    if (var && var->influence) {
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
	} else {
	  update_phi(doc_number, time, p, var, g);
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
             vget(p->model->alpha, k) + ((double) p->doc->total) / K);
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
    for (n = 0; n < N; n++) {
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
	// double w_phi_sum = gsl_matrix_get(
	// var->topic[k]->w_phi_sum, w, time);

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

	// Note: we're ignoring term3.
	// sgerrish: 20 May 2010: Noticed that
	// term2 has a *negative* coefficient.  Changing it to
	// positive, since it appears that it should be positive.
	total = term1 + term2 - term3 + term4;
	mset(p->log_phi, n, k, total);
      }
      
      // Normalize in log space.
      gsl_vector log_phi_row = gsl_matrix_row(p->log_phi, n).vector;
      gsl_vector phi_row = gsl_matrix_row(p->phi, n).vector;
      log_normalize(&log_phi_row);
      
      bool set_min(false);
      for (i = 0; i < K; i++) {
	if (vget(&log_phi_row, i) < -16) {
	  vset(&log_phi_row, i, -16);
	  set_min = true;
	}
      }
      if (set_min) {
	log_normalize(&log_phi_row);
      }
            
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
      // outlog("doc weight size: %d", p->doc_weight->size);
      assert (K == p->doc_weight->size);
      double influence_topic = gsl_vector_get(p->doc_weight, k);
      if (FLAGS_model == "dim"
	  || FLAGS_model == "multiple"
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
      (vget(p->model->alpha, k) - vget(p->gamma, k)) * e_log_theta_k +
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

    for (d = FLAGS_min_doc; d < FLAGS_max_doc; ++d) {
      if (((d % 1000) == 0) && (d > 0)) {
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
  for (k = FLAGS_min_topic; k < FLAGS_max_topic; k++) {
    gsl_vector ss_k = gsl_matrix_column(ss->topics_ss, k).vector;
    gsl_vector log_p = gsl_matrix_column(model->topics, k).vector;
    if (LDA_USE_VAR_BAYES == 0) {
      gsl_blas_dcopy(&ss_k, &log_p);
      normalize(&log_p);
      vct_log(&log_p);

      // Add simple smoothing: anything less than -20 becomes -20.
      for (int i=0; i < log_p.size; ++i) {
	if (vget(&log_p, i) < -20.0) {
	  vset(&log_p, i, -20.0);
	}
      }
    } else {
      double digsum = (sum(&ss_k)
		       + model->nterms * LDA_TOPIC_DIR_PARAM);
      digsum = gsl_sf_psi(digsum);
      double param_sum = 0;
      for (w = 0; w < model->nterms; w++) {
	double param = vget(&ss_k, w) + LDA_TOPIC_DIR_PARAM;
	param_sum += param;
                double elogprob = gsl_sf_psi(param) - digsum;
                vset(&log_p, w, elogprob);
                lhood += ((LDA_TOPIC_DIR_PARAM - param) * elogprob
			  + gsl_sf_lngamma(param));
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

void write_lda_suff_stats(lda_suff_stats* ss, const char* name) {
    mtx_fprintf(name, ss->topics_ss);
}

void write_lda_topics(lda* model, const char* name) {
    mtx_fprintf(name, model->topics);
}

lda_suff_stats* read_lda_suff_stats(
  const char* filename,
  int ntopics, int nterms) {

  lda_suff_stats* ss = (lda_suff_stats*) malloc(sizeof(lda_suff_stats));
  ss->topics_ss = gsl_matrix_alloc(nterms, ntopics);
  mtx_fscanf(filename, ss->topics_ss);
  return(ss);
}

void read_lda_topics(lda* model) {
  vector<string> topic_boundaries;
  SplitStringUsing(FLAGS_topic_boundaries,
		   ",",
		   &topic_boundaries);

  string name;
  for (int i=0; i + 1 < topic_boundaries.size(); ++i) {
    int topic_min = atoi(topic_boundaries[i].c_str());
    int topic_max = atoi(topic_boundaries[i + 1].c_str());
    gsl_matrix_view topics_submatrix = gsl_matrix_submatrix(
        model->topics,
	0,
	topic_min,
	model->topics->size1,
	topic_max - topic_min);
    name = StringPrintf("%s/lda-ss-topics_%d_%d.dat",
			FLAGS_root_directory.c_str(),
			topic_min,
			topic_max);
    mtx_fscanf(name.c_str(), &topics_submatrix.matrix);
  }
}

void read_lda_topics(const vector<int>& topic_boundaries,
		     lda* model) {
  string name;
  for (int i=0; i + 1 < topic_boundaries.size(); ++i) {
    int topic_min = topic_boundaries[i];
    int topic_max = topic_boundaries[i + 1];
    gsl_matrix_view topics_submatrix = gsl_matrix_submatrix(
        model->topics,
	0,
	topic_min,
	model->topics->size1,
	topic_max - topic_min);
    name = StringPrintf("%s/lda-ss-topics_%d_%d.dat",
			FLAGS_root_directory.c_str(),
			topic_min,
			topic_max);
    mtx_fscanf(name.c_str(), &topics_submatrix.matrix);
  }
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

void initialize_lda_ss_from_random(lda_suff_stats* ss) {
    int k, n;
    gsl_rng * r = new_random_number_generator();
    for (k = 0; k < ss->topics_ss->size2; k++)
    {
        gsl_vector topic = gsl_matrix_column(ss->topics_ss, k).vector;
        gsl_vector_set_all(&topic, 0);
        for (n = 0; n < topic.size; n++)
        {
            vset(&topic, n, gsl_rng_uniform(r) + 10);
        }
    }
}


/*
 * initialize sufficient statistics from a document collection
 *
 */

void initialize_lda_ss_from_data(corpus_t* data,
				 lda_suff_stats* ss) {
    int k, n, i, w;
    gsl_rng* r = new_random_number_generator();

    for (k = 0; k < ss->topics_ss->size2; k++) {
        gsl_vector topic = gsl_matrix_column(ss->topics_ss, k).vector;
        for (n = 0; n < LDA_SEED_INIT; n++) {
            int d = floor(gsl_rng_uniform(r) * data->ndocs);
            doc_t* doc = data->doc[d];
            for (i = 0; i < doc->nterms; i++) {
                vinc(&topic, doc->word[n], doc->count[n]);
            }
        }
        for (w = 0; w < topic.size; w++) {
            vinc(&topic, w, LDA_INIT_SMOOTH + gsl_rng_uniform(r));
        }
    }
}

/*
 * write LDA model
 *
 */

void write_lda(lda* model, const char* name) {
  string filename = StringPrintf("%s.beta", name);
  mtx_fprintf(filename.c_str(), model->topics);
  filename = StringPrintf("%s.alpha", name);
  vct_fprintf(filename.c_str(), model->alpha);
}

/*
 * read LDA
 *
 */

lda* read_lda(int ntopics, int nterms, const char* name) {
  string filename;
  lda* model = new_lda_model(ntopics, nterms);
  filename = StringPrintf("%s.beta", name);
  mtx_fscanf(filename.c_str(), model->topics);
  filename = StringPrintf("%s.alpha", name);
  vct_fscanf(filename.c_str(), model->alpha);
  
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
    do {
        iter++;
        old_lhood = lhood;
        lhood = lda_e_step(model, data, ss) + m_lhood;
        m_lhood = lda_m_step(model, ss);
        converged = (old_lhood - lhood) / (old_lhood);
        outlog("iter   = %d", iter);
        outlog("lhood  = % 10.3f", lhood);
        outlog("conv   = % 5.3e\n", converged);
	outlog("max_iter: %d\n", max_iter);
    }
    while (((converged > LDA_EM_CONVERGED) || (iter <= 5))
	   && (iter < max_iter));
    write_lda(model, outname);
}

}  //  namespace dtm
