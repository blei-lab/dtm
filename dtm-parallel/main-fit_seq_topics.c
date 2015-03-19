#include "gflags.h"

#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "lda-seq.h"
#include "lda.h"
#include "main.h"
#include <gsl/gsl_matrix.h>

DEFINE_string(sentinel_filename,
	      "",
	      "");

DECLARE_string(outname);
DECLARE_string(root_directory);
DECLARE_double(top_obs_var);
DECLARE_double(top_chain_var);
DECLARE_int32(ntopics);

DECLARE_string(corpus_prefix);
DECLARE_double(alpha);

// using namespace dtm;

/*
 * main function
 *
 * supports fitting a dynamic topic model
 *
 */

int main(int argc, char* argv[]) {
  // Initialize the flag objects.
  //    InitFlags(argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, 0);
  
  // mode for fitting a dynamic topic model
  double lhood = dtm::fit_lda_seq_st();
  dtm::CreateSentinel(FLAGS_sentinel_filename,
		      lhood);
  printf("... Job complete.\n");


  return(0);
}
