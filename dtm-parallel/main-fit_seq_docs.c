#include "gflags.h"
#include "lda-seq.h"
#include "main.h"
#include "c2_lib.h"

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

using namespace dtm;

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

  double lhood = fit_lda_seq_sd();

  CreateSentinel(FLAGS_sentinel_filename.c_str(),
		 lhood);
  printf("... Job complete.\n");
  
  return(0);
}
