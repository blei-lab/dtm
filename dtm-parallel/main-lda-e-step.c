#include "gflags.h"

#include "lda.h"
#include "main.h"
#include "c2_lib.h"

DEFINE_string(model,
	      "dim",
              "The function to perform. "
	      "Can be dtm or dim.");
DEFINE_string(sentinel_filename,
	      "",
              "A sentinel filename.");

DEFINE_string(outname, "", "");

using namespace dtm;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, 0);
  
  double l_hood = RunEStep();

  CreateSentinel(FLAGS_sentinel_filename,
		 l_hood);

  return(0);
}
