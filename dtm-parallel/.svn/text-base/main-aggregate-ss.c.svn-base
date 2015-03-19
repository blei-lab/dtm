#include "gflags.h"

#include "lda.h"
#include "main.h"
#include "c2_lib.h"

// Note that this is not necessary here.
DEFINE_string(model, "", "Model name.");

DEFINE_string(sentinel_filename,
	      "",
              "A sentinel filename.");

using namespace dtm;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, 0);
  
  AggregateSuffStats();
  CreateSentinel(FLAGS_sentinel_filename,
		 0.0);

  return(0);
}
