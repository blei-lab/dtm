#include "gflags.h"

#include "lda.h"
#include "lda-seq.h"
#include "strutil.h"
#include "main.h"

#include <vector.h>

DECLARE_string(outname);
DECLARE_string(resume_stage);
DECLARE_int32(ntopics);
DECLARE_int32(number_tasks);

namespace dtm {

extern int LDA_INFERENCE_MAX_ITER;

static void CreateTopicBoundaries(int ntopics,
				  int number_tasks,
				  vector<int>* topic_boundaries) {
  const int kTopicsPerTask = ((ntopics
			       + number_tasks - 1)
			      / number_tasks);
  const int kNumberBigTasks = (number_tasks
			       - (kTopicsPerTask * number_tasks
				  - ntopics));
  // Be sure to clear topic boundaries since we run this
  // multiple times.
  topic_boundaries->clear();
  topic_boundaries->push_back(0);
  int i=1;
  for (int t=kTopicsPerTask;
       t < ntopics;
       ) {
    topic_boundaries->push_back(t);
    if (i < kNumberBigTasks) {
      t += kTopicsPerTask;
    } else {
      t += kTopicsPerTask - 1;
    }
    ++i;
  }
  topic_boundaries->push_back(ntopics);;  
}

static bool fit_dtm() {
  // make the directory for this fit
  string run_dir = StringPrintf("%s/", FLAGS_outname.c_str());
  if (!directory_exist(run_dir.c_str())) {
    make_directory(run_dir.c_str());
  }

  // Initialize (a few iterations of LDA fitting)
  outlog("%s","### INITIALIZING MODEL FROM LDA ###\n");
  bool success = true;

  vector<int> topic_boundaries;
  CreateTopicBoundaries(FLAGS_ntopics,
			FLAGS_number_tasks,
			&topic_boundaries);

  if (FLAGS_resume_stage != "doc"
      && FLAGS_resume_stage != "topic") {
    success = RunParallelLDA(topic_boundaries);
  }

  if (success) {
    printf("... Done");
  } else {
    printf("... Failed");
  }
  
  // !!! make this an option
  // Read in the corpus so we know how many terms.
  if (success) {
    success = FitParallelLDASeq(topic_boundaries);
  }

  return success;
}

} // namespace dtm

using namespace dtm;

int main(int argc, char* argv[])
{
  // Initialize the flag objects.
  //    InitFlags(argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, 0);
  
  // usage: main (sums corpus_sequence|fit param|time params)
  
  bool success = fit_dtm();

  return(!success);
}
