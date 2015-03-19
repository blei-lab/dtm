#include "gflags.h"
#include "strutil.h"

#include "c2_lib.h"

#include <sys/stat.h>
#include <string.h>
#include <hash_set.h>
#include <hash_map.h>
#include <vector.h>
#include "sys/wait.h"
#include <stdlib.h>

DEFINE_bool(run_locally, false,
	    "If true, run locally.");
DEFINE_string(bin_directory, ".",
	      "The directory in which binaries live.");
DEFINE_bool(resume, false,
	    "If true, resume a halted job.");

namespace c2_lib {

bool FileRemove(string file_name) {
  return remove(file_name.c_str()) == 0;
}

string CurrentWorkingDirectory() {
  char* cwd = getcwd(NULL, 0);
  string result(cwd);
  free(cwd);
  return result;
}

bool FileExists(string file_name) {
  struct stat stFileInfo;
  bool blnReturn;
  int intStat;

  // Attempt to get the file attributes
  intStat = stat(file_name.c_str(), &stFileInfo);
  if(intStat == 0) {
    return(true);
  } else {
    return(false);
  }
}

void WaitOnTask(const Task& task) {
  bool done = false;
  double backoff(1.0);
  while (!done) {
    done = true;
    if (task.Status() != Task::SUCCESS) {
      done = false;
      if (task.Status() == Task::FAIL) {
	printf("Error.  Task failed. Debug: %s\n",
	       task.DebugString().c_str());
	exit(1);
      }
    }
    if (backoff > 2.0) {
      sleep(backoff);
    }
    backoff *= 1.1;
    if (backoff > 30) {
      backoff = 30;
    }
  }
}

void WaitOnTasks(const vector<Task*>& tasks) {
  double backoff(1.0);
  vector<Task*> pending_tasks = tasks;
  hash_map<long, int> task_tries;

  while (!pending_tasks.empty()) {
    for (vector<Task*>::iterator it=pending_tasks.begin();
	 it != pending_tasks.end();
	 ) {
      Task* task = *it;
      if (task->Status() != Task::SUCCESS) {
	if (task->Status() == Task::FAIL) {

	  if (task_tries.find((long) task) == task_tries.end()) {
	    task_tries[(long) task] = 1;
	  }

	  // Sometimes tasks fail for no good reason.
	  // Retry up to 3 times.
	  if (task_tries[(long) task] > 3) {
	    printf("Error.  Task %d failed.\n",
		   task->id);
	    printf("Debug string: %s\n",
		   task->DebugString().c_str());
	    exit(1);
	  } else {
	    printf("Task %d failed.  Retrying.  Try: %lld\n",
		   task->id,
		   task_tries[(long) task]);
	    printf("Debug string: %s\n",
		   task->DebugString().c_str());
	    
	    ++task_tries[(long) task];
	    task->Start();
	  }
	}
	// Don't break, since it's worth 
	// checking whether any tasks have failed
	// (in which case, we should die).
	++it;
      } else {
	printf("Removing task.\n");
	// Delete this element.
	it = pending_tasks.erase(it);
      }
    }
    printf("Waiting on %d tasks.\n",
	   pending_tasks.size());
    if (backoff > 2.0) {
      sleep(backoff);
    }
    backoff *= 1.1;
    if (backoff > 30) {
      backoff = 30;
    }
  }
}


Resource::Resource(string filename)
  : filename(filename) {
}

Resource::~Resource() {
  // We should not let this resource go out of scope if tasks are
  // still using it.
  if (references_.size() != 0) {
    printf("Error.  Task exists for deleted reference.");
    exit(1);
  }
    
}

bool Resource::Available() {
  return FileExists(filename);
}

void Resource::TaskDone(int id) {
  hash_set<int>::iterator it = references_.find(id);
  if (it != references_.end()) {
    references_.erase(it);
  }
}


// Task definitions.


string Task::working_directory_ = "";

void Task::AddResource(Resource* resource) {
  resources_.push_back(resource);
  resource->AddReference(id);
}

Task::Task(string name, int id)
  : command_(""),
    id(id),
    name(name),
    done_(false),
    resources_()
    {
}

Task::~Task() {
  for (vector<Resource*>::iterator it = resources_.begin();
       it != resources_.end();
       ++it) {
    (*it)->TaskDone(id);
  }
}

void Task::ResourcesAvailable_(bool* available,
			       string* resource_filename) {
  *available = true;
  for (vector<Resource*>::iterator it = resources_.begin();
       it != resources_.end() && *available;
       ++it) {
    if (!(*it)->Available()) {
      *available = false;
      *resource_filename = (*it)->filename;
    }
  }  
  *available = true;

  return;
}

}

namespace dtm {

using c2_lib::FileRemove;
using c2_lib::FileExists;
using c2_lib::CurrentWorkingDirectory;

string QSubTask::binary_ = "";

double ReadLikelihoods(const vector<string>& done_sentinels) {
  double l_hood = 0.0;
  for (vector<string>::const_iterator it = done_sentinels.begin();
       it != done_sentinels.end();
       ++it) {
    double l_hood_tmp;
    printf("Likelihood: %.2lf\n", l_hood);
    FILE* f = fopen((*it).c_str(), "r");
    if (f) {
      fscanf(f, "%lf", &l_hood_tmp);
      fclose(f);
    } else {
      printf("Error reading likelihood from file %s.\n",
	     (*it).c_str());
    }

    l_hood += l_hood_tmp;
    printf("Likelihood: %.2lf\n", l_hood);
  }
  return l_hood;
}


string QSubTask::DebugString() const {
  return StringPrintf("stderr: %s\n"
		      "stdout: %s\n"
		      "sentinel: %s\n"
		      "full commandline: %s\n",
		      stderr_filename_.c_str(),
		      stdout_filename_.c_str(),
		      done_sentinel_.c_str(),
		      full_commandline_.c_str());
}

static int vsystem(string command) {
  int pid;
  int return_code = 0;
  pid = vfork();
  if (pid == 0) {
    // We're the child process.  Execute.
    execl("/bin/sh",
	  "/bin/sh",
	  command.c_str(),
	  (char*) 0);
    _exit (EXIT_FAILURE);
  } else if (pid < 0) {
    // The fork failed.  Report failure.
    return_code = -1;
  } else {
    // This is the parent process.  Wait for the
    // child to complete.
    if (waitpid(pid, &return_code, 0) != pid) {
      return_code = -1;
    }
  }
  return return_code;
}

void QSubTask::Start() {
  string resource_filename;
  bool available;
  ResourcesAvailable_(&available,
		      &resource_filename);
  if (!available) {
    printf("Error.  Task resource unavailable:%s\n",
	   resource_filename.c_str());
  }

  if (done_sentinel_.empty()) {
    stdout_filename_ = StringPrintf(
	    "%s/%s_%d.out",
	    working_directory_.c_str(),
	    name.c_str(),
	    id);
    stderr_filename_ = StringPrintf(
	    "%s/%s_%d.err",
	    working_directory_.c_str(),
	    name.c_str(),
	    id);
    done_sentinel_ = StringPrintf(
	    "%s/%s_%d.done",
	    working_directory_.c_str(),
	    name.c_str(),
	    id);
  }

  if (FileExists(done_sentinel_)) {
    if (!FileRemove(done_sentinel_)) {
      printf("Error removing file.  Failing.\n");
      exit(1);
    }
  }
  if (FileExists(stdout_filename_)) {
    if (!FileRemove(stdout_filename_)) {
      printf("Error removing file.  Failing.\n");
      exit(1);
    }
  }
  if (FileExists(stderr_filename_)) {
    if (!FileRemove(stderr_filename_)) {
      printf("Error removing file.  Failing.\n");
      exit(1);
    }
  }

  string job_command = StringPrintf("%s/%s",
				    FLAGS_bin_directory.c_str(),
				    command_.c_str());
  const string kFilename = "/tmp/command_asdf.sh";
  if (FLAGS_run_locally) {
    full_commandline_ = StringPrintf("nohup ./%s",
				     job_command.c_str(),
				     binary_.c_str(),
				     stdout_filename_.c_str(),
				     stderr_filename_.c_str());
  } else {
    // TODO(sgerrish): Add proper temp file naming.
    full_commandline_ = StringPrintf(
      "echo \'cd %s ; %s ; \n \' "
      "| %s -o %s -e %s "
      "-l walltime=%d:00:00,mem=%dmb\n",
      CurrentWorkingDirectory().c_str(),
      job_command.c_str(),
      binary_.c_str(),
      stdout_filename_.c_str(),
      stderr_filename_.c_str(),
      walltime_hours_,
      memory_mb_);
  }
  printf("Current working directory: %s\n.", CurrentWorkingDirectory().c_str());
  FILE* f = fopen(kFilename.c_str(), "w");
  if (f) {
    fprintf(f, full_commandline_.c_str());
    fclose(f);
    chmod(kFilename.c_str(), 0744);
  } else {
    printf("Error opening file /tmp/command_asdf.txt\n  Skipping.");
  }

  printf("Running command: %s.\n", full_commandline_.c_str());
  int return_code = vsystem(kFilename.c_str());
  if (return_code) {
    printf("Return code %d when running command:\n"
	   "%s\n",
	   return_code,
	   full_commandline_.c_str());
    exit(1);
  }
}

Task::RunStatus QSubTask::Status() const {
  if (FileExists(done_sentinel_)) {
    return Task::SUCCESS;
  }

  if (!FileExists(stderr_filename_)) {
    return(Task::RUN);
  } else {
    if (!FileExists(done_sentinel_)) {
      return(Task::FAIL);
    }
  }
}

QSubTask* TaskFactory::NewTask(int id,
			       string job_name,
			       const char* root,
			       const hash_map<std::string, std::string>& flags,
			       const vector<Resource*>& resources,
			       const string& done_sentinel,
			       string binary) {
  QSubTask* task = new QSubTask(job_name, id);
  string flags_string;

  /*
  for (int i=0; i < resources.size(); ++i) {
    task->AddResource(resources[i]);
    if (i) {
      resource_string += " ";
    }
    resource_string += ("--"
			+ resources[i]->name
			+ "="
			+ resources[i]->filename);
  }
  */

  for (hash_map<string, string>::const_iterator it=flags.begin();
       it != flags.end();
       ++it) {
    if (it != flags.begin()) {
      flags_string += " ";
    }
    flags_string += ("--"
		     + it->first
		     + "="
		     + it->second);
  }

  task->set_done_sentinel(done_sentinel);
  task->set_working_directory(root);
  task->set_command(binary + " " + flags_string);
  if (flags.find("done_sentinel") != flags.end()) {
    //    task->set_done_sentinel(flags["done_sentinel"]);
    task->set_done_sentinel(flags.find("done_sentinel")->second);
  }

  // Set the parallel-processing binary appropriately.
  task->set_binary("/opt/torque/bin/qsub");

  return(task);
}

void CreateSentinel(const string& sentinel_filename,
		    double l_hood) {
  FILE* f = fopen(sentinel_filename.c_str(), "w");
  fprintf(f, "%lf\n", l_hood);
  fclose(f);
}

}  // namespace dtm
