#include "strutil.h"

#include <string>
#include <iostream>
#include <vector>
#include <hash_set>
#include <hash_map>

using namespace std;

#ifndef c2_lib_h__
#define c2_lib_h__

namespace __gnu_cxx {
template<> struct hash< std::string >                                                       
{                                                                                           
  size_t operator()( const std::string& x ) const                                           
  {                                                                                         
    return hash< const char* >()( x.c_str() );                                              
  }                                                                                         
};                                                                                          
}


namespace c2_lib {

bool FileRemove(string file_name);
string CurrentWorkingDirectory();
bool FileExists(string file_name);

class Task;

// Block until all tasks in tasks are complete.
// If any task has failed, print a message and exit(1).;
void WaitOnTask(const Task& task);
void WaitOnTasks(const vector<Task*>& tasks);

// Generally corresponds to a single file.
class Resource: public FILE {
 public:
  Resource(string filename);

  ~Resource();

  // True iff the resource is available.
  virtual bool Available();

  // Removes a task from references_.
  void TaskDone(int id);

  void AddReference(int id) {
    references_.insert(id);
  }

  // The filename corresponding to this resource.
  string filename;

  // A name given to this resource.
  string name;

 private:
  std::hash_set<int> references_;
};


// Generally corresponds to one "parallel" task.
class Task {
 public:
  // Name can be anything.  Id should be unique.
  Task(string name, int id);

  ~Task();

  // Add a resource.
  virtual void AddResource(Resource* resource);

  // Launch the task.
  virtual void Start() = 0;

  virtual int Priority() const {
    return 1;
  }

  virtual string DebugString() const {
    return "";
  }

  // Removes resources from its tasks.
  enum RunStatus {
    RUN,
    SUCCESS,
    FAIL,
  };
  virtual RunStatus Status() const = 0;

  void set_command(string command) {
    command_ = command;
  }

  static void set_working_directory(string working_directory) {
    working_directory_ = working_directory;
  }

  // A numeric integer describing this job's id.  Should be unique at
  // any given time (could be recycled over the duration of the
  // program).
  int id;

  // A string describing this job.  Need not be unique.
  string name;

  // The working directory for parallel tasks.
  static string working_directory_;

 protected:
  // If the resources are available
  void ResourcesAvailable_(bool* available,
			   string* resource_filename);

  // The command which has been requested run.
  string command_;

 private:

  bool done_;

  vector<Resource*> resources_;

};

}  // namespace c2_lib

using namespace c2_lib;

namespace dtm {

// Helper functions
double ReadLikelihoods(
  const vector<string>& done_sentinels);

class QSubTask
  : public Task {
 public:
  QSubTask(string job_name, int id) : Task(job_name, id),
				      stdout_filename_(""),
				      stderr_filename_(""),
				      full_commandline_(""),
				      done_sentinel_(""),
				      walltime_hours_(13),
				      memory_mb_(15000) {}
  void Start();

  RunStatus Status() const;

  int Priority() const {
    return walltime_hours_;
  }

  string DebugString() const;

  static void set_binary(const string& binary) {
    binary_ = binary;
  }

  void set_done_sentinel(const string& sentinel) {
    done_sentinel_ = sentinel;
    stdout_filename_ = sentinel + ".out";
    stderr_filename_ = sentinel + ".err";
  }

  void set_memory_mb(int mb) {
    memory_mb_ = mb;
  }

  void set_walltime_hours(int hours) {
    walltime_hours_ = hours;
  }

private:

  // The parallel processing binary.
  static string binary_;

  string stdout_filename_;

  string stderr_filename_;

  string full_commandline_;

  int walltime_hours_;

  int memory_mb_;

  // The name of a small file which should exist iff the program has
  // completed successfully.
  string done_sentinel_;
};

class TaskFactory {
 public:
  TaskFactory() {}
  ~TaskFactory() {}

  static Task* NewEStepTask(int id,
			    const char* root,
			    const hash_map<std::string, std::string>& flags,
			    const vector<Resource*>& resources,
			    const string& done_sentinel) {
    QSubTask* task = NewTask(id, "lda-e-step",
		   root,
		   flags,
		   resources,
		   done_sentinel,
		   "lda-e-step");
    task->set_memory_mb(4000);
    task->set_walltime_hours(5);
    return task;
  }

  static Task* NewMStepTask(int id,
			    const char* root,
			    const hash_map<std::string, std::string>& flags,
			    const vector<Resource*>& resources,
			    const string& done_sentinel) {
    QSubTask* task = NewTask(id, "lda-m-step",
			     root,
			     flags,
			     resources,
			     done_sentinel,
			     "lda-m-step");
    task->set_memory_mb(4000);
    task->set_walltime_hours(3);
    return task;
  }

  static Task* NewTopicsFitTask(int id,
				const char* root,
				const hash_map<std::string, std::string>& flags,
				const vector<Resource*>& resources,
				const string& done_sentinel) {
    QSubTask* task = NewTask(id, "dtm-fit_seq_topics",
			     root,
			     flags,
			     resources,
			     done_sentinel,
			     "fit-seq-topics");
    task->set_memory_mb(8000);
    task->set_walltime_hours(10);
    return task;
  }

  static Task* NewDocsFitTask(int id,
			      const char* root,
			      const hash_map<std::string, std::string>& flags,
			      const vector<Resource*>& resources,
			      const string& done_sentinel) {
    QSubTask* task = NewTask(id, "dtm-fit_seq_docs",
			     root,
			     flags,
			     resources,
			     done_sentinel,
			     "fit-seq-docs");
    task->set_memory_mb(4000);
    task->set_walltime_hours(18);
    return task;
  }

  static QSubTask* NewTask(int id,
			   string job_name,
			   const char* root,
			   const hash_map<std::string, std::string>& flags,
			   const vector<Resource*>& resources,
			   const string& done_sentinel,
			   string binary);


 private:
};

void CreateSentinel(const string& sentinel_filename,
		    double l_hood);

}  // namespace dtm

#endif
