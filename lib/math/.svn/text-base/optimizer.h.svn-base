#ifndef __LIB_MATH_OPTIMIZER__
#define __LIB_MATH_OPTIMIZER__

#include <iostream>
#include <limits>
#include <gsl/gsl_multimin.h>
#include "math/gsl_vector.h"
#include "util/flags.h"

using std::cout;
using std::endl;

DEFINE_double(multimin_convergence_threshold,
	      1e-5,
	      "Convergence threshold for conjugate gradient.");
DEFINE_size(max_multimin_iterations,
	    40,
	    "Maximum number of conjugate gradient iterations to perform.");

class Optimizer {
 public:
  Optimizer(size_t size) : size_(size) {  
  }
  
  void Optimize() {
    gsl_multimin_function_fdf my_func;
    my_func.n = size_;
    my_func.f = &MultiminObjectiveWrapper;
    my_func.df = &MultiminGradientWrapper;
    my_func.fdf = &MultiminObjectiveGradientWrapper;
    my_func.params = this;
    
    gsl_multimin_fdfminimizer* s =  
      gsl_multimin_fdfminimizer_alloc(gsl_multimin_fdfminimizer_conjugate_fr, size_);
    GslVector initial_guess(size_);
    MultiminInitialGuess(initial_guess.mutable_ptr());

    // step_size, tol
    //    gsl_multimin_fdfminimizer_set(s, &my_func, initial_guess.ptr(), 0.1, 1.0);
    gsl_multimin_fdfminimizer_set(s, &my_func, initial_guess.ptr(), 0.01, 0.01);
    
    size_t iter = 0;
    int status;
    
    double value = std::numeric_limits<double>::infinity();
    double prev_value;
    do {
      prev_value = value;
      iter++;
      status = gsl_multimin_fdfminimizer_iterate(s);
      if (status) {
	cout << "Error: " << gsl_strerror(status) << endl;
	break;
      }
      status = gsl_multimin_test_gradient(s->gradient, 1e-3);	
      if (status == GSL_SUCCESS) {
	cout << "Minimum found." << endl;
      }
      value = s->f;
      cout << "Iteration: " << iter << " Value: " << 
	value << " dValue:" << (prev_value - value)/fabs(value) << " " <<
	gsl_strerror(status) << endl;
    } while (status == GSL_CONTINUE &&
	       iter < FLAGS_max_multimin_iterations &&
	     (prev_value - value) / fabs(value) > FLAGS_multimin_convergence_threshold);    
    MultiminResult(s->x);
    gsl_multimin_fdfminimizer_free(s);
  }

  virtual void MultiminObjectiveGradient(const gsl_vector* x, 
					 double* objective, 
					 gsl_vector* gradient) = 0;

  virtual void MultiminInitialGuess(gsl_vector* v) = 0;
  
  virtual void MultiminResult(gsl_vector* x) = 0;

  virtual ~Optimizer() { }
 protected:
  static double MultiminObjectiveWrapper(const gsl_vector* x, void* params) {
    double objective;
    reinterpret_cast<Optimizer*>(params)->MultiminObjectiveGradient(x, &objective, NULL);
    return objective;
  }
  
  static void MultiminGradientWrapper(const gsl_vector* x, void* params, gsl_vector* g) {
    reinterpret_cast<Optimizer*>(params)->MultiminObjectiveGradient(x, NULL, g);
  }
  
  static void MultiminObjectiveGradientWrapper(const gsl_vector* x,
					       void* params,
					       double* f,
					       gsl_vector* g) {
    reinterpret_cast<Optimizer*>(params)->MultiminObjectiveGradient(x, f, g);
  }
  
 private:
  size_t size_;
};
#endif  // __LIB_MATH_OPTIMIZER__
