#include "logspace_base.h"

using namespace std;

double safe_log(double x) {
  if (x <= 0) {
    return(-1e4);
  } else {
    return(log(x));
  }
}

// Given log(a) and log(b), return log(a + b).
double log_sum(double log_a, double log_b) {
  double v;

  if (log_a == -std::numeric_limits<double>::infinity() &&
      log_b == log_a) {
    return -std::numeric_limits<double>::infinity();
  } else if (log_a < log_b) {
    v = log_b + log(1 + exp(log_a - log_b));
  } else {
      v = log_a + log(1 + exp(log_b - log_a));
  }
  return(v);
}

// Given log(a) and log(b), return log(a - b).
double log_diff(double log_a, double log_b) {
  double val;
  double dangerous_part = exp(log_b - log_a);
  assert(dangerous_part < 1.0);
  val = log_a + log(1.0 - dangerous_part);
  return val;
}

/*
 * returns the element randomly sampled from the log
 * probabilities in array (number is the number of elements)
 */
int log_sample(double* vals, int length) {
  double normalizer = safe_log(0.0);
  int ii;
  for(ii=0; ii<length; ++ii) {
    normalizer = log_sum(normalizer, vals[ii]);
  }

  double val = 0, sum = 0, cutoff = (double)rand() / ((double)RAND_MAX + 1.0);
  for(ii=0; ii<length; ++ii) {
    val = exp(vals[ii] - normalizer);
    sum += val;
    if(sum >= cutoff)
      break;
  }
  assert(ii < length);
  return ii;
}

/*
 * A stupid "sampling" function for deterministic testing
 */
int sample_first_nonzero(double* vals, int length) {
  int ii;
  for(ii=0; ii < length - 1 && exp(vals[ii]) < 0.01; ++ii) { }
  return ii;
}

bool is_nan(double val) {
  return val != val;
}
