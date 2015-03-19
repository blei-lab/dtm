/*
 * Author: Jordan Boyd-Graber
 * Date: March 2008
 *
 * This file was spun off from logspace.h in order to create a
 * logspace file that wouldn't depend on gsl.
 */
#ifndef __MATH_LOGSPACE_BASE_H__
#define __MATH_LOGSPACE_BASE_H__

#include <iostream>
#include <limits>
#include <assert.h>
#include <math.h>

using namespace std;

#ifndef isnan
# define isnan(x) \
  (sizeof (x) == sizeof (long double) ? isnan_ld (x) \
  : sizeof (x) == sizeof (double) ? isnan_d (x) \
  : isnan_f (x))
static inline int isnan_f  (float       x) { return x != x; }
static inline int isnan_d  (double      x) { return x != x; }
static inline int isnan_ld (long double x) { return x != x; }
#endif

#ifndef isinf
# define isinf(x) \
  (sizeof (x) == sizeof (long double) ? isinf_ld (x) \
  : sizeof (x) == sizeof (double) ? isinf_d (x) \
  : isinf_f (x))
static inline int isinf_f  (float       x) { return isnan (x - x); }
static inline int isinf_d  (double      x) { return isnan (x - x); }
static inline int isinf_ld (long double x) { return isnan (x - x); }
#endif

double safe_log(double x);

// Given log(a) and log(b), return log(a + b).
double log_sum(double log_a, double log_b);

// Given log(a) and log(b), return log(a - b).
double log_diff(double log_a, double log_b);

/*
 * returns the element randomly sampled from the log
 * probabilities in array (number is the number of elements)
 */
int log_sample(double* vals, int length);

/*
 * Stupid "sampling" function for deterministic testing (i.e. in unit tests)
 */
int sample_first_nonzero(double* vals, int length);
int sample_max(double* vals);

bool is_nan(double val);

#endif  // __MATH_LOGSPACE_BASE_H__
