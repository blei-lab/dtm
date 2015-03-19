#include "logspace.h"

double log_dirichlet_likelihood(const double sum,
                                const double prior_sum,
                                const std::vector<int>& counts,
                                bool debug) {
  double val = 0.0;
  int length = counts.size();

  double prior_value = prior_sum / (double)length;
  val += gsl_sf_lngamma(prior_sum);
  val -= (double)length * gsl_sf_lngamma(prior_value);

  if(debug) cout << "Likelihood (" << sum << "," << prior_sum << "," << 
              prior_value << "," << length << ") = " << val << endl;

  for(int ii = 0; ii < length; ++ii) {

    if(debug) cout << "\tGAMMA(" << prior_value << " + " <<  
                (double)counts[ii] << " = " << prior_value + 
                (double)counts[ii] <<  ") -> " << val << endl;
    val += gsl_sf_lngamma(prior_value + (double)counts[ii]);
  }
  val -= gsl_sf_lngamma(prior_sum + sum);

  if(debug) cout << endl;

  return val;
}

double log_dirichlet_likelihood(const double sum,
                                const double prior_scale,
                                const gsl_vector* prior,
                                const std::vector<int>& counts) {
  double val = 0.0;
  int length = counts.size();

  val += gsl_sf_lngamma(prior_scale);
  for(int ii=0; ii < length; ++ii) {
    double prior_value = gsl_vector_get(prior, ii);
    val -= gsl_sf_lngamma(prior_value);
    val += gsl_sf_lngamma(prior_value + (double)counts[ii]);
  }
  val -= gsl_sf_lngamma(prior_scale + sum);

  return val;

}

double log_dot_product(const gsl_vector* log_a, const gsl_vector* log_b) {
  double sum = gsl_vector_get(log_a, 0) + gsl_vector_get(log_b, 0);
  assert(log_a->size == log_b->size);
  for (unsigned int ii = 1; ii < log_a->size; ++ii) {
    sum = log_sum(sum, gsl_vector_get(log_a, ii) +
		       gsl_vector_get(log_b, ii));
  }
  return sum;
}

double log_sum(const gsl_vector* x) {
  double sum = gsl_vector_get(x, 0);

  for (unsigned int ii = 1; ii < x->size; ii++) {
    sum = log_sum(sum, gsl_vector_get(x, ii));
  }
  return sum;
}

// Given a log vector, log a_i, compute log sum a_i.  Returns the sum.
double log_normalize(gsl_vector* x) {
  double sum = gsl_vector_get(x, 0);
  unsigned int i;

  for (i = 1; i < x->size; i++) {
    sum = log_sum(sum, gsl_vector_get(x, i));
  }

  for (i = 0; i < x->size; i++) {
    double val = gsl_vector_get(x, i);
    gsl_vector_set(x, i, val - sum);
  }
  return sum;
}

// Given a log matrix, log a_i, compute log sum a_i.  Returns the sum.
double log_normalize_matrix(gsl_matrix* x) {
  double sum = gsl_matrix_get(x, 0, 0);

  for (size_t ii = 0; ii < x->size1; ++ii) {
    for (size_t jj = 0; jj < x->size2; ++jj) {
      if (ii == 0 && jj == 0) {
	continue;
      }
      sum = log_sum(sum, gsl_matrix_get(x, ii, jj));      
    }
  }

  for (size_t ii = 0; ii < x->size1; ++ii) {
    for (size_t jj = 0; jj < x->size2; ++jj) {
      double val = gsl_matrix_get(x, ii, jj);
      gsl_matrix_set(x, ii, jj, val - sum);
    }
  }
  return sum;
}
