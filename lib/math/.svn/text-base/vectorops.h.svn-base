#ifndef __MATH_VECTOROPS_INCLUDED
#define __MATH_VECTOROPS_INCLUDED

#include <cmath>
#include <limits>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
<<<<<<< vectorops.h
#include <gsl/gsl_cblas.h>
#include "math/specialfunc.h"
=======
#include <gsl/gsl_blas.h>
#include "specialfunc.h"
>>>>>>> 1.27

#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif

#ifndef isnan
# define isnan(x) ((x) != (x))
#endif

/* 
 * take the exponent of a vector
 *
 * If the exponent is infinite, then we replace the value with a
 * suitably large max_val
 */
void vexp(const gsl_vector* v, 
          gsl_vector* exp_v, 
          double max_val = std::numeric_limits<double>::infinity()) {
  assert(exp_v->size >= v->size);
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double val = exp(gsl_vector_get(v, ii));
    if (val == std::numeric_limits<double>::infinity() || val > max_val) {
      val = max_val;
    }
    gsl_vector_set(exp_v, ii, val);
  }
}

/* take the exponent of a matrix */
void mexp(const gsl_matrix* m, gsl_matrix* exp_m) {
  for (unsigned int ii = 0; ii < m->size1; ++ii) {
    for (unsigned int jj = 0; jj < m->size2; ++jj) {
      double val = exp(gsl_matrix_get(m, ii, jj));
      assert(!isnan(val));
      gsl_matrix_set(exp_m, ii, jj, val);
    }
  }
}

/* like vexp except that it also computes sum x log x */
double vexp_entropy(const gsl_vector* v, gsl_vector* exp_v) {
  double entropy = 0.0;
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double logval = gsl_vector_get(v, ii);
    double val = exp(logval);
    assert(!isnan(val));
    gsl_vector_set(exp_v, ii, val);
    if (val != 0) {
      entropy -= val * logval;
    }
  }
  return entropy;
}

double ventropy(const gsl_vector* v) {
  double entropy = 0.0;
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double val = gsl_vector_get(v, ii);
    if (val != 0) {
      entropy -= val * log(val);
    }
  }
  return entropy;
}

double lgamma(double x) {
    double x0,x2,xp,gl,gl0;
    int n=0,k=0;
    static double a[] = {
        8.333333333333333e-02,
       -2.777777777777778e-03,
        7.936507936507937e-04,
       -5.952380952380952e-04,
        8.417508417508418e-04,
       -1.917526917526918e-03,
        6.410256410256410e-03,
       -2.955065359477124e-02,
        1.796443723688307e-01,
       -1.39243221690590};
    
    x0 = x;
    if (x <= 0.0) return 1e308;
    else if ((x == 1.0) || (x == 2.0)) return 0.0;
    else if (x <= 7.0) {
        n = (int)(7-x);
        x0 = x+n;
    }
    x2 = 1.0/(x0*x0);
    xp = 2.0*M_PI;
    gl0 = a[9];
    for (k=8;k>=0;k--) {
        gl0 = gl0*x2 + a[k];
    }
    gl = gl0/x0+0.5*log(xp)+(x0-0.5)*log(x0)-x0;
    if (x <= 7.0) {
        for (k=1;k<=n;k++) {
            gl -= log(x0-1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

void mlog(const gsl_matrix* m, gsl_matrix* log_m) {
  for (unsigned int ii = 0; ii < m->size1; ++ii) {
    for (unsigned int jj = 0; jj < m->size2; ++jj) {
      gsl_matrix_set(log_m, ii, jj, log(gsl_matrix_get(m, ii, jj)));
    }
  }
}

void vlog(const gsl_vector* v, gsl_vector* log_v) {
  for (unsigned int ii = 0; ii < v->size; ++ii)
    gsl_vector_set(log_v, ii, log(gsl_vector_get(v, ii)));
}

void vlogit(const gsl_vector* v, gsl_vector* log_v) {
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double p = gsl_vector_get(v, ii);
    assert(p >= 0.0);
    assert(p <= 1.0);
    gsl_vector_set(log_v, ii, log(p / (1 - p)));
  }
}


void vsigmoid(const gsl_vector* v, gsl_vector* sig_v) {
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double p = gsl_vector_get(v, ii);
    gsl_vector_set(sig_v, ii, 1. / (1. + exp(-p)));
  }
}


double vlog_entropy(const gsl_vector* v, gsl_vector* log_v) {
  double entropy = 0;
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double val = gsl_vector_get(v, ii);
    entropy -= val * log(val);
    gsl_vector_set(log_v, ii, log(val));
  }
  return entropy;
}

double entropy(const gsl_vector* v) {
  double entropy = 0;
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    double val = gsl_vector_get(v, ii);
    entropy -= val * log(val);
  }
  return entropy;
}

void vdigamma(const gsl_vector* v, gsl_vector* digamma_v) {
  for (unsigned int ii = 0; ii < v->size; ++ii)
    // gsl_sf_psi throws an error when its argument is 0, whereas digamma returns inf.
    //    gsl_vector_set(digamma_v, ii, gsl_sf_psi(gsl_vector_get(v, ii)));
    gsl_vector_set(digamma_v, ii, digamma(gsl_vector_get(v, ii)));
}

void vlgamma(const gsl_vector* v, gsl_vector* lgamma_v) {
  for (unsigned int ii = 0; ii < v->size; ++ii)
    gsl_vector_set(lgamma_v, ii, lgamma(gsl_vector_get(v, ii)));
}

double gsl_blas_dsum(const gsl_vector* v) {
  double sum = 0;
  for (unsigned int ii = 0; ii < v->size; ++ii) {
    sum += gsl_vector_get(v, ii);
  }
  return sum;
}

double gsl_blas_dsum(const gsl_matrix* v) {
  double sum = 0;
  for (unsigned int ii = 0; ii < v->size1; ++ii) {
    for (unsigned int jj = 0; jj < v->size2; ++jj) {
      sum += gsl_matrix_get(v, ii, jj);
    }
  }
  return sum;
}

double gsl_matrix_rowsum(const gsl_matrix* m, const int row) {
  double sum = 0;
  for(unsigned int i=0; i < m->size2; i++) {
    sum += gsl_matrix_get(m, row, i);
  }
  return sum;
}

double dot_product(const gsl_vector* a, const gsl_vector* b) {
  assert(a->size == b->size);
  double val = 0;
  for(unsigned i=0; i<a->size; i++) {
    val += gsl_vector_get(a, i) * gsl_vector_get(b, i);
  }
  return val;
}

void uniform(gsl_vector* v) {
	gsl_vector_set_all(v, 1.0 / (double)v->size);
}

double normalize(gsl_vector* v) {
  double sum = gsl_blas_dsum(v);
  gsl_blas_dscal(1 / sum, v);
  return sum;
}

/*
  This function takes as input a multinomial parameter vector and
  computes the "total" variance, i.e., the sum of the diagonal of the
  covariance matrix.  

  If the multinomial parameter is unnormalized, then the variance of
  the normalized multinomial vector will be computed and then
  multiplied by the scale of the vector.
 */
double MultinomialTotalVariance(const gsl_vector* v) {
  double scale = gsl_blas_dsum(v);
  double variance = 0.0;
  for (size_t ii = 0; ii < v->size; ++ii) {
    double val = gsl_vector_get(v, ii) / scale;
    variance += val * (1. - val);
  }
  return variance * scale;
}

/*
  Computes covariance using the renormalization above and adds it to
  an existing matrix.
*/
void MultinomialCovariance(double alpha,
			   const gsl_vector* v, 
			   gsl_matrix* m) {
  double scale = gsl_blas_dsum(v);
  gsl_blas_dger(-alpha / scale, v, v, m);
  gsl_vector_view diag = gsl_matrix_diagonal(m);
  gsl_blas_daxpy(alpha, v, &diag.vector);
}

double MatrixProductSum(const gsl_matrix* m1,
			const gsl_matrix* m2) {
  double val = 0;
  assert(m1->size1 == m2->size1);
  assert(m1->size2 == m2->size2);
  for (size_t ii = 0; ii < m1->size1; ++ii) {
    for (size_t jj = 0; jj < m2->size2; ++jj) {
      val += gsl_matrix_get(m1, ii, jj) * 
	gsl_matrix_get(m2, ii, jj);
    }
  }
  return val;
}

double MatrixProductProductSum(const gsl_matrix* m1,
			       const gsl_matrix* m2,
			       const gsl_matrix* m3) {
  double val = 0;
  assert(m1->size1 == m2->size1);
  assert(m1->size2 == m2->size2);
  assert(m1->size1 == m3->size1);
  assert(m1->size2 == m3->size2);
  for (size_t ii = 0; ii < m1->size1; ++ii) {
    for (size_t jj = 0; jj < m2->size2; ++jj) {
      for (size_t kk = 0; kk < m3->size2; ++kk) {
	val += gsl_matrix_get(m1, ii, jj) * 
	  gsl_matrix_get(m2, ii, jj) *
	  gsl_matrix_get(m3, ii, jj);
      }
    }
  }
  return val;
}

double SumLGamma(const gsl_vector* v) {
  double s = 0.0;
  for (size_t ii = 0; ii < v->size; ++ii) {
    s += lgamma(gsl_vector_get(v, ii));
  }
  return s;
}

void mtx_fprintf(const char* filename, const gsl_matrix * m)
{
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_matrix_fprintf(fileptr, m, "%20.17e");
    fclose(fileptr);
}


void mtx_fscanf(const char* filename, gsl_matrix * m)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_matrix_fscanf(fileptr, m);
    fclose(fileptr);
}

double mtx_accum(const int i,
                 const int j,
                 const double contribution,
                 gsl_matrix* m) {

  double new_val = gsl_matrix_get(m, i, j) + contribution;
  gsl_matrix_set(m, i, j, new_val);
  return new_val;
}

void vct_fscanf(const char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_vector_fscanf(fileptr, v);
    fclose(fileptr);
}

void vct_fprintf(const char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_vector_fprintf(fileptr, v, "%20.17e");
    fclose(fileptr);
}

#endif
