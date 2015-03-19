#ifndef GSL_WRAPPERS_H
#define GSL_WRAPPERS_H

// #include <gsl/gsl_check_range.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define outlog(format, args...) \
    fprintf(stderr, format, args); \
    fprintf(stderr, "\n");

double safe_log(double);
double log_sum(double, double);

static inline double vget(const gsl_vector* v, int i)
{ return(gsl_vector_get(v, i)); };

static inline void vset(gsl_vector* v, int i, double x)
{ gsl_vector_set(v, i, x); };

// Increment a vector element by a double.
void vinc(gsl_vector*, int, double);

static inline double mget(const gsl_matrix* m, int i, int j)
{ return(gsl_matrix_get(m, i, j)); };

static inline void mset(gsl_matrix* m, int i, int j, double x)
{ gsl_matrix_set(m, i, j, x); };

void msetcol(gsl_matrix* m, int r, const gsl_vector* val);

// Increment a matrix element by a double.
void minc(gsl_matrix*, int, int, double);
void msetrow(gsl_matrix*, int, const gsl_vector*);

void col_sum(gsl_matrix*, gsl_vector*);

void vct_printf(const gsl_vector* v);
void mtx_printf(const gsl_matrix* m);
void vct_fscanf(const char*, gsl_vector* v);
void mtx_fscanf(const char*, gsl_matrix* m);
void vct_fprintf(const char* filename, gsl_vector* v);
void mtx_fprintf(const char* filename, const gsl_matrix* m);

double log_det(gsl_matrix*);

void matrix_inverse(gsl_matrix*, gsl_matrix*);

void sym_eigen(gsl_matrix*, gsl_vector*, gsl_matrix*);

double sum(const gsl_vector* v);

double norm(gsl_vector * v);

void vct_log(gsl_vector* v);
void vct_exp(gsl_vector* x);

void choose_k_from_n(int k, int n, int* result);

void log_normalize(gsl_vector* x);
void normalize(gsl_vector* x);

void optimize(int dim,
              gsl_vector* x,
              void* params,
              void (*fdf)(const gsl_vector*, void*, double*, gsl_vector*),
              void (*df)(const gsl_vector*, void*, gsl_vector*),
              double (*f)(const gsl_vector*, void*));

void optimize_fdf(int dim,
                  gsl_vector* x,
                  void* params,
                  void (*fdf)(const gsl_vector*, void*, double*, gsl_vector*),
                  void (*df)(const gsl_vector*, void*, gsl_vector*),
                  double (*f)(const gsl_vector*, void*),
                  double* f_val,
                  double* conv_val,
                  int* niter);

void log_write(FILE* f, char* string);
int directory_exist(const char *dname);
void make_directory(char* name);

gsl_rng* new_random_number_generator();

#endif
