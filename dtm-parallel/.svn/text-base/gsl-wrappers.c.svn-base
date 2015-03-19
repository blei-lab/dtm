#include "gflags.h"
#include "gsl-wrappers.h"

static gsl_rng* RANDOM_NUMBER_GENERATOR = NULL;

DEFINE_int64(rng_seed,
	     0,
	     "Specifies the random seed.  If 0, seeds pseudo-randomly.");

// The maximum number of iterations for each update.
const double MAX_ITER = 15;

/*
 * safe logarithm function
 *
 */

double safe_log(double x)
{
    if (x == 0)
    {
        return(-1000);
    }
    else
    {
        return(log(x));
    }
}


/*
 * given log(a) and log(b), return log(a+b)
 *
 */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a == -1) return(log_b);

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}


void vinc(gsl_vector* v, int i, double x)
{
    vset(v, i, vget(v, i) + x);
}

void minc(gsl_matrix* m, int i, int j, double x)
{
    mset(m, i, j, mget(m, i, j) + x);
}


void msetrow(gsl_matrix* m, int r, const gsl_vector* val)
{
    int i;
    gsl_vector v = gsl_matrix_row(m, r).vector;
    for (i = 0; i < v.size; i++)
        vset(&v, i, vget(val, i));
}


void msetcol(gsl_matrix* m, int r, const gsl_vector* val)
{
    int i;
    gsl_vector v = gsl_matrix_column(m, r).vector;
    for (i = 0; i < v.size; i++)
        vset(&v, i, vget(val, i));
}


/*
 * compute the column sums of a matrix
 *
 */

void col_sum(gsl_matrix* m, gsl_vector* val)
{
    int i, j;
    gsl_vector_set_all(val, 0);

    for (i = 0; i < m->size1; i++)
        for (j = 0; j < m->size2; j++)
            vinc(val, j, mget(m, i, j));
}


/*
 * print a vector to standard out
 *
 */

void vct_printf(const gsl_vector * v)
{
    int i;
    for (i = 0; i < v->size; i++)
	printf("%5.5f ", vget(v, i));
    printf("\n\n");
}


/*
 * print a matrix to standard out
 *
 */

void mtx_printf(const gsl_matrix * m)
{
    int i, j;
    for (i = 0; i < m->size1; i++)
    {
	for (j = 0; j < m->size2; j++)
	    printf("%5.5f ", mget(m, i, j));
	printf("\n");
    }
}


/*
 * read/write a vector/matrix from a file
 *
 */

void vct_fscanf(const char* filename, gsl_vector* v)
{
    outlog("reading %ld vector from %s", v->size, filename);
    FILE* fileptr;
    if (!fileptr) {
      outlog("Error opening file %s. Failing.", filename);
      exit(1);
    }
    fileptr = fopen(filename, "r");
    gsl_vector_fscanf(fileptr, v);
    fclose(fileptr);
}

void mtx_fscanf(const char* filename, gsl_matrix * m)
{
    FILE* fileptr = fopen(filename, "r");

    outlog("reading %ld x %ld matrix from %s",
           m->size1, m->size2, filename);
    if (!fileptr) {
      outlog("Error opening file %s. Failing.", filename);
      exit(1);
    }

    gsl_matrix_fscanf(fileptr, m);
    fclose(fileptr);
}

void vct_fprintf(const char* filename, gsl_vector* v)
{
    outlog( "writing %ld vector to %s", v->size, filename);
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    if (!fileptr) {
      outlog("Error opening file %s. Failing.", filename);
      exit(1);
    }
    gsl_vector_fprintf(fileptr, v, "%20.17e");
    fclose(fileptr);
}


void mtx_fprintf(const char* filename, const gsl_matrix * m)
{
    outlog( "writing %ld x %ld matrix to %s",
            m->size1, m->size2, filename);
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    if (!fileptr) {
      outlog("Error opening file: %s", filename);
      exit(1);
    }
    gsl_matrix_fprintf(fileptr, m, "%20.17e");
    fclose(fileptr);
}


/*
 * matrix inversion using blas
 *
 */

void matrix_inverse(gsl_matrix* m, gsl_matrix* inverse)
{
    gsl_matrix *lu;
    gsl_permutation* p;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    gsl_linalg_LU_invert(lu, p, inverse);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);
}


/*
 * log determinant using blas
 *
 */

double log_det(gsl_matrix* m)
{
    gsl_matrix* lu;
    gsl_permutation* p;
    double result;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    result = gsl_linalg_LU_lndet(lu);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);

    return(result);
}


/*
 * eigenvalues of a symmetric matrix using blas
 *
 */

void sym_eigen(gsl_matrix* m, gsl_vector* vals, gsl_matrix* vects)
{
    gsl_eigen_symmv_workspace* wk;
    gsl_matrix* mcpy;
    int r;

    mcpy = gsl_matrix_alloc(m->size1, m->size2);
    wk = gsl_eigen_symmv_alloc(m->size1);
    gsl_matrix_memcpy(mcpy, m);
    r = gsl_eigen_symmv(mcpy, vals, vects, wk);
    gsl_eigen_symmv_free(wk);
    gsl_matrix_free(mcpy);
}


/*
 * sum of a vector
 *
 */

double sum(const gsl_vector* v)
{
    double val = 0;
    int i, size = v->size;
    for (i = 0; i < size; i++)
        val += vget(v, i);
    return(val);
}


/*
 * take log of each element
 *
 */

void vct_log(gsl_vector* v)
{
    int i, size = v->size;
    for (i = 0; i < size; i++)
        vset(v, i, safe_log(vget(v, i)));
}


/*
 * l2 norm of a vector
 *
 */

// !!! this can be BLASified

double norm(gsl_vector *v)
{
    double val = 0;
    int i;

    for (i = 0; i < v->size; i++)
        val += vget(v, i) * vget(v, i);
    return(sqrt(val));
}


/*
 * draw K random integers from 0..N-1
 *
 */

void choose_k_from_n(int k, int n, int* result)
{
    int i, x[n];

    if (RANDOM_NUMBER_GENERATOR == NULL)
        RANDOM_NUMBER_GENERATOR = gsl_rng_alloc(gsl_rng_taus);
    for (i = 0; i < n; i++)
        x[i] = i;

    gsl_ran_choose (RANDOM_NUMBER_GENERATOR, (void *) result,  k,
                    (void *) x, n, sizeof(int));
}


/*
 * normalize a vector in log space
 *
 * x_i = log(a_i)
 * v = log(a_1 + ... + a_k)
 * x_i = x_i - v
 *
 */

void log_normalize(gsl_vector* x)
{
    double v = vget(x, 0);
    int i;

    for (i = 1; i < x->size; i++)
        v = log_sum(v, vget(x, i));

    for (i = 0; i < x->size; i++)
        vset(x, i, vget(x,i)-v);
}


/*
 * normalize a positive vector
 *
 */

void normalize(gsl_vector* x)
{
    double v = 0;
    int i;

    for (i = 0; i < x->size; i++)
        v += vget(x, i);

    for (i = 0; i < x->size; i++)
        vset(x, i, vget(x, i) / v);
}


/*
 * exponentiate a vector
 *
 */

void vct_exp(gsl_vector* x)
{
    int i;

    for (i = 0; i < x->size; i++)
        vset(x, i, exp(vget(x, i)));
}


/*
 * maximize a function using its derivative
 *
 */

void optimize_fdf(int dim,
                  gsl_vector* x,
                  void* params,
                  void (*fdf)(const gsl_vector*, void*, double*, gsl_vector*),
                  void (*df)(const gsl_vector*, void*, gsl_vector*),
                  double (*f)(const gsl_vector*, void*),
                  double* f_val,
                  double* conv_val,
                  int* niter)
{
    gsl_multimin_function_fdf obj;
    obj.f = f;
    obj.df = df;
    obj.fdf = fdf;
    obj.n = dim;
    obj.params = params;

//    const gsl_multimin_fdfminimizer_type * method =
//        gsl_multimin_fdfminimizer_vector_bfgs;
    const gsl_multimin_fdfminimizer_type * method =
        gsl_multimin_fdfminimizer_conjugate_fr;

    gsl_multimin_fdfminimizer * opt =
        gsl_multimin_fdfminimizer_alloc(method, dim);

    gsl_multimin_fdfminimizer_set(opt, &obj, x, 0.01, 1e-3);

    int iter = 0, status;
    double converged, f_old = 0;
    do
    {
        iter++;
        status = gsl_multimin_fdfminimizer_iterate(opt);
        // assert(status==0);
        converged = fabs((f_old - opt->f) / (dim * f_old));
        // status = gsl_multimin_test_gradient(opt->gradient, 1e-3);
        // printf("f = %1.15e; conv = %5.3e; norm = %5.3e; niter = %03d\n",
        // opt->f, converged, norm(opt->gradient), iter);
        f_old = opt->f;
    }
    while (converged > 1e-8 && iter < MAX_ITER);
    // while (status == GSL_CONTINUE);
    *f_val = opt->f;
    *conv_val = converged;
    *niter = iter;
    gsl_multimin_fdfminimizer_free(opt);
}



/*
 * maximize a function
 *
 */

void optimize_f(int dim,
                gsl_vector* x,
                void* params,
                double (*f)(const gsl_vector*, void*))
{
    gsl_multimin_function obj;
    obj.f = f;
    obj.n = dim;
    obj.params = params;

    const gsl_multimin_fminimizer_type * method =
        gsl_multimin_fminimizer_nmsimplex;

    gsl_multimin_fminimizer * opt =
        gsl_multimin_fminimizer_alloc(method, dim);

    gsl_vector * step_size = gsl_vector_alloc(dim);
    gsl_vector_set_all(step_size, 1);
    gsl_multimin_fminimizer_set(opt, &obj, x, step_size);

    int iter = 0, status;
    double converged, f_old;
    do
    {
        iter++;
        f_old = opt->fval;
        status = gsl_multimin_fminimizer_iterate(opt);
        converged = fabs((f_old - opt->fval) / f_old);
        printf("f = %1.15e; conv = %5.3e; size = %5.3e; niter = %03d\n",
               opt->fval, converged, opt->size, iter);
    }
    while ((converged > 1e-10) || (iter < 10000));
    // while (status == GSL_CONTINUE);
    printf("f = %1.15e; conv = %5.3e; niter = %03d\n",
           opt->fval, converged, iter);

    gsl_multimin_fminimizer_free(opt);
    gsl_vector_free(step_size);
}


/*
 * check if a directory exists
 *
 * !!! shouldn't be here
 */

int directory_exist(const char *dname)
{
    struct stat st;
    int ret;

    if (stat(dname,&st) != 0)
    {
        return 0;
    }

    ret = S_ISDIR(st.st_mode);

    if(!ret)
    {
        errno = ENOTDIR;
    }

    return ret;
}

void make_directory(const char* name)
{
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}

gsl_rng* new_random_number_generator()
{
    gsl_rng* random_number_generator = gsl_rng_alloc(gsl_rng_taus);
    long t1;
    (void) time(&t1);

    if (FLAGS_rng_seed) {
      t1 = FLAGS_rng_seed;
    }

    // !!! DEBUG
    // t1 = 1147530551;
    printf("RANDOM SEED = %ld\n", t1);
    gsl_rng_set(random_number_generator, t1);
    return(random_number_generator);
}

