
#include "gradient_projection.h"

namespace GradientProjection {

void display(const gsl_vector* v, const char* name) {
  std::cout << name << " = <";
  for(unsigned int i=0; i<v->size; i++) {
    std::cout << gsl_vector_get(v, i);
    if (i < v->size - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ">    (" << v->size << ")" << std::endl;
}

void display(const gsl_matrix* m, const char* name) {
  std::cout << name << "\t = |";
  for(unsigned int i=0; i<m->size1; i++) {
    if(i!=0) {
      std::cout << "\t   |";
    }
    for(unsigned int j=0; j<m->size2; j++) {
      std::cout << gsl_matrix_get(m, i, j) << "\t";
    }
    std::cout << "|" << std::endl;
  }
  std::cout << "                                SIZE: " << m->size1 << " x " << m->size2 << std::endl;
}

void createProjection(const gsl::matrix& activeConstraints,
                      const gsl::vector& g,
                      const gsl::vector& grad,
                      gsl::matrix& projection,
                      gsl::vector& direction,
                      gsl::vector& correction) {
  int n = activeConstraints.size1();
  int r = activeConstraints.size2();

  correction.resize(n);
  direction.resize(n);

  // This could be done with cholesky or QR decomposition, but I
  // couldn't get it to work.  Given that this happens infrequently
  // and the matrices are not *that* big, it's not that bad
  gsl::matrix S(r,r);
  // S = N^T N
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, 
                 activeConstraints.gslobj(), activeConstraints.gslobj(), 
                 0.0, S.gslobj());
  // T = (N^{T} N) ^{-1}
  gsl::matrix T = S.LU_invert();
  S.set_dimensions(n, r);
  // S = -N(N^{T} N)^{-1}
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, 
                 activeConstraints.gslobj(), T.gslobj(), 0.0, S.gslobj());

  // Set the correction
  gsl_blas_dgemv(CblasNoTrans, 1.0, S.gslobj(), g.gslobj(), 0.0, 
                 correction.gslobj());

  // Set the direction
  // P = -N(N^{T} N)^{-1}N + I 
  projection.identity(n);
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, 
                 S.gslobj(), activeConstraints.gslobj(), 
                 1.0, projection.gslobj());
  gsl_blas_dgemv(CblasNoTrans, -1.0, projection.gslobj(), grad.gslobj(), 0.0, 
                 direction.gslobj());
}

bool createActiveConstraints(const gsl::vector& x, 
                             gsl::matrix& n, 
                             gsl::vector& g) {
  bool sumConstraintViolated = false;
  int dimension = x.size();
  double margin = SAFETY_BOX;

  if(x.sum() >= 1.0 - margin) {
    sumConstraintViolated = true;
  }

  int nonNegativeConstraintsViolated = 0;
  for(int ii = 0; ii < dimension; ++ii) {
    if (x[ii] <= margin) {
      ++nonNegativeConstraintsViolated;
    }
  }

  int newSize = nonNegativeConstraintsViolated;
  if(sumConstraintViolated) {
    newSize += 1;
  }

  if(newSize > 0) {
    n.set_dimensions(dimension, newSize);
    g.resize(newSize);
    g.set_all(SAFETY_BOX);
    int col = 0;
    if(sumConstraintViolated) {
      g[0] = -(1.0 - SAFETY_BOX);
      for(int ii = 0; ii < dimension; ++ii) {
        n(ii, 0) = -1.0;
      }
      ++col;
    }

    for(int ii = 0; ii < dimension; ++ii) {
      if(x[ii] <= margin) {
        n(ii, col) = 1.0;
        ++col;
      }
    }
    assert(col == newSize);

    gsl_blas_dgemv(CblasTrans, 1.0, n.gslobj(), x.gslobj(), -1.0, g.gslobj());

    //display(n.gslobj(), "N");
    //display(g.gslobj(), "g");

    return true;
  } else {
    return false;
  }
}

double descend(gsl::vector& x, 
               gsl::vector& s,
               const double gamma, 
               const double obj_value,             
               const gsl::vector& correction,
               const gsl::vector& grad) {
  double alpha = 0.0;

  gsl_blas_ddot(s.gslobj(), grad.gslobj(), &alpha);
  //std::cout << "dot prod= " << alpha << " ";
  if(alpha == 0) {
    return alpha;
  }
  alpha = -(gamma * obj_value) / alpha;
  //std::cout << " alpha= " << alpha << " ";

  s *= alpha;
  s += correction;
  x += s;

  //display(s.gslobj(), "final move");

  if(alpha < 0) {
    alpha = -alpha;
  }
  return alpha;
}

double updateState(gsl::vector& x,
                   const double gamma,
                   const gsl::vector grad,
                   const double f) {
  /*
   * First we see if we're up against constraints
   */
  int dim = x.size();
  gsl::matrix n;
  gsl::vector g;
  gsl::vector s;
  gsl::vector correction(dim);
  
  if(createActiveConstraints(x, n, g)) {
    s.resize(dim);
    gsl::matrix p;
    
    createProjection(n, g, grad, p, s, correction);
    //std::cout << "Constraints violated." << std::endl;

    //display(p.gslobj(), "p");
    //display(s.gslobj(), "s");
    //display(correction.gslobj(), "correction");
    return descend(x, s, gamma, f, correction, grad);
  } else {
    //std::cout << "No constraints violated." << std::endl;
    s.copy(grad);
    s *= -gamma * GRADIENT_DESCENT_SLOWDOWN;
    x += s;
    double diff;
    gsl_blas_ddot(s.gslobj(), s.gslobj(), &diff);
    return diff * gamma;
  }


}


}
