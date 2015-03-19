#ifndef MATH_GRADIENTPROJECTION_INCLUDED
#define MATH_GRADIENTPROJECTION_INCLUDED

#define SAFETY_BOX 0.001
#define GRADIENT_DESCENT_SLOWDOWN 1.0

#include <iostream>

#include "gslwrap/vector_double.h"
#include "gslwrap/matrix_double.h"

namespace GradientProjection {
/*
 * Returns true if the sum to less than one constraint is violated,
 * fills in with the active constraint matrix.  Caller is responsible
 * for memory management of newly created matrix.
 */
bool createActiveConstraints(const gsl::vector& x, 
                             gsl::matrix& n,
                             gsl::vector& g);

void display(const gsl_vector* v, const char* name);

void display(const gsl_matrix* m, const char* name);

void createProjection(const gsl::matrix& activeConstraints,
                      const gsl::vector& g,
                      const gsl::vector& grad,
                      gsl::matrix& projection,
                      gsl::vector& direction,
                      gsl::vector& correction);

double updateState(gsl::vector& x,
                   const double gamma,
                   const gsl::vector grad,
                   const double f);
 
double descend(gsl::vector& x, 
               gsl::vector& s,
               const double gamma, 
               const double obj_value,             
               const gsl::vector& correction,
               const gsl::vector& grad);
 
}

#endif
