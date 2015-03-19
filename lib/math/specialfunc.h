#ifndef __MATH_SPECIALFUNC_H__
#define __MATH_SPECIALFUNC_H__
#include <cmath>

#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif

 /**
   * Proc to calculate the value of the trigamma, the second
   * derivative of the loggamma function. Accepts positive matrices.
   * From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
   * recurrence formula 6.4.6.  Each requires workspace at least 5
   * times the size of X.
   *
   **/

double trigamma(double x);


/*
 * taylor approximation of first derivative of the log gamma function
 *
 */
double digamma(double x);
double InverseDigamma(double x);


// lgamma.cpp -- log gamma function of real argument.
//      Algorithms and coefficient values from "Computation of Special
//      Functions", Zhang and Jin, John Wiley and Sons, 1996.
//
//  (C) 2003, C. Bond. All rights reserved.
//
//  Returns log(gamma) of real argument.
//  NOTE: Returns 1e308 if argument is 0 or negative.
//
double log_gamma(double x);

double sigmoid(double x);

// First derivative of sigmoid function.
double dsigmoid(double x);

// Second derivative of sigmoid function.
double d2sigmoid(double x);

// Log of the CDF of a Gaussian.  
double LogPGaussian(double x);

// Log of the PDF of a Gaussian.  
double LogDGaussian(double x);

// Computes the inverse of PGaussian.
double InversePGaussian(double x);

#endif
