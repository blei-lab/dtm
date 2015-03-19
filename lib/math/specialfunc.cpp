#include "specialfunc.h"
#include <limits>
#include <gsl/gsl_sf_erf.h>

double trigamma(double x)
{
    double p;
    int i;

    x=x+6;
    p=1/(x*x);
    p=(((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
         *p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
    for (i=0; i<6 ;i++)
    {
        x=x-1;
        p=1/(x*x)+p;
    }
    return(p);
}


double digamma(double x) {
  if (x == 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
 
  double p;
  x=x+6;
  p=1/(x*x);
  p=(((0.004166666666667*p-0.003968253986254)*p+
      0.008333333333333)*p-0.083333333333333)*p;
  p=p+log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
  return p;
}

/*
  We invert the gamma by making a reasonable initial guess (typically
  this is correct to within a few percent).  An iteration of Newton's
  method is then used; this yields errors whose worst case are around
  .3% and typically around .01%.

  For small x, digamma is approximately -1/x and for large x it is
  approximately log(x).  Thus we make the initial guesses -1/x and
  exp(x) (with some fudge factors) depending on where x lies.
 */
double InverseDigamma(double x) {
  double guess = 0.0;
  if (x < -2) {
    guess = -1/x;
  } else {
    guess = exp(x) - 1 / (x + 7) + 0.5772157;  // Euler-Mascheroni constant.
  }
  guess -= (digamma(guess) - x) / trigamma(guess);
  return(guess);
}


double log_gamma(double x)
{
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

double sigmoid(double x) {
  return 1./(1 + exp(-x));
}

// First derivative of sigmoid function.
double dsigmoid(double x) {
  double s = sigmoid(x);
  return s * s * exp(-x);
}

// Second derivative of sigmoid function.
double d2sigmoid(double x) {
  double s = sigmoid(x);
  double ds = dsigmoid(x);
  return ds * (2 * s * exp(-x) - 1);
}

double LogPGaussian(double x) {
  // Phi(x) = 0.5 * erfc( - x / sqrt(2))
  // log Phi(x) = log(0.5) + log erfc( -x / sqrt(2))
  return log(0.5) + gsl_sf_log_erfc(-x / sqrt(2));
}

// d Phi(x) = 0.5 erfc'( - x / sqrt(2)) * (- 1 / sqrt(2))
// Note that d erfc = d (1 - erf) = - d erf = - 2 / sqrt(pi) exp(-x^2)
// => d Phi(x) = 0.5 * (-2 / sqrt(pi)) * exp(-x^2/2) * (-1 / sqrt(2))
//             = 1 / sqrt(2 pi) exp(-x^2 / 2)
double LogDGaussian(double x)  {
  return -x * x / 2 - 0.5 * log(2 * M_PI);
}

// Computes the inverse of PGaussian.  We use Newton iteration on the
// *log*.  This makes the iteration converge much better for small x.
// I dunno how well it will work for large values of x.
// 5 iterations seems to be enough.  It's still pretty slow so don't use 
// this in time-critical code.
double InversePGaussian(double x) {
  double y = 0;
  x = log(x);
  for (int ii = 0; ii < 5; ++ii) {
    double pgy = LogPGaussian(y);
    y -= (pgy - x) * exp(pgy) / exp(LogDGaussian(y));
  }
  return y;
}


