//  This random generator is a C++ wrapper for the GNU Scientific Library
//  Copyright (C) 2001 Torbjorn Vik

//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#ifndef __min_fminimizer_h
#define __min_fminimizer_h 

#include <gsl/gsl_errno.h>
#include <gsl/gsl_min.h>

namespace gsl{

//! Derive this class provide a user defined function for minimisation
struct min_f
{
	//! This operator must be overridden
	virtual double operator()(const double& x)=0;
	
	//! This is the function gsl calls to optimize f
	static double f(double x, void *p)
	{
		return (*(min_f *)p)(x);
	}
};

//! Class for minimizing one dimensional functions. 
/*!
  Usage: 
       - Create with optional minimize type
	   - Set with function object and inital bounds
	   - Loop the  iterate function until convergence or maxIterations (extra facility)

	   - Recover minimum and bounds
 */
class min_fminimizer 
{
 public:
	//! choose between gsl_min_fminimizer_goldensection and gsl_min_fminimizer_brent
	min_fminimizer(const gsl_min_fminimizer_type* type=gsl_min_fminimizer_brent) : s(NULL), maxIterations(100), isSet(false)
	{
		s=gsl_min_fminimizer_alloc(type);
		nIterations=0;
		if (!s)
		{
			//error
			//cout << "ERROR Couldn't allocate memory for minimizer" << endl;
			//throw ? 
			exit(-1);
		}
	}
	~min_fminimizer(){if (s) gsl_min_fminimizer_free(s);}
	//! returns GSL_FAILURE if the interval does not contain a minimum
	int set(min_f& function, double minimum, double x_lower, double x_upper)
	{
		isSet=false;
		f.function = &function.f;
		f.params = &function;
		int status=	gsl_min_fminimizer_set(s, &f, minimum, x_lower, x_upper);
		if (!status)
		{
			isSet=true;
			nIterations=0;
		}
		return status;
	}
	int set_with_values(min_f& function, 
						double minimum, double f_minimum, 
						double x_lower,double f_lower, 
						double x_upper, double f_upper)
	{
		isSet=false;
		f.function = &function.f;
		f.params = &function;
		int status=	gsl_min_fminimizer_set_with_values(s, &f, minimum, f_minimum, x_lower, f_lower, x_upper, f_upper);
		if (!status)
		{
			isSet=true;
			nIterations=0;
		}
		return status;
	}
	int iterate()
	{
		assert_set();
		int status=gsl_min_fminimizer_iterate(s);
		nIterations++;
		if (status==GSL_FAILURE)
			isConverged=true;
		return status;
	}
	double minimum(){assert_set();return gsl_min_fminimizer_minimum(s);}
	double x_upper(){assert_set();return gsl_min_fminimizer_x_upper(s);}
	double x_lower(){assert_set();return gsl_min_fminimizer_x_lower(s);}
	void SetMaxIterations(int n){maxIterations=n;}
	int GetNIterations(){return nIterations;}
	bool is_converged(){if (nIterations>=maxIterations) return true; if (isConverged) return true; return false;}
	//string name() const;
	
 private:
	void assert_set(){if (!isSet)exit(-1);} // Old problem of error handling: TODO
	
	bool isSet;
	bool isConverged;
	int nIterations;
	int maxIterations;
	gsl_min_fminimizer* s;
	gsl_function f;
};
};	 // namespace gsl

#endif //__min_fminimizer_h
