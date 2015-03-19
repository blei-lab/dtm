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
#ifndef __multimin_fdfminimizer_h
#define __multimin_fdfminimizer_h 

#include <gsl/gsl_errno.h>
#include <gsl/gsl_multimin.h>
#include <gslwrap/vector_double.h>

namespace gsl{

//! Create an instance of this class with a user defined function
/*!
  A template class with the function operator()(const vector& x) 
  and derivative(const vector&, vector&), as well as a reference to an object of this class must be fournished

  User is responsible for deleting this reference !

 */
template <class fdf_function>
class multimin_fdf
{
 public:
	fdf_function* fct;

	//! These operators can be overridden
	virtual double operator()(const vector& x)
		{
			return (*fct)(x);
		}
	virtual void derivative(const vector& x, vector& g)
		{
			(*fct).derivative(x, g);
		}
	
	//! This operator can be overridden to gain performance in calculating the value and its derivative in a scoop
	virtual double fval_and_derivative(const vector&x, vector& g )
	{
		derivative(x, g);
		return (*this)(x);
	}


	//! This is the function gsl calls to calculate the value of f at x
	static double f(const gsl_vector* x, void *p)
	{
		vector_view x_view(*x);
		return (*(multimin_fdf *)p)(x_view);
	}

	//! This is the function gsl calls to calculate the value of g=f' at x
	static void df(const gsl_vector* x, void *p, gsl_vector* g)
	{
		vector_view x_view(*x);
		vector_view g_view(*g);
		(*(multimin_fdf *)p).derivative(x_view, g_view);
	}

	//! This is the function gsl calls to calculate the value of g=f' at x
	static void fdf(const gsl_vector* x, void *p, double* f, gsl_vector* g)
	{
		vector_view x_view(*x);
		vector_view g_view(*g);
		*f=(*(multimin_fdf *)p).fval_and_derivative(x_view, g_view);
	}

	//! Constructor (User is responsible for deleting the fdf_function object)
	multimin_fdf(fdf_function* _fct):fct(_fct){assert (fct!=NULL);}
};

//! Class for multiminimizing one dimensional functions. 
/*!
  Usage: 
       - Create with optional multiminimize type
	   - Set with function object and inital bounds
	   - Loop the  iterate function until convergence or maxIterations (extra facility)

	   - Recover multiminimum and bounds
 */
class multimin_fdfminimizer 
{
 public:
//! 
/*! Choose between : 
  - gsl_multimin_fdfminimizer_conjugate_fr
  - gsl_multimin_fdfminimizer_conjugate_pr
  - gsl_multimin_fdfminimizer_vector_bfgs
  - gsl_multimin_fdfminimizer_steepest_descent
  
 */
	multimin_fdfminimizer(uint _dim, 
						  const gsl_multimin_fdfminimizer_type* type=gsl_multimin_fdfminimizer_conjugate_fr) : 
		dim(_dim), isSet(false), maxIterations(100), s(NULL)
	{
		s=gsl_multimin_fdfminimizer_alloc(type, dim);
		nIterations=0;
		if (!s)
		{
			//error
			//cout << "ERROR Couldn't allocate memory for multiminimizer" << endl;
			//throw ? 
			exit(-1);
		}
	}
	~multimin_fdfminimizer(){if (s) gsl_multimin_fdfminimizer_free(s);}
	//! returns GSL_FAILURE if the interval does not contain a multiminimum
	template <class  fdf_function>
	int set(multimin_fdf<fdf_function>& function, const vector& initial_x, double step_size, double tol)
	{
		isSet=false;
		f.f   = &function.f;
		f.df  = &function.df;
		f.fdf = &function.fdf;
		f.n   = dim;
		f.params = &function;
		int status=	gsl_multimin_fdfminimizer_set(s, &f, initial_x.gslobj(), step_size, tol);
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
		int status=gsl_multimin_fdfminimizer_iterate(s);
		nIterations++;
		if (status==GSL_FAILURE)
			isConverged=true;
		return status;
	}
	int restart(){return gsl_multimin_fdfminimizer_restart(s);}
  	double minimum(){assert_set();return gsl_multimin_fdfminimizer_minimum(s);} 
	vector x_value(){assert_set();return vector_view(*gsl_multimin_fdfminimizer_x(s));}  
	vector gradient(){assert_set();return vector_view(*gsl_multimin_fdfminimizer_gradient(s));}  


	void SetMaxIterations(int n){maxIterations=n;}
	int GetNIterations(){return nIterations;}
	bool is_converged(){if (nIterations>=maxIterations) return true; if (isConverged) return true; return false;}
	//string name() const;
	
 private:
	void assert_set(){if (!isSet)exit(-1);} // Old problem of error handling: TODO
	
	uint dim;
	bool isSet;
	bool isConverged;
	int nIterations;
	int maxIterations;
	gsl_multimin_fdfminimizer* s;
	gsl_multimin_function_fdf f;
};
};	 // namespace gsl

#endif //__multimin_fdfminimizer_h
