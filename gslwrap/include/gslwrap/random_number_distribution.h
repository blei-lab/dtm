//  This matrix class is a C++ wrapper for the GNU Scientific Library
//  Copyright (C)  ULP-IPB Strasbourg

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

#ifndef __random_number_distribution_h
#define __random_number_distribution_h

#include "gslwrap/random_generator.h"
#include "gsl/gsl_randist.h"

namespace gsl
{

class random_number_distribution
{
 public:
	random_number_distribution(const random_generator& _generator) : generator(_generator){;}

	//Methods:
	virtual double get()=0;
	virtual double pdf(const double& x)=0;
	virtual ~random_number_distribution()
		{
			;
		}
 protected:
	random_generator generator;
};

class gaussian_random : public random_number_distribution
{
 public:
	gaussian_random(const random_generator& _generator, const double& _sigma=1.0) : random_number_distribution(_generator), sigma(_sigma){;}

	//methods:
	double get(){return gsl_ran_gaussian(generator.gslobj(), sigma);}
	double get(double _sigma){return gsl_ran_gaussian(generator.gslobj(), _sigma);}
	double pdf(const double& x){return gsl_ran_gaussian_pdf(x, sigma);}
	
	double ratio_method(){return gsl_ran_gaussian_ratio_method(generator.gslobj(), sigma);}
 protected:
	double sigma;
};

}

#endif //__random_number_distribution_h
