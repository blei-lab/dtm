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
#ifndef __random_generator_h
#define __random_generator_h

#include "gsl/gsl_rng.h"
#include <string>

namespace gsl
{
#ifndef __HP_aCC
using std::string;
#endif

//class RandomNumberGenerator 
class random_generator 
{
private:
	gsl_rng* generator;
public:
// Construction and Initializing:
	//! Default args reads environment variable GSL_RNG_TYPE and GSL_RNG_SEED to initialize. If these are not set the generator gsl_rng_mt19937 will be used with seed 0.
	random_generator (const random_generator& other) : generator(NULL) {generator = gsl_rng_clone(other.generator);}
	random_generator (const gsl_rng_type* type=NULL, unsigned long int seed=0) : generator(NULL)
	{
		gsl_rng_env_setup();
		if (!type)
		{
			generator = gsl_rng_alloc (gsl_rng_default);
		}
		else 
		{
			generator = gsl_rng_alloc (type) ; 
			if (seed)
				gsl_rng_set(generator, seed);
		}
	}
	~random_generator () {gsl_rng_free(generator);}
	random_generator& operator=(const random_generator& other){if (generator) gsl_rng_free(generator); generator = gsl_rng_clone(other.generator);return *this;}
	void set(unsigned long int seed){gsl_rng_set(generator, seed);}
	
// Sampling:
	unsigned long int get(unsigned long int n=0) {if (n) return gsl_rng_uniform_int(generator, n); else return gsl_rng_get(generator);} // returns value in range [min, max]
	double uniform() { return gsl_rng_uniform(generator);} // returns value in range [0, 1)
	double uniform_positive() { return gsl_rng_uniform_pos(generator);}// returns value in range (0, 1)
	unsigned long int uniform_int(unsigned long int n) 
		{ return gsl_rng_uniform_int(generator, n);}// returns value in range [0, n-1]

// Information:
	string name(){return gsl_rng_name(generator);}
	unsigned long int max(){return gsl_rng_max(generator);}
	unsigned long int min(){return gsl_rng_min(generator);}

// For calling gsl functions directly
	gsl_rng*       gslobj()       { return generator;}
	const gsl_rng* gslobj() const { return generator;}
//	static void Test();
};

}

#endif //__random_generator_h
