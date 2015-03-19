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

#ifndef _permutation_h
#define _permutation_h

#include<gsl/gsl_permutation.h>

namespace gsl
{
class permutation
{
	friend class matrix;
	friend class matrix_float;
	friend class matrix_int;

	gsl_permutation *gsldata;
public:
	permutation(size_t n,bool clear=true)
	{
		gsldata=(clear ? gsl_permutation_calloc(n) : gsl_permutation_alloc(n));
	}
	permutation():gsldata(NULL){;}
	void resize(size_t n){gsldata= gsl_permutation_calloc(n);}
};
}
#endif// _permutation_h
