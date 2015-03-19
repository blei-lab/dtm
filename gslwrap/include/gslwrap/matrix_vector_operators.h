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

#ifndef __matrix_vector_operators_h
#define __matrix_vector_operators_h

#include "gsl/gsl_blas.h"
#include <gslwrap/matrix_double.h>
#include <gslwrap/matrix_float.h>
#include <gslwrap/vector_double.h>

namespace gsl
{

inline
vector_float operator*(const matrix_float& m, const vector_float& v)
{
	vector_float y(m.get_rows());
	gsl_blas_sgemv(CblasNoTrans, 1.0, m.gslobj(), v.gslobj(), 0.0, y.gslobj());
	return y;
}

inline
vector operator*(const matrix& m, const vector& v)
{
	vector y(m.get_rows());
	gsl_blas_dgemv(CblasNoTrans, 1.0, m.gslobj(), v.gslobj(), 0.0, y.gslobj());
	return y;
}

}

#endif //__matrix_vector_operators_h
