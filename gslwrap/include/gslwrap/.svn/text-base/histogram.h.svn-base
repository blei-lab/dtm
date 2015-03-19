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
#ifndef __histogram_h
#define __histogram_h

#include <gsl/gsl_histogram.h>
#include <stdexcept>
#include <exception>

namespace gsl{
#ifndef __HP_aCC
using std::string;
using std::runtime_error;
#endif

//! Encapsulates the histogram object of gsl. Only uniformly spaced bins yet.
class histogram
{
public:
	histogram(int nBins, double xmin, double xmax)
	{
		h=gsl_histogram_calloc(nBins);
		if (!h)
		{
			throw runtime_error("Couldn't allocate memory for histogram");
		}
		gsl_histogram_set_ranges_uniform(h, xmin, xmax);
	}
	~histogram(){gsl_histogram_free(h);}

//@{ Updating and Accessing Methods
	int increment(double x){return gsl_histogram_increment(h, x);}
	int accumulate(double x, double weight){return gsl_histogram_accumulate(h, x, weight);}
	double get(int i) const {return gsl_histogram_get(h, i);}
	double& operator[](const uint & i) 
	{
		const uint n = h->n;
		
		if (i >= n)
		{
			throw runtime_error("index lies outside valid range of 0 .. n - 1");
//			GSL_ERROR_VAL ("index lies outside valid range of 0 .. n - 1", GSL_EDOM, 0);
		}
		
		return h->bin[i];
	}
	const double& operator[](const uint & i) const //{return (*this)[i];/*gsl_histogram_get(h, i);*/}
	{
		const uint n = h->n;
		
		if (i >= n)
		{
			throw runtime_error("index lies outside valid range of 0 .. n - 1");
//			GSL_ERROR_VAL ("index lies outside valid range of 0 .. n - 1", GSL_EDOM, 0);
		}
		
		return h->bin[i];
	}

	void get_range(int i, double& xmin, double& xmax) const {gsl_histogram_get_range(h, i, &xmin, &xmax);}
//@}

//@{These functions return the maximum upper and minimum lower range limits
// and the number of bins of the histogram h. They provide a way of determining these values without
//    accessing the gsl_histogram struct directly. 
	double max() const {return gsl_histogram_max(h);}
	double min() const {return gsl_histogram_min(h);}
	int bins()const {return gsl_histogram_bins(h);}
	int size()const {return gsl_histogram_bins(h);}

//@}

//@{ Histogram statistics
	double mean()const {return gsl_histogram_mean(h);} // not in gsl library ?
	double max_val() const {return gsl_histogram_max_val(h);}
	int max_bin() const {return gsl_histogram_max_bin(h);}
	double min_val() const {return gsl_histogram_min_val(h);}
	int min_bin() const {return gsl_histogram_min_bin(h);}
	double sum() const {return gsl_histogram_sum(h);}
//@}


//@{ Accessor for gsl compatibility
	gsl_histogram*       gslobj()       { return h;}
	const gsl_histogram* gslobj() const { return h;}
//@}

protected:
	gsl_histogram * h;
};
}

#endif // __histogram_h
