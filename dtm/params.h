// Author: David Blei (blei@cs.princeton.edu)
//
// Copyright 2006 David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#ifndef PARAMSH
#define PARAMSH

#define MAX_LINE_LENGTH 100000;

#include "gsl-wrappers.h"
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <string.h>

void params_read_string(FILE* f, char* name, char* x);

void params_read_int(FILE* f, char* name, int* x);

void params_write_int(FILE *, char *, int);

void params_read_double(FILE* f, char* name, double* x);

void params_write_double(FILE *, char *, double);

void params_read_gsl_vector(FILE* f, char* name, gsl_vector** x);

void params_write_gsl_vector(FILE *, char* , gsl_vector *);

void params_write_gsl_vector_multiline(FILE *, char* , gsl_vector *);

void params_write_gsl_matrix(FILE *, char* , gsl_matrix *);

void params_write_sparse_gsl_matrix(FILE *, char* , gsl_matrix *);

#endif
