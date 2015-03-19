// Author: David Blei (blei@cs.princeton.edu)
//
// Copyright 2006 David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#include "params.h"

/*
 * check label
 *
 */

void check_label(FILE* f, char* name)
{
    char label[400];
    fscanf(f, "%s", label);
    assert(strcmp(label, name) == 0);
}


/*
 * read and write strings
 *
 */

void params_read_string(FILE* f, char* name, char* x)
{
    check_label(f, name);
    fscanf(f, "%s", x);
    outlog("%-10s READ NAME=%-10s STRING=%s", "[PARAMS]", name, x);
}

/*
 * read and write integers
 *
 */

void params_read_int(FILE* f, char* name, int* x)
{
    check_label(f, name);
    assert(fscanf(f, "%d", x) > 0);
    outlog("%-10s READ NAME=%-10s INT=%d", "[PARAMS]", name, *x);
}

void params_write_int(FILE* f, char* name, int x)
{
    fprintf(f, "%s %d\n", name, x);
}


/*
 * read and write doubles
 *
 */

void params_read_double(FILE* f, char* name, double* x)
{
    check_label(f, name);
    assert(fscanf(f, "%lf", x) > 0);
    outlog("%-10s READ NAME=%-10s DBL=%1.14e", "[PARAMS]", name, *x);
}

void params_write_double(FILE* f, char* name, double x)
{
    fprintf(f, "%s %17.14f\n", name, x);
}


/*
 * read and write gsl vectors and matrices.
 *
 */

void params_read_gsl_vector(FILE* f, char* name, gsl_vector** x)
{
    int size, i;
    double val;

    check_label(f, name);
    assert(fscanf(f, "%d", &size) > 0);
    *x = gsl_vector_calloc(size);
    for (i = 0; i < size; i++)
    {
        assert(fscanf(f, "%lf", &val) > 0);
        gsl_vector_set(*x, i, val);
    }
}


void params_write_gsl_vector(FILE* f, char* name, gsl_vector* x)
{
    fprintf(f, "%s %d", name, (int) x->size);
    int i;
    for (i = 0; i < x->size; i++)
        fprintf(f, " %17.14f", gsl_vector_get(x, i));
    fprintf(f, "\n");
}


//void params_write_doc_

void params_write_gsl_vector_multiline(FILE* f, char* name, gsl_vector* x)
{
    fprintf(f, "%s %d\n", name, (int) x->size);
    int i;
    if (x->size) {
        fprintf(f, "%17.14f", gsl_vector_get(x, 0));
    }
    for (i = 1; i < x->size; i++)
        fprintf(f, ",%17.14f", gsl_vector_get(x, i));
    fprintf(f, "\n");
}


void params_write_gsl_matrix(FILE* f, char* name, gsl_matrix* x)
{
    fprintf(f, "%s %ld %ld\n", name, x->size1, x->size2);
    int i, j;
    if (x->size1 == 0) {
      return;
    }
    for (i = 0; i < x->size1; i++) {
      fprintf(f, "%17.14f", gsl_matrix_get(x, i, 0));
      for (j = 1; j < x->size2; j++) {
        fprintf(f, ",%17.14f", gsl_matrix_get(x, i, j));
      }
      fprintf(f, "\n");
    }
}

void params_write_sparse_gsl_matrix(FILE* f, char* name, gsl_matrix* x)
{
    fprintf(f, "%s %ld %ld\n", name, x->size1, x->size2);
    int i, j;
    if (x->size1 == 0) {
      return;
    }
    for (i = 0; i < x->size1; i++) {
      for (j = 0; j < x->size2; j++) {
	//	outlog("%d %d %d %d", i, j, x->size1, x->size2);
	double value = gsl_matrix_get(x, i, j);
	if (fabs(value) > 1e-12) {
	  fprintf(f, "%d,%d,%17.14f\n", i, j, value);
	}
      }
    }
}
