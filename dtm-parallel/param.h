#include <stdio.h>
#include "util.h"

int    param_geti(const char *parameter_name, int default_value);

double param_getf(const char *parameter_name, double default_value);

char  *param_getc(const char *parameter_name, char *default_value);

char  *param_gets(const char *parameter_name);

int param_getb(const char *parameter_name, int default_value);
  /* Returns true if the value of <parameter_name> is 1, true, or yes,
     (case insensitive), false for any other value, and default_value
     for no value. */

int param_symvarie(const char *parameter_name, int *returned_value);
  /* Returns true if a value was found, false otherwise */

int param_symvarfe(const char *parameter_name, double *returned_value);
  /* Ditto */

int param_symvarce(const char *parameter_name, char *returned_value);
  /* Ditto. Note that the second argument is a "pointer to a char *",
     i.e., approximately a pointer to a string. */

void param_set(const char *parameter_name, char *new_value);
  /* Changes the value of ddinf parameter <parameter_name>. This can be
     used to communicate with subroutines which expect ddinf
     parameters without having to make sure they exist in the ddinf file.
     Note, however, that values assigned in the ddinf file are 
     OVERRIDDEN by a call to param_set. */
  /* One might want to implement a param_add which would allow adding
     new ddinf parameters within a program, but which could not
     override values from the ddinf file. */

/* if the following isn't called, param.c looks for a %restart 
binding in the param file */
void param_set_restart_file(const char *restart_name_p);

/* The following three calls write values to the restart file: */
void   param_puti(const char *parameter_name, int value);

void   param_putf(const char *parameter_name, double value);

void   param_putc(const char *parameter_name, char *value);


int    param_checkpointed(void);
  /* If there is a restart file, reads it in and returns TRUE. Otherwise
     returns false. */

void   param_checkpoint(void);
  /* Commits all of the param_put calls so far, are starts a new
     checkpoint. (I.e., subsequent `param_put's supersede earlier ones.) */


void  param_dump (FILE *stream);
  /* Writes the current ddinf bindings to a stream */

void  param_push_prefix (const char *hot_prefix);
  /* Push the current prefix to be applied to all ddnames */

void  param_pop_prefix (void);
  /* Pop the current prefix */

int param_push_file (const char *fn);
  /* Use the file for all bindings */

char *param_pop_file (void);
  /* Pop current bindings */







