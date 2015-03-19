#ifndef _UTIL_INCLUDED
#define _UTIL_INCLUDED 1

#include <stdarg.h>

#define EOS  '\0'
#define CRLF  printf("\n")
#define TRUE  1
#define FALSE 0

extern const char*  quote (const char *s);
extern char*  dequote (char *s);
extern void   quote_no_matter_what (const char *s, char *t);
extern int    verify (char *s, char *t);
extern char*  strip (char *s);
extern char*  upper (char *s);
extern char*  lower (char *s);
extern int    qfilef (const char *fname); /* TRUE if file exists */
extern int    free_storage (char *fn); /* returns free storage in file system of fn */
extern char*  util_strdup(char *string);
extern void*  util_malloc (int size);
extern void*  util_realloc (void *p, int size);
extern void*  util_calloc (int num, int size);
extern void   util_free (void *p);
extern int    util_space_in_use (void);
extern int    util_pointers_in_use (void);
extern void error(char *fmt, ...);

#endif
