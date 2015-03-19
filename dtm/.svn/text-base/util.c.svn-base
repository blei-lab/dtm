// Author: David Blei (blei@cs.princeton.edu)
//
// Copyright 2006 David Blei
// All Rights Reserved.
//
// See the README for this package for details about modifying or
// distributing this software.

#define ABNORMAL_RETURN_CODE 1
#define MAX_STRING_LENGTH    65535

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>
#include <ctype.h>
#include "util.h"

#ifdef __alpha__

#include <sys/mount.h>
#include <malloc.h>
#include <stdlib.h>
#include <unistd.h>

#else

/*#include <malloc.h>*/
#include <stdlib.h>
#include <unistd.h>
/*#include <sys/vfs.h>*/

#endif

char   buf[1024];
static int space_in_use=0;
static int pointers_in_use=0;
int    display_allocs=FALSE;


void error(char *fmt, ...){
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args); CRLF;
    va_end(args);
    fprintf(stderr, "\n");
    if (errno > 0) {
        perror(buf);
        fprintf(stderr, "errno=%d\n", errno);
        fprintf(stderr, buf);
      fprintf(stderr, "\n");
    }
    fflush(stderr);
    fflush(stdout);
    assert(0);
}

void bomb(char *fmt, ...)
{
   /* just break out, with error code =1 (fail) */

   va_list args;
   va_start(args, fmt);
   vfprintf(stderr, fmt, args); CRLF;
   va_end(args);
   fprintf(stderr, "\n");
   fflush(stderr);
   fflush(stdout);
   exit(1);
}


void bail(char *fmt, ...)
{
   /* just break out, with error code =0 (success) */

   va_list args;
   va_start(args, fmt);
   vfprintf(stderr, fmt, args); CRLF;
   va_end(args);
   fprintf(stderr, "\n");
   fflush(stderr);
   fflush(stdout);
   exit(0);
}



char *dequote (char *s) {
    static char *sbuf=NULL;
    char *t;
    int i;
    if (s[0] != '\'') return s;
    else if ((i=strlen(s)) < 2) return s;
    else if (s[i-1] != '\'')
       error("Illegal string passed to dequote: %s", s);
    if (sbuf == NULL)
        sbuf = (char *) malloc(MAX_STRING_LENGTH);
    t = sbuf;
    s++;
    while(*s != EOS) {
       if (*s == '\'') s++;
       *t = *s;
       s++; t++;
    }
    *t = EOS;
    return sbuf;
}

void quote_no_matter_what (const char *s, char *t) {
    *t = '\'';
    t++;
    while((*s != EOS)) {
        *t = *s;
        if (*s == '\'') {
           t++; *t = '\'';
	}
        s++; t++;
    }
    *t = '\''; t++;
    *t = EOS;
}


const char *quote (const char *s) {
    static char *sbuf=NULL;
    if (sbuf == NULL)
        sbuf = (char *) malloc(MAX_STRING_LENGTH);
    if ( strchr(s,' ')  == NULL  &&
         strstr(s,"/*") == NULL && strstr(s,"*/") == NULL ) return s;
    else {
       quote_no_matter_what(s, sbuf);
       return sbuf;
    }
}




/* returns TRUE iff string only contains chars in valid. */
int verify(char *string, char *valid)
{
   int i;
   for(i=0;i<strlen(string);i++)
      if (!strchr(valid, string[i])) return TRUE;
   return FALSE;
}


/* strips leading and trailing white space */
char * strip(char *s) {
   int i,j;
   int hit_char;

   j = 0;
   hit_char = FALSE;
   for (i=0; i<=strlen(s); ++i) {
       if (s[i] != ' ') hit_char = TRUE;
       if (hit_char) s[j++] = s[i];
   }
   for (i=strlen(s)-1; i>0; --i)
       if (s[i] != ' ') break;
   s[i+1] = '\0';
   return s;
}


/* converts s to upper case */
char * upper(char *s) {
   int i;
   for (i=0; i<strlen(s); ++i) s[i] = toupper(s[i]);
   return s;
}

/* converts s to lower case */
char * lower(char *s) {
   int i;
   for (i=0; i<strlen(s); ++i) s[i] = tolower(s[i]);
   return s;
}


/* queries existence of file */
int qfilef(const char *fname) {
   if (fname == FALSE) return FALSE;
   if (access(fname, F_OK)==0) return TRUE;
   else return FALSE;
}


/* returns free storage in file system */
int free_storage (char *fn)
{
  /* uses a defunct function call. Also, not ever called */
  abort();
  /*
  struct statfs sfs;
  if (statfs(fn, &sfs) == -1)
    return -1;
  return sfs.f_bsize * sfs.f_bfree;
*/
}

/* Return the size of file named filename */
int file_size(char *filename)
{
  struct stat status;

  if (stat(filename,&status) != 0)
    return -1;
  return (int)status.st_size;
}

/* Return an allocated duplicate of string */
char *util_strdup(char *string)
{
  int len = strlen(string);
  char *dup = (char *)malloc(len+1);

  if (dup == NULL)
    {
      perror("malloc");
      return NULL;
    }
  strcpy(dup, string);
  return dup;
}


void * util_malloc (int size)
{
    char * p = (char *) malloc(size+sizeof(int));
    if (p == NULL) error("UTIL_MALLOC: Ran out of space. Space in use: %d (%d pointers)\n",
                                           space_in_use, pointers_in_use);
    space_in_use += size;
    ++pointers_in_use;
    *((int *) p) = size;
    if (display_allocs)
       fprintf(stderr, "UTIL_MALLOC: Allocated %d bytes, %d bytes total, %d pointers\n",
                                    size, space_in_use, pointers_in_use);
    return (void *) (p+sizeof(int));
}

void * util_calloc (int num, int size)
{
    char * p = (char *) calloc(num*size+sizeof(int), 1);
    if (p == NULL) error("UTIL_CALLOC: Ran out of space. Space in use: %d (%d pointers)\n",
                                           space_in_use, pointers_in_use);
    space_in_use += num*size;
    ++pointers_in_use;
    *((int *) p) = num*size;
    if (display_allocs)
       fprintf(stderr, "UTIL_CALLOC: Allocated %d bytes, %d bytes total, %d pointers\n",
                                           num*size, space_in_use, pointers_in_use);
    return (void *) (p+sizeof(int));
}

void * util_realloc (void * p, int size)
{
    int oldsize;
    char *realp;
    realp = ((char *)p)-sizeof(int);
    oldsize = *((int *)(realp));
    realp = (char *) realloc(realp, size+sizeof(int));
    if (realp == NULL) error("UTIL_REALLOC: Ran out of space. Space in use: %d (%d pointers)\n",
                                           space_in_use, pointers_in_use);
    *((int *)(realp)) = size;
    space_in_use += (size-oldsize);
    if (display_allocs)
        fprintf(stderr, "UTIL_REALLOC: Allocated %d bytes, %d bytes total, %d pointers\n",
                                        size, space_in_use, pointers_in_use);
    return (realp+sizeof(int));
}

void util_free (void * p)
{
    int size;
    size = *((int *) (((char *) p)-sizeof(int)));
    space_in_use -= size;
    --pointers_in_use;
    free(((char *)p)-sizeof(int));
    if (display_allocs)
       fprintf(stderr, "UTIL_FREE: Freed up %d bytes, %d bytes remaining, %d pointers\n",
                                        size, space_in_use, pointers_in_use);
}

int util_space_in_use (void)
{
   return space_in_use;
}

int util_pointers_in_use (void)
{
   return pointers_in_use;
}
