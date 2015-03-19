/*====================================================================*/
/*  A group of routines to extract parameters from scripts and import */
/* them into C/C++ programs.  J. Lafferty 2/2/95                      */
/*====================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "param.h"
#include "util.h"

/* #define TRACE 1 */
#define EOL '\n'
#define EOS '\0'
#define DELIMIT_CHAR '='
#define ENDMARK_CHAR ';'
#define COMMAND_CHAR '%'
#define EMBEDDED_VAR_CHAR '@'
#define FIXED_CHAR '+'
#define QUOTE_CHAR '\''
#define DQUOTE_CHAR '"'
#define DDTABLE_SIZE 1001
#define MAX_SYMBOL_LENGTH 65535
#define DDINF_DEFAULT_NAME "DDINF"
#define TRUE 1
#define FALSE 0

static  int  restart_inited = FALSE;
static  int  restart_is_new = FALSE;
static  int  checkpointed = FALSE;
static  FILE *restart_unit=NULL;
static  char *restart_name=NULL;
static  char *ddname_buffer;
static  char *ddvalue_buffer;
static  char *command_buffer;
static  char *argument_buffer;
static  int  nameBufferIndex=0, valueBufferIndex=0;
static  int  commandBufferIndex, argumentBufferIndex;
static  char *begin_restart_header="/***BEGIN_RESTART***/\n";
static  char *end_restart_header="/***END_RESTART***/\n";
static  int  NumberOfYesStrings = 3;
static  char *YesString[] = {"YES", "TRUE", "1"};

/*********************************************************/
/*    Various linked lists:                              */
/*********************************************************/
typedef struct ddlink ddlink;
typedef struct clink clink;
typedef struct ddinf_link ddinf_link;
typedef struct ddpar_link ddpar_link;

struct ddlink {
       char *ddname;
       char *ddvalue;
       int   fixed;
       ddlink *next;
};
struct clink {
       char *name;
       clink *next;
};
struct ddinf_link {
       char *name;
       FILE *stream;
       ddinf_link *next;
};
struct ddpar_link {
       char *file_name;
       int   par_inited;
       ddlink **ddtable;
       ddpar_link *next;
};

static ddinf_link *ddinf=NULL;
static clink      *prefix_stack=NULL;
static ddpar_link *ddpar=NULL;
static char       *ddinfName;

/***************************************************/
/*  Internal procedures:                           */
/***************************************************/
int     param_init(void);
void    param_init_restart(void);
int     same_string_upcase_first_arg(char *s, char *t);
void    parse_stream(FILE *stream, char *file_name);
ddinf_link *push_last_complete_block (char *iname);
ddinf_link *push_ddinf_stream (FILE* instream, char *inname);
ddinf_link *push_ddinf (char *inname);
void    pop_ddinf (void);
ddlink *insert (const char *newddname, const char *newddvalue);
void    ddinf_error (char *ermsg);
double  safe_atod (const char *name, const char *expr);
long    safe_atol (const char *name, const char *expr);

void param_init_buffers(void) {
    static int buffers_inited=FALSE;

    if (buffers_inited) return;
    ddname_buffer = (char *) malloc(MAX_SYMBOL_LENGTH);
    ddvalue_buffer = (char *) malloc(MAX_SYMBOL_LENGTH);
    command_buffer = (char *) malloc(80);
    argument_buffer = (char *) malloc(MAX_SYMBOL_LENGTH);
    buffers_inited = TRUE;
}


/***************************************************/
int param_init(){
   static int ddinf_inited=FALSE;

   param_init_buffers();
   if (ddinf_inited) return TRUE;

   ddinfName = (char *) malloc(80);
   ddinfName = getenv("PARAM");
   ddinf_inited = TRUE;

   if (ddinfName != NULL) param_push_file(ddinfName);
   return TRUE;
}

/***************************************************/
void param_init_par() {
   FILE *stream;

   param_init();
   if ((ddpar == NULL) || (ddpar->par_inited)) return;
   printf("filename: %s", ddpar->file_name);
   stream = fopen(ddpar->file_name, "r");
   parse_stream(stream, ddpar->file_name);
   ddpar->par_inited = TRUE;
}



/*******************************************************/
/*      Initialize ddinf names and values.  This is    */
/*      organized as a simple finite-state machine.    */
/*******************************************************/
void parse_stream(FILE *instream, char *inname){
   printf("ddinf: %d\n", 0);
   int c, next_c;
          /* These are really characters, but the library functions use ints */
   int commentLevel = 0;

   enum buffer_state {
      BEGIN_LINE, ENDMARKED, COMMENT, NAME, NAME_ENDED,
      VALUE, DIRECTIVE, ARGUMENT, QUOTATION, QUOTATION_ENDED, DQUOTATION, DQUOTATION_ENDED
     } state = BEGIN_LINE, prev_state = BEGIN_LINE;
   enum character_class {
      WHITESPACE, COMMENT_START, COMMENT_END, DELIMITER, ENDMARK,
      END_OF_LINE, COMMAND, QUOTE, DQUOTE, OTHER, FIXIT
     } cclass;

   ddinf = push_ddinf_stream(instream, inname);

   printf("ddinf: %d\n", ddinf);

   while (ddinf != NULL) {
      while ((c=getc(ddinf->stream)) != EOF) {
	 switch (c) {
	  case DELIMIT_CHAR: cclass = DELIMITER; break;
	  case ENDMARK_CHAR: cclass = ENDMARK; break;
	  case EOL:          cclass = END_OF_LINE; break;
	  case COMMAND_CHAR: cclass = COMMAND; break;
	  case QUOTE_CHAR:   cclass = QUOTE; break;
	  case DQUOTE_CHAR:  cclass = DQUOTE; break;
	  case FIXED_CHAR:   cclass = FIXIT; break;
	  case '/':
            if ((state == QUOTATION)  || (state == DQUOTATION))
               cclass = OTHER;
            else
               if ((next_c = getc(ddinf->stream)) == '*')
	          cclass = COMMENT_START;
	       else {
	          cclass = OTHER;
	          if (ungetc(next_c, ddinf->stream) == EOF)
                      error("PARAM: ungetc failed.");
	       }
	    break;
	  case '*':
            if ((state == QUOTATION) || (state == DQUOTATION))
               cclass = OTHER;
            else
	       if ((next_c = getc(ddinf->stream)) == '/')
	          cclass = COMMENT_END;
	       else {
	          cclass = OTHER;
	          if (ungetc(next_c, ddinf->stream) == EOF)
		     error("PARAM: ungetc failed.");
	       }
	    break;
	  default:
	    if (isspace(c)) cclass = WHITESPACE;
	    else cclass = OTHER;
	 }
	 switch (state) {
	  case BEGIN_LINE:
	    switch (cclass) {
	     case WHITESPACE: case END_OF_LINE: break;
	     case COMMENT_START:
	       prev_state = BEGIN_LINE;
	       state = COMMENT;
               commentLevel++;
	       break;
	     case COMMENT_END: ddinf_error("Unmatched comment ending");
	     case DELIMITER: ddinf_error("Missing parameter name.");
	     case QUOTE: ddinf_error("Quotations for values only.");
	     case DQUOTE: ddinf_error("Quotations for values only.");
	     case ENDMARK: state = ENDMARKED; break;
	     case COMMAND: commandBufferIndex = 0; state = DIRECTIVE; break;
             case FIXIT: ddname_buffer[nameBufferIndex++] = c; break;
	     case OTHER:
	       ddname_buffer[nameBufferIndex++] = c;
	       state = NAME;
               break;
	    }
	    break;
	  case ENDMARKED:
	    switch (cclass) {
	     case WHITESPACE: break;
	     case COMMENT_START:
	       prev_state = ENDMARKED;
	       state = COMMENT;
               commentLevel++;
	       break;
	     case COMMENT_END: ddinf_error("Unmatched comment ending");
	     case DELIMITER: ddinf_error("Missing parameter name.");
	     case QUOTE: ddinf_error("Quotations for values only.");
	     case DQUOTE: ddinf_error("Quotations for values only.");
	     case ENDMARK: ddinf_error("Null production");
	     case COMMAND: commandBufferIndex = 0; state = DIRECTIVE; break;
	     case OTHER:
	       ddname_buffer[nameBufferIndex++] = c;
	       state = NAME;
               break;
	     case FIXIT:
	       ddname_buffer[nameBufferIndex++] = c;
	       state = NAME;
               break;
	     case END_OF_LINE: state = BEGIN_LINE; break;
 	     default: 
	       ddinf_error("Illegal character after end marker.");
	    }
	    break;
	  case COMMENT:
            switch (cclass) {
               case COMMENT_END: commentLevel--;
                    if (commentLevel == 0)
                       state = prev_state;
                    break;
               case COMMENT_START: commentLevel++; break;
	       default: break;
	     }
            break;
	  case NAME:
	    switch (cclass) {
	     case WHITESPACE: case END_OF_LINE:
	       state = NAME_ENDED;
	       break;
	     case COMMENT_START:
	       prev_state = NAME_ENDED;
	       state = COMMENT;
               commentLevel++;
	       break;
	     case COMMENT_END: ddinf_error("Unmatched comment ending");
	     case QUOTE: ddinf_error("Quotes for values only.");
	     case DQUOTE: ddinf_error("Quotes for values only.");
	     case DELIMITER:
	       state = VALUE;
	       break;
	     case ENDMARK: ddinf_error("Missing value.");
	     case COMMAND:  /* XXX shouldn't this be allowed if name is begun ?*/
	       ddinf_error("Illegal use of COMMAND character.");
             case FIXIT:
	       ddinf_error("Illegal use of FIXIT character.");
	     case OTHER: ddname_buffer[nameBufferIndex++] = c;
	    }
	    break;
	  case NAME_ENDED:
	    switch (cclass) {
	     case WHITESPACE: case END_OF_LINE: break;
	     case COMMENT_START:
	       prev_state = NAME_ENDED;
	       state = COMMENT;
               commentLevel++;
	       break;
	     case DELIMITER:
	       state = VALUE;
	       break;
	     default:
	       ddinf_error("Illegal character after parameter name.");
	    }
	    break;
	  case VALUE:
	    switch (cclass) {
	     case WHITESPACE:
                break;
	     case COMMENT_START:  /* XXX be sure this means not quoted!! */
	        prev_state = VALUE;
	        state = COMMENT;
                commentLevel++;
	        break;
	     case COMMENT_END: ddinf_error("Unmatched comment ending");
	     case DELIMITER: ddinf_error("Missing delimiter or endmark.");
	     case ENDMARK:
                ddname_buffer[nameBufferIndex++] = EOS;
                ddvalue_buffer[valueBufferIndex++] = EOS;
                if (insert(ddname_buffer, strip(ddvalue_buffer)) == NULL)
                   ddinf_error("error adding (name,value) pair");
                nameBufferIndex = 0;
                valueBufferIndex = 0;
	        state = ENDMARKED;
	        break;
	     /* case END_OF_LINE: ddinf_error("Missing end marker."); */
	     case END_OF_LINE: break;
	     case QUOTE: state = QUOTATION; break;
	     case DQUOTE: state = DQUOTATION; break;
	     default:
	       ddvalue_buffer[valueBufferIndex++] = c;
	    }
	    break;
	  case DIRECTIVE:
	    switch (cclass) {
	     case WHITESPACE:
	       command_buffer[commandBufferIndex] = EOS;
	       argumentBufferIndex = 0;
	       state = ARGUMENT;
	       break;
	     case OTHER: command_buffer[commandBufferIndex++] = c; break;
	     default: ddinf_error("Illegal character in ddinf directive");
	    }
	    break;
	  case ARGUMENT:
	    switch (cclass) {
	     case WHITESPACE: break;
	     case COMMENT_START:
	       prev_state = ARGUMENT;
	       state = COMMENT;
               commentLevel++;
	       break;
	     case COMMENT_END: ddinf_error("Unmatched comment ending");
	     case ENDMARK:
	       argument_buffer[argumentBufferIndex] = EOS;
	       if (strcmp(command_buffer, "include") == 0) {
                  ddinf = push_ddinf(argument_buffer);
		  if (ddinf->stream == NULL)
		    ddinf_error("Unable to open ddinf");
		  state = BEGIN_LINE;
	       }
	       else if (strcmp(command_buffer, "include_restart") == 0) {
                  ddinf = push_last_complete_block(argument_buffer);
		  if (ddinf->stream == NULL)
		    ddinf_error("Unable to open ddinf");
		  state = BEGIN_LINE;
	       }
               else if (strcmp(command_buffer, "restart")==0) {
                  if (strstr(argument_buffer,"(new") != 0) {
                      *strchr(argument_buffer, '(') = EOS;
                      restart_is_new = TRUE;
		  }
                  insert("%restart", argument_buffer);
                  state = BEGIN_LINE;
               }
	       else ddinf_error("Unrecognized ddinf directive.");
	       break;
	     case END_OF_LINE: ddinf_error("Missing end marker");
             default:
	       argument_buffer[argumentBufferIndex++] = c;
	    }
	    break;
	  case QUOTATION:
	    if (cclass == QUOTE) state=QUOTATION_ENDED;
	    else ddvalue_buffer[valueBufferIndex++] = c;
	    break;
	  case DQUOTATION:
	    if (cclass == DQUOTE) state=DQUOTATION_ENDED;
	    else ddvalue_buffer[valueBufferIndex++] = c;
	    break;
	  case QUOTATION_ENDED:
	    switch (cclass) {
	       case WHITESPACE: break;
	       case COMMENT_START:
		 prev_state = VALUE;
		 state = COMMENT;
                 commentLevel++;
		 break;
	       case COMMENT_END: ddinf_error("Unmatched comment end.");
	       case ENDMARK:
                 ddname_buffer[nameBufferIndex++] = EOS;
                 ddvalue_buffer[valueBufferIndex++] = EOS;
                 if (insert(ddname_buffer, ddvalue_buffer) == NULL)
                    ddinf_error("error adding (name,value) pair");
                 nameBufferIndex = 0;
                 valueBufferIndex = 0;
	         state = ENDMARKED;
	         break;
	       case END_OF_LINE: break;
	       case QUOTE:
		 state = QUOTATION;
		 ddvalue_buffer[valueBufferIndex++] = c;
		 break;
	       default:
		 ddinf_error("Illegal character after quotation.");
	    }
            break;
	  case DQUOTATION_ENDED:
	    switch (cclass) {
	       case WHITESPACE: break;
	       case COMMENT_START:
		 prev_state = VALUE;
		 state = COMMENT;
                 commentLevel++;
		 break;
	       case COMMENT_END: ddinf_error("Unmatched comment end.");
	       case ENDMARK:
                 ddname_buffer[nameBufferIndex++] = EOS;
                 ddvalue_buffer[valueBufferIndex++] = EOS;
                 if (insert(ddname_buffer, ddvalue_buffer) == NULL)
                    ddinf_error("error adding (name,value) pair");
                 nameBufferIndex = 0;
                 valueBufferIndex = 0;
	         state = ENDMARKED;
	         break;
	       case END_OF_LINE: break;
	       case DQUOTE:
		 ddinf_error("Illegal character after quotation.");
		 break;
	       default:
		 ddinf_error("Illegal character after quotation.");
                 break;
	    }
            break;
	 } /* Matches switch on current state */
      } /* Matches while not end-of-file */
      if ((state != ENDMARKED) && (state != BEGIN_LINE)) {
	 if ((state == QUOTATION) || (state == DQUOTATION)) {
            ddinf_error("end of file inside quoted string.");
	 }
         else if (commentLevel != 0) {
            ddinf_error("end of file inside comment.");
         }
	 else {
            ddinf_error("Not in proper state at end of file.");
         }
      }
      pop_ddinf();
   } /* Matches while ddinf != NULL */
}


/*******************************************************/
void param_dump (FILE *stream) {
    int i;
    ddlink *lp;
    param_init_par();
    for (i=0; i<DDTABLE_SIZE; ++i) {
        for (lp = ddpar->ddtable[i]; lp != NULL; lp = lp->next) {
            if (lp->fixed) fputc('+', stream);
            fputs(lp->ddname, stream);
            fputs(" = ", stream);
            fputs(quote(lp->ddvalue), stream);
            fputs(";\n", stream);
        }
    }
}


/*******************************************************/
static unsigned hash (const char *s) {
    unsigned hashval;
    for (hashval=0; *s != EOS; s++)
         hashval = *s + 31*hashval;
    return hashval % DDTABLE_SIZE;
}

/*******************************************************/
ddlink *lookup (const char *s) {
    ddlink *lp;
    int i;
    i = hash(s);
    if (ddpar == NULL) return NULL;
    for (lp = ddpar->ddtable[hash(s)]; lp != NULL; lp = lp->next)
        if (strcmp(s, lp->ddname) == 0)
           return lp;
    return NULL;
}

/*******************************************************/
ddlink *prefixed_lookup (const char *s) {
    static char *prefixed_s=NULL;
    if (prefixed_s == NULL)
        prefixed_s = (char *) malloc (MAX_SYMBOL_LENGTH);
    if (prefix_stack == NULL) return lookup(s);
    sprintf(prefixed_s, "%s%s", prefix_stack->name, s);
    return lookup(prefixed_s);
}

/*******************************************************/

/* begin ALB 9/96 */

char *lookup_ddval(const char *p, int *len)
{
    ddlink *lp;
    char ddparm[MAX_SYMBOL_LENGTH];
    int i, nopenparens;
    if (p[0] !='(')
      {
	  /* No (), embedded ddparm takes up remainder of p */
	  strcpy(ddparm,p);
	  *len = strlen(p)+1;
      }
    else
      {
	  nopenparens=1;
	  for (i=1; p[i] && nopenparens>0; i++)
	    {
		if (p[i]=='(') nopenparens++;
		if (p[i]==')') nopenparens--;
	    }
	  if (nopenparens>0) error("PARAM: Unmatched (");
	  memcpy(ddparm, &p[1], i-2);
	  ddparm[i-2]=0;
	  *len = i+1;
      }

    /* sanity check: no nested % */
    if (strchr(ddparm, EMBEDDED_VAR_CHAR)!=NULL)
      error("PARAM: No nested %% allowed.");

    if ((lp = lookup(ddparm)) == NULL)
      error("PARAM: symbol %s not defined (yet)\n", ddparm);
    return lp->ddvalue;
}


void expand_ddval(const char *ddval_raw, char *ddval_expanded)
{
    char c, *ddval;
    int i=0,j=0,len;
    while((c=ddval_raw[i])!='\0')
      {
	  if (c == EMBEDDED_VAR_CHAR)
	    {
		ddval = lookup_ddval(&ddval_raw[i+1], &len);
		strcpy(&ddval_expanded[j], ddval);
		i+= len;
		j+= strlen(ddval);
	    }
	  else
	    ddval_expanded[j++] = ddval_raw[i++];
      }
    ddval_expanded[j] = 0;
}

/*******************************************************/
ddlink *insert (const char *newddname, const char *newddvalue_raw) {
    ddlink *lp;
    unsigned hashval;
    char newddvalue_buff[MAX_SYMBOL_LENGTH];
    char *newddvalue = &newddvalue_buff[0];
    int fixit=FALSE;

    if (newddname[0] == '+') {
       fixit = TRUE;
       newddname++;
    }

    expand_ddval(newddvalue_raw,newddvalue); /* ALB 9/96 */

    if ((lp = lookup(newddname)) == NULL) {
       lp = (ddlink *) malloc (sizeof(*lp));
       lp->ddname = (char *) malloc(strlen(newddname)+1);
       lp->ddvalue = (char *) malloc(strlen(newddvalue)+1);
       if (lp == NULL || lp->ddname == NULL || lp->ddvalue == NULL) return NULL;
       hashval = hash(newddname);
       strcpy(lp->ddname, newddname);
       strcpy(lp->ddvalue, newddvalue);
       lp->next = ddpar->ddtable[hashval];
       lp->fixed = fixit;
       ddpar->ddtable[hashval] = lp;
    }
    else if (!lp->fixed) {
       free ((char *) lp->ddvalue);
       if ((lp->ddvalue = (char *) malloc(strlen(newddvalue)+1)) == NULL)
          return NULL;
       strcpy(lp->ddvalue, newddvalue);
    }
    else if (lp->fixed && fixit) {
       if (strcmp(lp->ddvalue, newddvalue) == 0)
          printf("PARAM: Warning: Symbol %s fixed more than once\n", lp->ddname);
       else
          error("PARAM: Error: Symbol %s already fixed", lp->ddname);
    }
    return lp;
}



/*******************************************************/
int param_checkpointed(void){
   param_init_par();
   if (checkpointed) return TRUE;
   if (restart_is_new || qfilef(param_getc("%restart",""))==0)
      return FALSE;
   else {
      param_init_restart();
      checkpointed = TRUE;
      return TRUE;
   }
}


/*******************************************************/
void param_checkpoint(void){
   param_init_par();
   if (!param_checkpointed()) param_init_restart();
   fputs(end_restart_header, restart_unit);
   fputs(begin_restart_header, restart_unit);
   fflush(restart_unit);
   checkpointed = TRUE;
}


/*******************************************************/
ddinf_link *push_last_complete_block (char *inname) {
   int i, j;
   int last_restart_rec, next_to_last_restart_rec;
   FILE *unit;
   ddinf_link *dl;
   char *buffer;

   dl = (ddinf_link *) malloc(sizeof(*dl));
   dl->next = ddinf;
   dl->name = (char *) malloc(strlen(inname)+1);
   strcpy(dl->name, inname);
   buffer = (char *) malloc(MAX_SYMBOL_LENGTH);
   if (!(unit = fopen(inname, "r")))
      error("PARAM: Unable to open restart %s", inname);
   j = last_restart_rec = next_to_last_restart_rec = 0;
   while(fgets(buffer, 100, unit)) {
      j++;
      if (strcmp(buffer,begin_restart_header)==0){
         next_to_last_restart_rec = last_restart_rec;
         last_restart_rec = j;
      }
   }
   fclose(unit);
   dl->stream = fopen(inname, "r");
   if (next_to_last_restart_rec != 0)  /* reposition manually */
      for (i=0; i<next_to_last_restart_rec; ++i)
          fgets(buffer, MAX_SYMBOL_LENGTH, dl->stream);
   free(buffer);
   return dl;
}


/*******************************************************/
void param_init_restart(void){
   ddinf_link *dl;

   param_init_par();
   if (restart_inited) return;

   if (restart_name==NULL)
     {
       restart_name = (char *) malloc(strlen(param_getc("%restart",""))+1);
       strcpy(restart_name, param_getc("%restart",""));
     }

   if (restart_is_new || qfilef(restart_name)==0) {
      if (!(restart_unit = fopen(restart_name, "w")))
         error("PARAM: Unable to open restart %s", restart_name);
      fputs(begin_restart_header, restart_unit);
      fflush(restart_unit);
   }
   else {
      dl = push_last_complete_block(restart_name);
      parse_stream(dl->stream, restart_name);
      if (!(restart_unit = fopen(restart_name, "a")))
         error("PARAM: Unable to open restart %s", restart_name);
   }
   restart_inited = TRUE;
}


/*******************************************************/
ddinf_link *push_ddinf_stream (FILE* instream, char *inname) {
    ddinf_link *dl;
    dl = (ddinf_link *) malloc(sizeof(*dl));
    dl->next = ddinf;
    dl->name = (char *) malloc(strlen(inname)+1);
    strcpy(dl->name, inname);
    dl->stream = instream;
    return dl;
}

/*******************************************************/
ddinf_link *push_ddinf (char *inname) {
    ddinf_link *dl;
    dl = (ddinf_link *) malloc(sizeof(*dl));
    dl->next = ddinf;
    dl->name = (char *) malloc(strlen(inname)+1);
    strcpy(dl->name, inname);
    if ((dl->stream = fopen(dl->name, "r")) == NULL)
       error("PARAM: Unable to open ddinf %s", dl->name);
    return dl;
}

/*******************************************************/
void pop_ddinf (void) {
    if (ddinf == NULL) return;
    free (ddinf->name);
    fclose (ddinf->stream);
    ddinf = ddinf->next;
}

/*******************************************************/
int param_set_file (char *fn) {
    return 0;
}

/*******************************************************/
void param_unset_file (char *fn) {
}


/*******************************************************/
void param_push_prefix (const char *hot_prefix) {
    clink *pp;
    pp = (clink *) malloc(sizeof(*pp));
    pp->next = prefix_stack;
    pp->name = (char *) malloc(strlen(hot_prefix)+1);
    strcpy(pp->name, hot_prefix);
    prefix_stack = pp;
}


/*******************************************************/
void param_pop_prefix (void) {
    if (prefix_stack == NULL) return;
    free ((char *) prefix_stack->name);
    prefix_stack = prefix_stack->next;
}


/*******************************************************/
void param_set_restart_file(const char *restart_name_p)
{
    restart_name = (char *) malloc (strlen(restart_name_p)+1);
    strcpy(restart_name, restart_name_p);
}

/*******************************************************/
void param_puti(const char *var_name, int val){
    param_init_restart();
    fprintf(restart_unit, "%s = %d;\n", var_name, val);
    fflush(restart_unit);
}


/*******************************************************/
void param_putf(const char *var_name, double val){
    param_init_restart();
    fprintf(restart_unit, "%s = %18.11e;\n", var_name, val);
    fflush(restart_unit);
}


/*******************************************************/
void param_putc(const char *var_name, char *val){
    param_init_restart();
    fprintf(restart_unit, "%s = %s;\n", var_name, quote(val));
    fflush(restart_unit);
}


/*******************************************************/
void param_fwritei(FILE *fp, char *var_name, int val){
    fprintf(fp, "%s = %d;\n", var_name, val);
}


/*******************************************************/
void param_fwritef(FILE *fp, char *var_name, double val){
    fprintf(fp, "%s = %18.11e;\n", var_name, val);
}

/*******************************************************/
void param_fwritec(FILE *fp, char *var_name, char *val) {
    fprintf(fp, "%s = %s;\n", var_name, quote(val));
}


/*******************************************************/
void param_set(const char *parameter_name, char *new_value) {
   param_init_par();
   insert(parameter_name, new_value);
}


/*******************************************************/
int param_geti(const char *var_name, int dflt){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return dflt;
    else return safe_atol(var_name, lp->ddvalue);
}


/*******************************************************/
int param_symvarie(const char *var_name, int *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return FALSE;
    else *var = safe_atol(var_name, lp->ddvalue);
    return TRUE;
}

/*******************************************************/
void param_symvari(char *var_name, int *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL)
    {
       printf("\nddinf var %s not defined.\n",var_name);
       assert(0);
    }
    else *var = safe_atol(var_name, lp->ddvalue);
 }

/*******************************************************/
double param_getf(const char *var_name, double dflt){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return dflt;
    else return safe_atod(var_name, lp->ddvalue);
  }


/*******************************************************/
int param_symvarfe(const char *var_name, double *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return FALSE;
    else *var = safe_atod(var_name, lp->ddvalue);
    return TRUE;
}


/*******************************************************/
void param_symvarf(char *var_name, double *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL)
    {
       printf("\nddinf var %s not defined.\n",var_name);
       assert(0);
    }
    else *var = safe_atod(var_name, lp->ddvalue);
 }


/*******************************************************/
char *param_getc(const char *var_name, char *dflt){
    ddlink *lp;
    printf("getc\n");
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return dflt;
    else return lp->ddvalue;
    }

/*******************************************************/
char *param_gets(const char *var_name){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) {
	 error("PARAM: variable %s not found.", var_name);
	 return NULL;
    }
    else return lp->ddvalue;
    }

/*******************************************************/
int param_symvarce(const char *var_name, char *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return FALSE;
    else strcpy(var, lp->ddvalue);
    return TRUE;
}


/*******************************************************/
void param_symvarc(char *var_name, char *var){
    ddlink *lp;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL)
    {
       printf("\nddinf var %s not defined.\n",var_name);
       assert(0);
    }
    else strcpy(var, lp->ddvalue);
 }

/*******************************************************/
int param_getb(const char *var_name, int dflt) {
    ddlink *lp;
    int i;
    param_init_par();
    if ((lp = prefixed_lookup(var_name)) == NULL) return dflt;
    else for (i=0; i< NumberOfYesStrings; i++)
           if (same_string_upcase_first_arg(lp->ddvalue, YesString[i]))
	     return TRUE;
    return FALSE;
}


/*******************************************************/
int same_string_upcase_first_arg(char *s, char *t) {
   while ((*s != EOS) && (toupper(*s) == *t)) {
      s++;
      t++;
   }
   return *s == *t;
}


/*******************************************************/
void  ddinf_error (char *ermsg) {
    error("ERROR IN DDINF=%s: %s", ddinf->name, ermsg);
}


/*******************************************************/
double safe_atod (const char *name, const char *expr) {
    double f = atof(expr);
    /* if (strcmp(suffix,"") != 0)
       error("PARAM: Unable to convert value \"%s\" for ddname \"%s\"", expr, name); */
    return f;
}

/*******************************************************/
long safe_atol (const char *name, const char *expr) {
    long d;
    char *suffix;
    d = strtol(expr, &suffix, 0);
    if (strcmp(suffix,"") != 0)
       error("PARAM: Unable to convert value \"%s\" for ddname \"%s\"",
       expr, name);
    return d;
}


/*******************************************************/
int param_push_file (const char *fn) {
    ddpar_link *d, *e;
    const char *pname=fn;

    param_init();

    if (strcmp(DDINF_DEFAULT_NAME, fn)==0) pname = ddinfName;

    for (e=ddpar; e != NULL; e = e->next)
        if (strcmp(e->file_name, pname)==0) break;

    /* If not found, initialize */
    if (e == NULL) {
        if (!qfilef(fn)) return FALSE;
        d = (ddpar_link *) malloc(sizeof(ddpar_link));
        d->file_name = (char *) malloc(strlen(fn)+1);
        strcpy(d->file_name, fn);
        d->ddtable = (ddlink **) calloc(DDTABLE_SIZE, sizeof(ddlink *));
        d->par_inited = FALSE;
        d->next = ddpar;
        ddpar = d;
        /* stream = fopen(fn, "r"); */
        /* parse_stream(stream, fn); */
    }
    /* If already found, use its table */
    else {
        d = (ddpar_link *) malloc(sizeof(ddpar_link));
        d->file_name = (char *) malloc(strlen(fn)+1);
        strcpy(d->file_name, fn);
        d->ddtable = e->ddtable;
        d->par_inited = e->par_inited;
        d->next = ddpar;
        ddpar = d;
    }
    return TRUE;
}


/*******************************************************/
char * param_pop_file (void) {
    ddpar_link *d;

    d = ddpar;
    if ((d != NULL) && (d->next == NULL)) return NULL; /* Don't allow the stack to be */
    else if (ddpar != NULL) ddpar = ddpar->next;       /* completely cleared          */

    free (d->ddtable);
    return d->file_name;
}
