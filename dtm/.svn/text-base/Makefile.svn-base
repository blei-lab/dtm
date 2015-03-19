.SUFFIXES: .c .u
CC= g++

LIB=../lib
GSLWRAP_LIB=../gslwrap
LOCAL_DIR=../local

CFLAGS = -I ${LIB} -I ${LIB}/math -I ${GSLWRAP_LIB}/include -I	\
  ${GSLWRAP_LIB}/include/gslwrap -I ${LOCAL_DIR}/include -I	\
  ${LIB}/util/gflags-1.1/src/gflags

GDB_CFLAGS = -ggdb -I ${LIB} -I ${LIB}/math -I ${GSLWRAP_LIB}/include	\
  -I ${GSLWRAP_LIB}/include/gslwrap -I ${LOCAL_DIR}/include -I		\
  ${LIB}/util/gflags-1.1/src/gflags

BASE_INCLUDES = -I ${LIB}

LDFLAGS = -L${LOCAL_DIR}/lib -L${LOCAL_DIR}/lib/stl -lgsl -lm -lgslcblas
LOBJECTS = ss-lm.o gsl-wrappers.o data.o param.o util.o lda-seq.o	\
  lda.o params.o main.o

GFLAGS = gflags.o gflags_reporting.o gflags_completions.o

all:	main

gflags:	${LIB}/util/gflags-1.1/src/gflags.cc
	$(CC) -c $(CFLAGS) \
        ${LIB}/util/gflags-1.1/src/gflags.cc \
        ${LIB}/util/gflags-1.1/src/gflags_reporting.cc \
        ${LIB}/util/gflags-1.1/src/gflags_completions.cc

main:	$(LOBJECTS) gflags
	$(CC) $(CFLAGS) $(LOBJECTS) $(GFLAGS) -o main $(LDFLAGS)

dbg:	$(LOBJECTS) gflags
	$(CC) $(GDB_CFLAGS) $(LOBJECTS) $(GFLAGS) -o main $(LDFLAGS)

clean:
	-rm -f *.o
	-rm -f main