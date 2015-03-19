.SUFFIXES: .c .u
CC= g++

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

TOPICS_DIR=/home/sgerrish/portal/src1/topics
TOPICS_DIR=/u/sgerrish/src1/topics

GGDB = -ggdb
#GGDB =

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

E_STEP_FLAGS = ${GGDB} -lgsl -lm -lgslcblas -L/sw/lib
E_STEP_SOURCE =  gsl-wrappers.c params.c data.c util.c lda.c main-lda-e-step.c c2_lib.c strutil.c
E_STEP_OBJECTS = gsl-wrappers.o params.o data.o util.o lda.o main-lda-e-step.o c2_lib.o strutil.o

M_STEP_FLAGS = ${GGDB} -lgsl -lm -lgslcblas -L/sw/lib
M_STEP_SOURCE =  gsl-wrappers.c params.c data.c util.c lda.c main-lda-m-step.c c2_lib.c strutil.c
M_STEP_OBJECTS = gsl-wrappers.o params.o data.o util.o lda.o main-lda-m-step.o c2_lib.o strutil.o

AGGREGATE_FLAGS = ${GGDB} -lgsl -lm -lgslcblas -L/sw/lib
AGGREGATE_SOURCE =  gsl-wrappers.c params.c data.c util.c lda.c main-aggregate-ss.c c2_lib.c strutil.c
AGGREGATE_OBJECTS = gsl-wrappers.o params.o data.o util.o lda.o main-aggregate-ss.o c2_lib.o strutil.o

ST_FLAGS = ${GGDB} -lgsl -lm -lgslcblas -L/sw/lib
ST_SOURCE =  ss-lm.c gsl-wrappers.c data.c param.c util.c lda.c lda-seq.c params.c main-fit_seq_topics.c c2_lib.c strutil.c
ST_OBJECTS = ss-lm.o gsl-wrappers.o data.o param.o util.o lda.o lda-seq.o params.o main-fit_seq_topics.o c2_lib.o strutil.o

SD_FLAGS = ${GGDB} -lgsl -lm -lgslcblas -L/sw/lib
SD_SOURCE =  gsl-wrappers.c data.c param.c util.c lda-seq.c lda.c params.c main-fit_seq_docs.c ss-lm.c c2_lib.c strutil.c
SD_OBJECTS = gsl-wrappers.o data.o param.o util.o lda-seq.o lda.o params.o main-fit_seq_docs.o ss-lm.o c2_lib.o strutil.o
GFLAGS = gflags.o gflags_reporting.o gflags_completions.o

gflags:	${LIB}/util/gflags-1.1/src/gflags.cc
	$(CC) -c $(CFLAGS) \
        ${LIB}/util/gflags-1.1/src/gflags.cc \
        ${LIB}/util/gflags-1.1/src/gflags_reporting.cc \
        ${LIB}/util/gflags-1.1/src/gflags_completions.cc
main:	$(LOBJECTS) gflags
	$(CC)  $(CFLAGS) $(LOBJECTS) ${GFLAGS} -o main $(LDFLAGS_PTON)

st:	$(ST_OBJECTS) gflags
	$(CC)  $(CFLAGS) $(ST_OBJECTS) $(GFLAGS) -o fit-seq-topics $(ST_FLAGS)

sd:	$(SD_OBJECTS) gflags
	$(CC)  $(CFLAGS) $(SD_OBJECTS) $(GFLAGS) -o fit-seq-docs $(SD_FLAGS)

lda-e:	$(E_STEP_OBJECTS) gflags
	$(CC)  $(CFLAGS) $(E_STEP_OBJECTS) $(GFLAGS) -o lda-e-step $(E_STEP_FLAGS)

lda-m:	$(M_STEP_OBJECTS) gflags
	$(CC)  $(CFLAGS) $(M_STEP_OBJECTS) $(GFLAGS) -o lda-m-step $(M_STEP_FLAGS)

aggregate:	$(AGGREGATE_OBJECTS) gflags
	$(CC)  $(CFLAGS) $(AGGREGATE_OBJECTS) $(GFLAGS) -o main-aggregate-ss $(AGGREGATE_FLAGS)

all:	gflags main st sd lda-e lda-m aggregate

clean:
	-rm -f *.o
