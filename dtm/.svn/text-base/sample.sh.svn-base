This file provides information about running the Dynamic Topic Model
or the Document Influence Model.  It gives two command-line examples
for running the software and several example commands in R for reading
output files.

Dynamic topic models and the influence model have been implemented
here in c / c++.  This implementation takes two input files:

 (a) foo-mult.dat, which is one-doc-per-line, each line of the form

   unique_word_count index1:count1 index2:count2 ... indexn:counnt

   where each index is an integer corresponding to a unique word.

 (b) foo-seq.dat, which is of the form

   Number_Timestamps
   number_docs_time_1
   ...
   number_docs_time_i
   ...
   number_docs_time_NumberTimestamps

   - The docs in foo-mult.dat should be ordered by date, with the first
     docs from time1, the next from time2, ..., and the last docs from
     timen.

When working with data like this, I've found it helpful to create
the following files:
  - the mult.dat file (described in (a) above)
  - the seq.dat file (described in (b) above)
  - a file with all of the words in the vocabulary, arranged in
    the same order as the word indices
  - a file with information on each of the documents, arranged in
    the same order as the docs in the mult file.

The code creates at least the following files:

 - topic-???-var-e-log-prob.dat: the e-betas (word distributions) for
   topic ??? for all times.  This is in row-major form, i.e.:

  > a = scan("topic-002-var-e-log-prob.dat")
  > b = matrix(a, ncol=10, byrow=TRUE)

  # The probability of term 100 in topic 2 at time 3:
  exp(b[100, 3])

 - gam.dat: The gammas associated with each document.  Divide these by
  the sum for each document to get expected topic mixtures.

  > a = scan("gam.dat")
  > b = matrix(a, ncol=10, byrow=TRUE)
  > rs = rowSums(b)
  > e.theta = b / rs
  # Proportion of topic 5 in document 3:
  e.theta[3, 5]

If you are running this software in "dim" mode to find document
influence, it will also create the following files:

 - influence_time-??? : the influence of documents at time ??? for
  each topic, where time is based on in your -seq.dat file and the
  document index is given by the ordering of documents in the mult
  file.

  For example, in R:
  > a = scan("influence-time-010")
  > b = matrix(a, ncol=10, byrow=TRUE)
  # The influence of the 2nd document on topic 5:
  > b[2, 5]

# Here are some example commands:
# Run the dynamic topic model.
./main \
  --ntopics=20 \
  --mode=fit \
  --rng_seed=0 \
  --initialize_lda=true \
  --corpus_prefix=example/test \
  --outname=example/model_run \
  --top_chain_var=0.005 \
  --alpha=0.01 \
  --lda_sequence_min_iter=6 \
  --lda_sequence_max_iter=20 \
  --lda_max_em_iter=10

# Run the influence model.
./main \
    --mode=fit \
    --rng_seed=0 \
    --model=fixed \
    --initialize_lda=true \
    --corpus_prefix=example/test \
    --outname=example/output \
    --time_resolution=2 \
    --influence_flat_years=5 \
    --top_obs_var=0.5 \
    --top_chain_var=0.005 \
    --sigma_d=0.0001 \
    --sigma_l=0.0001 \
    --alpha=0.01 \
    --lda_sequence_min_iter=6 \
    --lda_sequence_max_iter=20 \
    --save_time=-1 \
    --ntopics=10 \
    --lda_max_em_iter=10

