#!/usr/bin/env bash

set -x

LAMBDAS=(1e-1)
F=(10)
MAX_ITER=(10)

for l in "${LAMBDAS[@]}"
do
  for f in "${F[@]}"
  do
    for m in "${MAX_ITER[@]}"
    do
      `rm -rf userMatrix_* itemMatrix_* training test`
      bin/run-example ml.MovieLensALS --rank $f --numBlocks 1 --maxIter $m --regParam $l --ratings data/mllib/als/sample_movielens_ratings.txt \
        --movies data/mllib/als/sample_movielens_movies.txt > ${f}_${l}_${m}.txt 2> ${f}_${l}_${m}.err
    done
  done
done

