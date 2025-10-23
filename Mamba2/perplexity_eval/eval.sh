for size in 130m 370m 780m 1.3b 2.7b;
do
    mkdir -p data/mamba2-$size
    for tokenized in tokenized_proof_pile_test_neox pile govreport PG19;
    do
        python -u eval.py \
            --tokenized $tokenized \
            --min-tokens 2048 \
            --max-tokens 131072 \
            --dataset-min-tokens 8192 \
            --samples 10 \
            --calib-samples 20 \
            --split test \
            --output-file data/mamba2-$size/$tokenized.csv \
            --truncate \
            --delta_ratio 1.0 \
            -m state-spaces/mamba2-$size
    done
done