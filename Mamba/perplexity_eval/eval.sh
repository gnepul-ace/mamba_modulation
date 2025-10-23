for size in 130m 370m 790m 1.4b 2.8b;
do
    mkdir -p data/mamba-$size
    for tokenized in pile govreport PG19;
    do
            python -u eval.py \
                --tokenized $tokenized \
                --min-tokens 2048 \
                --max-tokens 131072 \
                --dataset-min-tokens 8192 \
                --samples 10 \
                --calib-samples 20 \
                --split test \
                --output-file data/mamba-$size/$tokenized-with_max.csv \
                --truncate \
                --delta_ratio 1.0 \
                -m state-spaces/mamba-$size
    done
done