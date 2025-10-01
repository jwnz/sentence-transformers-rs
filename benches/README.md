# Benchmarks

Time to embed 32 strings of ~512 tokens.

**CPU**: 11th Gen Intel(R) Core(TM) i7-11700k @ 3.60GHz <br>
**GPU**: NVIDIA GeForce RTX 3070 Ti, 8GB

```raw
sentence-transformers/all-MiniLM-L6-v2
time:   [54.152 ms 54.260 ms 54.401 ms]

sentence-transformers/all-MiniLM-L12-v2
time:   [46.374 ms 47.109 ms 47.939 ms]

sentence-transformers/LaBSE
time:   [228.76 ms 233.54 ms 238.68 ms]

sentence-transformers/paraphrase-MiniLM-L6-v2
time:   [26.161 ms 26.359 ms 26.615 ms]

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
time:   [46.853 ms 46.977 ms 47.131 ms]

sentence-transformers/paraphrase-multilingual-mpnet-base-v2
time:   [111.21 ms 113.64 ms 116.26 ms]

sentence-transformers/distiluse-base-multilingual-cased-v2
time:   [61.013 ms 62.242 ms 63.614 ms]

BAAI/bge-small-en-v1.5
time:   [274.06 ms 277.56 ms 281.81 ms]

intfloat/multilingual-e5-large
time:   [1.5825 s 1.5903 s 1.6004 s]

intfloat/multilingual-e5-base
time:   [516.95 ms 521.50 ms 527.38 ms]

intfloat/multilingual-e5-small
time:   [276.75 ms 278.61 ms 281.15 ms]

sentence-transformers/all-mpnet-base-v2
time:   [852.03 ms 856.32 ms 861.43 ms]

sentence-transformers/paraphrase-mpnet-base-v2
time:   [1.2077 s 1.2122 s 1.2179 s]

BAAI/bge-base-en-v1.5   
time:   [504.83 ms 507.12 ms 510.87 ms]

```

## Run Bench

```Bash
cargo bench --features cuda
```