# Benchmarks

Time to embed 100 strings of random length ranging from 64 to 512 characters. 

**CPU**: 11th Gen Intel(R) Core(TM) i7-11700k @ 3.60GHz <br>
**GPU**: NVIDIA GeForce RTX 3070 Ti, 8GB

```raw
sentence-transformers/all-MiniLM-L6-v2
time:   [41.519 ms 41.657 ms 41.807 ms]

sentence-transformers/all-MiniLM-L12-v2
time:   [80.662 ms 81.077 ms 81.520 ms]

sentence-transformers/LaBSE
time:   [202.80 ms 203.77 ms 204.82 ms]

sentence-transformers/paraphrase-MiniLM-L6-v2
time:   [42.839 ms 43.131 ms 43.435 ms]

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
time:   [175.90 ms 176.52 ms 177.17 ms]

sentence-transformers/paraphrase-multilingual-mpnet-base-v2
time:   [321.68 ms 325.99 ms 330.83 ms]

sentence-transformers/distiluse-base-multilingual-cased-v2
time:   [118.12 ms 120.42 ms 122.86 ms]

intfloat/multilingual-e5-large
time:   [2.0778 s 2.0961 s 2.1156 s]

intfloat/multilingual-e5-base
time:   [702.11 ms 711.10 ms 720.91 ms]

intfloat/multilingual-e5-small
time:   [482.00 ms 490.06 ms 499.09 ms]

sentence-transformers/all-mpnet-base-v2
time:   [407.76 ms 414.05 ms 420.76 ms]

sentence-transformers/paraphrase-mpnet-base-v2
time:   [450.43 ms 459.55 ms 469.56 ms]
```

## Run Bench

```Bash
cargo bench --features cuda
```