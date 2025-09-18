# Benchmarks

Time to embed 100 strings of random length from 64 to 512 characters. 

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
```

## Run Bench

```Bash
cargo bench --features cuda
```