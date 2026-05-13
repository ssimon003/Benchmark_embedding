# Embedding Model Benchmarking

This project compares embedding models for Dutch-heavy, Dutch/English student-survey responses.

The most important point:

> `bench.py` is useful for exploratory clustering, but it is not a reliable embedding-model benchmark.
>
> `survey_embedding_benchmark.py` and `mteb_dutch_english_benchmark.py` are more reliable because they evaluate embeddings against labeled tasks and use the original embedding space instead of judging a UMAP projection.

## The Simple Explanation

An embedding model should put texts with similar meaning close together.

There are two ways to test that:

1. **Ask a real question with a known answer.**
   Example: "Given this Dutch survey answer, can the model retrieve other answers about the same topic?"

2. **Look at a picture of the data and see if it forms nice clusters.**
   This is what `bench.py` mostly does.

The first approach is stronger. It checks whether the model solves the kind of task we care about. The second approach can be misleading because dimensionality reduction and clustering can make the data look cleaner than it really is.

## Benchmark Files

| File | Purpose | Reliability |
|---|---|---|
| `bench.py` | Exploratory UMAP + HDBSCAN clustering on one survey column | Weak as a model benchmark |
| `survey_embedding_benchmark.py` | Domain benchmark on the survey CSV using survey-question topics as labels | Stronger for this project |
| `survey_reranker_benchmark.py` | Domain benchmark for reranker models using topic-prompt, answer-to-answer, and cross-language ranking tasks | Stronger for choosing rerankers |
| `mteb_dutch_english_benchmark.py` | Public Dutch/Dutch-English MTEB benchmark | Stronger external sanity check |

## Reranker Benchmark

Use `survey_reranker_benchmark.py` when you want to compare reranker models such
as `Qwen/Qwen3-Reranker-8B`, `BAAI/bge-reranker-v2-m3`,
`zeroentropy/zerank-2-reranker`, and `jinaai/jina-reranker-v3`.

Default run:

```bash
.venv/bin/python survey_reranker_benchmark.py --device mps
```

Quick smoke test:

```bash
.venv/bin/python survey_reranker_benchmark.py --quick --device mps
```

The benchmark writes `domain_reranker_benchmark_log.jsonl` and
`reranker_benchmark_results/latest_summary_ranking.csv`. See
`RERANKER_BENCHMARK_EXPLAINED.md` for the score explanation.

## Why `bench.py` Is Weaker

`bench.py` does this:

```text
survey answers -> embedding model -> UMAP -> HDBSCAN -> silhouette/DBI/Calinski-Harabasz
```

That sounds reasonable, but it has several technical problems.

### 1. It Scores the UMAP Space, Not the Embedding Space

The model produces high-dimensional embeddings. Then `bench.py` compresses them with UMAP:

```python
reduced = reducer.fit_transform(embeddings)
```

The metrics are computed on `reduced`, not on the original embeddings:

```python
clean_emb = reduced[mask]
sil = silhouette_score(clean_emb, clean_labs, metric="euclidean")
```

This means the score mostly answers:

> "Did UMAP and HDBSCAN create visually/separably nice clusters?"

It does **not** directly answer:

> "Which embedding model has the best semantic representation?"

UMAP can stretch, compress, and reshape neighborhoods. With `min_dist=0.0`, UMAP is explicitly encouraged to make points clump together. That can inflate silhouette-like scores.

### 2. It Uses Internal Clustering Metrics Without Ground Truth

Silhouette, Davies-Bouldin, and Calinski-Harabasz are **internal** clustering metrics.

They only look at distances between points and clusters. They do not know whether a cluster is semantically correct.

For example, a cluster can look compact but still mix several real-world themes:

```text
"teachers give unclear feedback"
"exam feedback is unclear"
"study coach gives unclear feedback"
```

Distance-based metrics may like this cluster because the wording is similar, but for survey analysis those may belong to different actionable topics.

### 3. HDBSCAN Noise Is Removed Before Scoring

`bench.py` removes noise points:

```python
mask = labels != -1
clean_emb = reduced[mask]
clean_labs = labels[mask]
```

This makes the score cleaner because the difficult points are excluded.

That is not necessarily wrong for cluster inspection, but it is risky for model ranking. A model that marks more difficult responses as noise can appear better because only its easiest points are scored.

### 4. The Metric Can Reward the Pipeline Instead of the Model

The benchmark includes many non-model choices:

```text
UMAP dimensions
UMAP n_neighbors
UMAP min_dist
HDBSCAN min_cluster_size
HDBSCAN min_samples
HDBSCAN cluster_selection_method
```

Changing these can change the ranking, even with the same embeddings.

That means `bench.py` evaluates:

```text
embedding model + UMAP settings + HDBSCAN settings + noise filtering
```

not just:

```text
embedding model
```

### 5. It Uses Only One Survey Column

`bench.py` benchmarks only this column:

```python
ANSWER_COL = "Would you like to give your institution any other feedback on the content and organisation of your course programme?"
```

But the actual problem is broader: student responses across content, careers, teachers, support, assessment, contact, and special circumstances.

A model can perform well on one column and worse on the full survey task.

### 6. Small Logging Bug

`bench.py` logs scores like this:

```python
"silhouette": round(sil, 4) if sil else None
```

If a valid score is `0.0`, it will be logged as `None`.

The safer check is:

```python
"silhouette": round(sil, 4) if sil is not None else None
```

This is not the biggest issue, but it is another reason not to treat the old log as a final benchmark.

## Why the Survey Benchmark Is More Reliable

`survey_embedding_benchmark.py` is more reliable because it tests the model against known survey-topic labels.

It uses all seven survey columns as labels:

```text
content and organisation
professional practice / careers
teachers
support / mentoring
examination and assessment
engagement and contact
special circumstances
```

Instead of asking "do clusters look compact?", it asks practical questions:

> "For this answer, are the nearest neighbors about the same survey topic?"

> "Can Dutch responses retrieve English responses about the same topic?"

> "Can English queries find Dutch answers?"

> "Can a simple classifier use these frozen embeddings to predict the topic?"

These are closer to the real product need.

### Metrics Used

The survey benchmark uses the original normalized embeddings and cosine similarity.

Important metrics include:

| Metric | Meaning |
|---|---|
| `map_at_10` | Do the top 10 nearest neighbors contain same-topic answers? |
| `ndcg_at_10` | Are same-topic answers ranked near the top? |
| `cross_lang_map_at_10_70_30` | Dutch-English retrieval, weighted for 70% Dutch / 30% English usage |
| `mixed_pool_map_at_10_70_30` | Retrieval from a mixed Dutch/English pool |
| `linear_probe_f1_macro` | How much topic information is present in the embeddings? |
| `kmeans_v_measure` | How well simple clustering matches the known topics? |

This is technically stronger because it evaluates the original embedding geometry directly.

### Why the 70/30 Weight Matters

The real usage is expected to be about:

```text
70% Dutch
30% English
```

The CSV's detected split may differ, especially because it is dummy data. Therefore the benchmark computes Dutch-heavy weighted scores such as:

```text
cross_lang_map_at_10_70_30
dutch_heavy_cross_language_score
```

This prevents the benchmark from accidentally optimizing for the language mix of the dummy file instead of the expected real-world mix.

## Why MTEB Is More Reliable Than `bench.py`

MTEB is a public benchmark suite for text embeddings.

`mteb_dutch_english_benchmark.py` runs a focused Dutch/Dutch-English MTEB subset, including tasks such as:

```text
Dutch semantic similarity
Dutch-English semantic similarity
Dutch-English bitext mining
Dutch pair classification
Dutch retrieval
Dutch classification
Dutch clustering
```

MTEB is stronger than `bench.py` because:

1. It uses public labeled datasets.
2. It evaluates multiple task types, not just clustering.
3. It uses task-appropriate metrics.
4. It tests general Dutch and cross-language ability outside the dummy CSV.
5. It reduces the risk that the chosen model only looks good on generated survey data.

In simple terms:

> The survey benchmark tells us whether a model fits our local survey-like task.
>
> MTEB tells us whether the model is generally strong on real public Dutch and Dutch-English tasks.
>
> `bench.py` mostly tells us whether one clustering pipeline produced neat-looking clusters.

## Important Caveat About the CSV

`DummyData_Final 1.csv` was generated by another AI, so it is not perfect evidence.

The labels are useful because each survey column represents a known topic. But because the data is synthetic, the benchmark may overestimate performance. AI-generated text can be cleaner, more balanced, and more repetitive than real student feedback.

That is why MTEB matters here.

If the survey benchmark and MTEB agree, confidence increases.

If they disagree, we should not blindly trust the dummy CSV. We should inspect:

```text
which task types disagree
which language direction disagrees
whether Dutch-English retrieval or Dutch-only retrieval is the problem
whether the real use case is closer to survey retrieval, classification, or clustering
```

## Best Reliability Strategy

For the real project, use evidence in this order:

1. **Small human-labeled sample from real student responses**
   Best evidence. Even 100-300 manually labeled examples can be very valuable.

2. **Focused MTEB Dutch/Dutch-English benchmark**
   Best public external check.

3. **Survey CSV domain benchmark**
   Useful domain simulation, but limited because the data is synthetic.

4. **`bench.py` clustering diagnostics**
   Useful for exploration and visualization, not final model selection.

## How to Run the Benchmarks

### Survey/domain benchmark

Run on Apple Silicon GPU through PyTorch MPS:

```bash
.venv/bin/python survey_embedding_benchmark.py --device mps --log-path domain_embedding_benchmark_gpu_log.jsonl
```

The script prints the device:

```text
Device: mps | torch=... | mps_available=True
```

### Focused Dutch/English MTEB benchmark

Run the quicker MTEB subset first:

```bash
MTEB_CACHE=.mteb-cache HF_DATASETS_CACHE=.hf-datasets-cache \
.venv/bin/python mteb_dutch_english_benchmark.py --device mps --quick
```

Run the full focused suite:

```bash
MTEB_CACHE=.mteb-cache HF_DATASETS_CACHE=.hf-datasets-cache \
.venv/bin/python mteb_dutch_english_benchmark.py --device mps --overwrite
```

Results are written to:

```text
mteb_dutch_english_summary.jsonl
mteb_dutch_english_results/
```

### Old exploratory clustering benchmark

```bash
.venv/bin/python bench.py
```

Use this only for exploratory clustering diagnostics.

## How to Interpret Final Results

If a model wins on:

```text
survey Dutch-heavy cross-language score
MTEB Dutch/Dutch-English weighted score
manual review of nearest neighbors
```

then it is a strong candidate.

If a model wins only on `bench.py`, that is weak evidence. It may simply work better with the chosen UMAP/HDBSCAN settings.

For unlabeled real data, the safest final validation is to sample real responses and manually judge nearest neighbors:

```text
For each model:
1. Embed real responses.
2. Pick random Dutch and English responses.
3. Retrieve top 5-10 nearest neighbors.
4. Ask humans whether the neighbors are semantically useful.
5. Compare win rate across models.
```

That test is small, practical, and directly aligned with the real use case.
