# Survey Embedding Benchmark Explained

This document explains how `survey_embedding_benchmark.py` evaluates embedding
models, how the final score is calculated, and why the benchmark is a reasonable
basis for choosing an embedding model for student survey answers.

The short version: this benchmark is much more defensible than judging a UMAP
plot or unsupervised clustering score, because it tests the embedding model on
the actual intended task: short Dutch/English student survey responses grouped
by survey topic.

## Goal Of The Benchmark

The goal is to answer this question:

> Which embedding model best represents student survey answers so that answers
> about the same survey topic are close together, even when answers are written
> in Dutch and English?

This is the relevant question for the project because the embedding model will
be used on open student answers from surveys. The benchmark therefore evaluates
whether semantically related survey responses are close in the original
embedding space.

It does not mainly ask whether a visual cluster plot looks nice. A visual plot
can be useful for exploration, but it is weaker evidence for model selection.

## Dataset Construction

The script reads `DummyData_Final 1.csv` with semicolon separation.

It uses seven open-answer survey columns as known topic labels:

| Topic key | Meaning |
|---|---|
| `content_organisation` | Content, structure, and organisation of the programme |
| `professional_practice` | Link with professional practice, careers, internships |
| `teachers` | Teachers, lessons, teacher support |
| `support_mentoring` | Study support, mentoring, personal help |
| `examination_assessment` | Exams, assessment, grading, feedback on marks |
| `engagement_contact` | Engagement, contact moments, communication, availability |
| `special_circumstances` | Studying under special circumstances and flexibility |

For each topic column, the script:

1. Reads all non-empty answers.
2. Cleans whitespace.
3. Stores the answer text.
4. Assigns the topic key as the label.
5. Detects whether the answer is likely Dutch, English, or unknown.

In the current GPU log, the benchmark used:

| Property | Value |
|---|---:|
| Responses | 6,993 |
| Topics | 7 |
| Detected Dutch | 58.19% |
| Detected English | 41.51% |
| Unknown language | 0.30% |

The labels come from the survey columns themselves. For example, an answer from
the teachers feedback column receives the label `teachers`.

## Embedding Step

For each model, the script loads the model with `SentenceTransformer`:

```python
model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
```

It then embeds every survey answer:

```python
model.encode(
    texts,
    batch_size=batch_size,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
```

The important part is `normalize_embeddings=True`. This makes the vector length
equal to 1, so cosine similarity can be computed efficiently as a dot product:

```python
similarities = embeddings @ embeddings.T
```

The diagonal is set to negative infinity:

```python
np.fill_diagonal(similarities, -np.inf)
```

This prevents an answer from retrieving itself as its nearest neighbor.

## Main Retrieval Evaluation

The main test is nearest-neighbor retrieval.

For every answer, the script asks:

> Are the most similar answers about the same survey topic?

For example, if the query answer is from the `teachers` topic, then retrieved
answers from `teachers` count as relevant. Retrieved answers from other topics
count as non-relevant.

This directly tests whether the embedding model places same-topic answers near
each other.

### `top1_topic_acc`

This metric checks only the nearest neighbor.

It answers:

> Is the single most similar answer from the same topic?

A score of `0.76` means that 76% of answers retrieve a same-topic answer as
their closest neighbor.

Because there are seven topics, a rough balanced random baseline is about:

```text
1 / 7 = 0.143
```

So scores around 0.72-0.76 are far above chance.

### `map_at_10`

`MAP@10` means Mean Average Precision at 10.

It looks at the top 10 retrieved answers and rewards the model when:

1. Many of the top 10 answers have the same topic.
2. The relevant answers appear early in the ranking.

This is one of the most important metrics because survey analysis often depends
on retrieving several similar answers, not just one nearest neighbor.

### `ndcg_at_10`

`NDCG@10` means Normalized Discounted Cumulative Gain at 10.

It also evaluates the top 10 results, but it gives more weight to relevant
answers near the top of the list. It is useful when ranking quality matters.

### `mrr_at_10`

`MRR@10` means Mean Reciprocal Rank at 10.

It checks how soon the first same-topic answer appears in the top 10. If the
first result is relevant, the reciprocal rank is 1.0. If the first relevant
result is at rank 2, it is 0.5.

This metric is useful, but it is less complete than `MAP@10` because it only
cares about the first relevant result.

## Cross-Language Retrieval

Because the survey data contains Dutch and English answers, the script also
tests cross-language retrieval.

It asks:

> If the query answer is Dutch, can the model retrieve English answers about the
> same topic?

and:

> If the query answer is English, can the model retrieve Dutch answers about the
> same topic?

The benchmark reports:

| Metric group | Meaning |
|---|---|
| `cross_lang_*` | Retrieval where candidates must be from the opposite language |
| `cross_lang_nl_to_en_*` | Dutch query answers retrieving English answers |
| `cross_lang_en_to_nl_*` | English query answers retrieving Dutch answers |
| `mixed_pool_nl_query_*` | Dutch query answers retrieving from the full mixed-language pool |
| `mixed_pool_en_query_*` | English query answers retrieving from the full mixed-language pool |

These metrics are important because a model may perform well in one language but
fail to align Dutch and English answers in the same semantic space.

For this project, cross-language retrieval is especially relevant if Dutch and
English student responses will be analyzed together.

## Topic Query Evaluation

The script also creates short Dutch and English topic descriptions, such as:

```text
Feedback over docenten, begeleiding door docenten en de kwaliteit van lessen.
Feedback about teachers, teacher support and the quality of teaching.
```

It embeds these topic descriptions and compares every answer against the topic
description embeddings.

The reported metrics include:

| Metric | Meaning |
|---|---|
| `query_acc_dutch` | Accuracy when classifying answers using Dutch topic prompts |
| `query_acc_english` | Accuracy when classifying answers using English topic prompts |
| `query_acc_mixed` | Accuracy using the better score from Dutch or English prompts |
| `query_f1_mixed` | Macro F1 for the mixed-prompt prediction |

This checks whether the embedding model aligns answers with explicit topic
descriptions. It is relevant if the final application will search answers using
topic labels, natural-language queries, or predefined themes.

## Linear Probe Evaluation

The benchmark trains a simple logistic regression classifier on the frozen
embeddings:

```python
LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
```

The data is split into:

```text
75% train
25% test
```

The model itself is not fine-tuned. Only the lightweight classifier is trained.

This evaluates how much topic information is already present in the embeddings.
If a simple classifier can predict the survey topic from the embeddings, then
the embedding space contains useful topic structure.

The most important metric here is:

```text
linear_probe_f1_macro
```

Macro F1 is useful because it gives each topic equal importance instead of
letting larger topics dominate the score.

## K-Means Evaluation

The script runs K-Means with exactly seven clusters:

```python
KMeans(n_clusters=len(TOPICS), n_init=20, random_state=42)
```

It then compares the discovered clusters with the known survey-topic labels.

The reported metrics are:

| Metric | Meaning |
|---|---|
| `kmeans_ari` | Adjusted Rand Index, corrected for chance |
| `kmeans_nmi` | Normalized Mutual Information |
| `kmeans_v_measure` | Harmonic mean of cluster homogeneity and completeness |

This is useful as a secondary check. However, it should not be treated as the
main score because real survey answers can overlap semantically. For example,
an answer about unclear feedback may belong to teachers, exams, or mentoring
depending on the survey question.

## Silhouette Score

The script computes:

```text
silhouette_true_topics_cosine
```

This is calculated on the original embeddings using cosine distance, sampled up
to 2,000 examples.

Silhouette is only a diagnostic metric here. It can be low even when retrieval
performance is good, because survey topics naturally overlap. Student answers
are short, noisy, and often mention multiple issues.

For this reason, silhouette is printed but not included in the final
`domain_score`.

## Final Scoring System

The benchmark creates one final weighted score called:

```text
domain_score
```

The formula is:

```text
domain_score =
    0.30 * map_at_10
  + 0.25 * query_acc_mixed
  + 0.20 * cross_lang_map_at_10
  + 0.15 * linear_probe_f1_macro
  + 0.10 * kmeans_v_measure
```

The weights are:

| Metric | Weight | Why it matters |
|---|---:|---|
| `map_at_10` | 0.30 | Main same-topic retrieval quality |
| `query_acc_mixed` | 0.25 | Ability to match answers to topic descriptions |
| `cross_lang_map_at_10` | 0.20 | Dutch-English semantic alignment |
| `linear_probe_f1_macro` | 0.15 | Topic information available in frozen embeddings |
| `kmeans_v_measure` | 0.10 | Whether unsupervised clusters roughly match topics |

This scoring system is reasonable because most of the weight goes to retrieval
and topic-query behavior, which are closest to the intended use case. Clustering
is included, but it receives only 10% of the final score because clustering is a
weaker and less direct test.

## Example Results From The Current Log

The current `domain_embedding_benchmark_gpu_log.jsonl` contains these complete
model comparisons:

| Model | Domain score | MAP@10 | Cross-lang MAP@10 | Query acc mixed | Linear probe F1 | K-Means V |
|---|---:|---:|---:|---:|---:|---:|
| `Octen/Octen-Embedding-8B` | 0.5560 | 0.6143 | 0.5712 | 0.4506 | 0.7591 | 0.3096 |
| `Qwen/Qwen3-Embedding-8B` | 0.5343 | 0.5902 | 0.5481 | 0.4314 | 0.7595 | 0.2577 |
| `jinaai/jina-embeddings-v5-text-small` | 0.5104 | 0.5658 | 0.5148 | 0.4017 | 0.7414 | 0.2610 |
| `zeroentropy/zembed-1-embedding` | 0.5034 | 0.5564 | 0.5070 | 0.3937 | 0.7475 | 0.2456 |
| `BAAI/bge-m3` | 0.4683 | 0.5366 | 0.4613 | 0.3536 | 0.7127 | 0.1973 |

Based on this benchmark, `Octen/Octen-Embedding-8B` is the strongest model in
the log. It has the best `domain_score`, best `MAP@10`, best cross-language
retrieval score, best mixed query accuracy, and best K-Means V-measure. Qwen is
very close on linear probe F1, but Octen is stronger on the retrieval metrics
that matter most for survey-answer similarity.

`jinaai/jina-embeddings-v5-text-small` performs respectably and is faster than
the larger 8B models in this run, but it does not beat Octen or Qwen on the
overall benchmark. Its result is slightly stronger than zembed on several
retrieval-oriented metrics, while zembed is slightly stronger on linear probe
F1.

## Why This Benchmark Is Defendable

This benchmark is defendable because it is aligned with the actual application.

It uses the same kind of text that the final system must embed: short,
open-ended student survey responses. It evaluates the original embedding space,
not a reduced visualization. It uses known survey topics as labels, so the
evaluation has a clear ground truth. It tests both same-language and
cross-language retrieval, which matters for Dutch/English survey data.

It also combines several perspectives:

1. Nearest-neighbor retrieval: are similar answers close together?
2. Cross-language retrieval: are Dutch and English answers aligned?
3. Topic-query matching: can topic descriptions find the right answers?
4. Linear probing: is topic information encoded in the vectors?
5. K-Means: does unsupervised structure roughly match the topics?

Because the same model performs well across several tests, the conclusion is
stronger than relying on only one metric.

## Limitations And Caveats

The benchmark is useful, but it is not perfect. These are the main limitations
to mention honestly.

### The CSV Appears To Be Dummy/Synthetic Data

The benchmark uses `DummyData_Final 1.csv`. If this data is generated or cleaned
artificially, it may be easier than real student feedback. Synthetic answers can
be more repetitive, more balanced, and more clearly separated by topic.

This does not make the benchmark useless, but it means the result should be
described as domain-simulation evidence rather than final real-world proof.

Best improvement:

```text
Validate the selected model on a small manually reviewed sample of real student
answers.
```

Even 100-300 real labeled answers would make the model choice much stronger.

### Labels Come From Survey Columns

The benchmark assumes that an answer belongs to the topic of the survey column
where it was written. This is reasonable, but not always perfect. A student may
write about teachers in the assessment column, or about support in the special
circumstances column.

This can add label noise. However, all models are tested on the same noisy
labels, so the comparison is still useful.

### Language Detection Is Simple

The script uses marker-word lists to detect Dutch and English. This is fast and
transparent, but it is not a full language detector.

Short answers such as "good", "prima", "none", or "n/a" may be misclassified or
marked as unknown. This mainly affects the cross-language metrics.

Best improvement:

```text
Use a stronger language detector or manually verify a sample of detected
languages.
```

### The Final Score Weights Are Chosen Manually

The `domain_score` weights are sensible, but they are still a design choice.
Different weights could slightly change the ranking if models are close.

This is acceptable if the report explains why retrieval and cross-language
behavior are weighted highly. It is also good practice to show the component
metrics, not only the final score.

### Query Prompts May Influence Results

The topic-query metrics depend on the manually written Dutch and English topic
descriptions. Better or worse wording could change `query_acc_mixed`.

This is why `query_acc_mixed` should be treated as one important signal, not the
only decision criterion.

### Linear Probe Can Overestimate Real Performance

The linear probe trains and tests on splits from the same dataset. If the dummy
data has repeated patterns, the classifier may perform better than it would on
real future survey responses.

This is another reason to prefer retrieval metrics and real-sample validation.

## Mistakes Or Issues Found

I noticed several issues worth fixing or mentioning.

### 1. Current Default Model List Is Narrow

In `survey_embedding_benchmark.py`, the current `DEFAULT_MODELS` list contains
only:

```text
jinaai/jina-embeddings-v5-text-small
```

The previous comparison models are commented out. This is not wrong, but it
means that running the script without `--models` now benchmarks only Jina.

This means:

```text
For a fair model-selection run, pass all candidate models explicitly with
--models, or put all candidate models back into DEFAULT_MODELS.
```

### 2. The README Mentions 70/30 Metrics That The Current Script Does Not Compute

The README mentions metrics such as:

```text
cross_lang_map_at_10_70_30
mixed_pool_map_at_10_70_30
dutch_heavy_cross_language_score
```

Some older log rows contain these fields, but the current
`survey_embedding_benchmark.py` does not compute them anymore.

If the expected real-world usage is 70% Dutch and 30% English, either the script
should add these metrics back, or the README should be updated.

### 3. `topk_indices` Is Defined But Not Used

The helper function `topk_indices` is present in the script but not used. This
is harmless, but it can be removed to keep the benchmark script cleaner.

### 4. Optional MTEB In This Script Does Not Produce A Summary

The `--run-mteb` option runs MTEB tasks, but it does not summarize the MTEB
results into a simple ranking. The separate `mteb_dutch_english_benchmark.py`
script is better for this because it writes a focused summary.

## Recommended Final Validation Strategy

For a report or thesis, the strongest argument is:

1. Use `survey_embedding_benchmark.py` as the primary model-selection benchmark,
   because it matches the student survey use case.
2. Run the focused Dutch/English MTEB benchmark as an external sanity check.
3. If possible, manually inspect or label a small sample of real student
   answers and compare nearest-neighbor quality across the top models.

The survey benchmark alone is enough for a practical engineering decision. For
academic defense, adding MTEB and a small real-data review makes the choice much
harder to challenge.

## Suggested Wording For The Report

You can describe the choice like this:

> The embedding model was selected using a domain-specific benchmark based on
> Dutch and English student survey responses. Each response was embedded in the
> original normalized embedding space and evaluated against known survey-topic
> labels. The benchmark measured same-topic nearest-neighbor retrieval,
> Dutch-English cross-language retrieval, topic-query matching, linear-probe
> classification, and unsupervised cluster agreement. The final weighted domain
> score prioritized retrieval and cross-language behavior because these are most
> relevant for analyzing multilingual open survey answers. On this benchmark,
> `Octen/Octen-Embedding-8B` achieved the strongest overall result among the
> tested models.
