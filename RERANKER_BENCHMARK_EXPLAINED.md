# Survey Reranker Benchmark Explained

This document explains how `survey_reranker_benchmark.py` evaluates reranker
models on `DummyData_Final 1.csv`, what the scores mean, and how to read the
results.

## Goal

The benchmark answers this question:

> Which reranker is best at putting relevant student-survey answers above
> non-relevant answers for our Dutch/English survey topics?

This is different from an embedding benchmark. An embedding model creates one
vector per text. A reranker directly scores a pair:

```text
query + candidate answer -> relevance score
```

Higher reranker scores should mean that the candidate answer better matches the
query.

## Dataset

The benchmark reuses the same seven open-answer survey columns as
`survey_embedding_benchmark.py`:

| Topic key | Meaning |
|---|---|
| `content_organisation` | Content, structure, and organisation of the programme |
| `professional_practice` | Link with professional practice, careers, internships |
| `teachers` | Teachers, lessons, teacher support |
| `support_mentoring` | Study support, mentoring, personal help |
| `examination_assessment` | Exams, assessment, grading, feedback on marks |
| `engagement_contact` | Engagement, contact moments, communication, availability |
| `special_circumstances` | Studying under special circumstances and flexibility |

Each non-empty answer gets the topic label of the survey column it came from.
That means the labels are useful but not perfect: a student can mention teachers
inside the assessment column, for example.

## Benchmarks Built From The CSV

The script builds three deterministic reranking tests.

### 1. Topic-Prompt Reranking

The query is a Dutch or English topic description.

Example:

```text
Query: Feedback about teachers, teacher support and the quality of teaching.
Relevant candidates: answers from the teachers column
Non-relevant candidates: answers from the other six topic columns
```

This tests whether the reranker can use predefined topic descriptions to find
the right survey answers.

### 2. Answer-To-Answer Reranking

The query is one student answer. Candidates include same-topic positive answers
and other-topic negative answers.

Example:

```text
Query: The teachers explain the material clearly.
Relevant candidates: other answers from the teachers column
Non-relevant candidates: answers from other topic columns
```

This tests whether the reranker can identify similar survey answers.

### 3. Cross-Language Reranking

The query is Dutch and candidates are English, or the query is English and
candidates are Dutch.

This tests whether the reranker can match Dutch and English answers about the
same topic.

The script reports both directions separately:

```text
cross_language_nl_to_en_*
cross_language_en_to_nl_*
```

## Metrics

All core metrics are between `0.0` and `1.0`. Higher is better.

### `top1_topic_acc`

This asks:

> Is the highest-ranked candidate from the same topic as the query?

If `answer_to_answer_top1_topic_acc = 0.80`, then 80% of answer queries had a
same-topic answer at rank 1.

### `map_at_10`

`MAP@10` means Mean Average Precision at 10.

It rewards the reranker when:

1. The top 10 contains many relevant answers.
2. Relevant answers appear early.

This is a very important retrieval-quality metric.

### `ndcg_at_10`

`NDCG@10` means Normalized Discounted Cumulative Gain at 10.

It rewards relevant answers near the top of the ranking. It is often the best
single metric for reranking because rerankers are mostly used to improve the
top of a result list.

### `mrr_at_10`

`MRR@10` means Mean Reciprocal Rank at 10.

It checks how quickly the first relevant answer appears. If the first result is
relevant, the query gets `1.0`. If the first relevant result is rank 2, it gets
`0.5`.

### `recall_at_10`

This asks:

> How many available relevant candidates were found in the top 10?

Recall is useful, but it should not be read alone. A model can have decent
recall while still putting the best results too low.

### `candidate_positive_rate`

This is the positive rate in the candidate pool. It is the rough random
baseline for `top1_topic_acc`.

Example:

```text
candidate_positive_rate = 0.1429
top1_topic_acc = 0.7143
top1_lift_vs_candidate_rate = 5.0
```

That means the model's first result is about five times better than random
selection from the candidate pool.

## Final Score

The script computes one weighted summary score:

```text
reranker_score =
    0.30 * answer_to_answer_ndcg_at_10
  + 0.25 * answer_to_answer_map_at_10
  + 0.20 * cross_language_ndcg_at_10
  + 0.15 * topic_prompt_ndcg_at_10
  + 0.10 * answer_to_answer_top1_topic_acc
```

The score prioritizes answer-to-answer reranking because that is closest to
finding similar survey responses. Cross-language quality is also weighted
strongly because the data contains Dutch and English answers.

Use `reranker_score` for the overall ranking, but always inspect the component
metrics. A model can be strong overall while still being weak in one direction,
for example Dutch-to-English.

## How To Run

Default benchmark:

```bash
.venv/bin/python survey_reranker_benchmark.py --device mps
```

Quick smoke test:

```bash
.venv/bin/python survey_reranker_benchmark.py --quick --device mps
```

The default models are:

```text
Qwen/Qwen3-Reranker-8B
BAAI/bge-reranker-v2-m3
zeroentropy/zerank-2-reranker
jinaai/jina-reranker-v3
```

All models run without an extra instruction prompt so the comparison is strict
apples-to-apples.

For Qwen 8B, start with a small batch size if memory is limited:

```bash
.venv/bin/python survey_reranker_benchmark.py \
  --device mps \
  --batch-size 1 \
  --models Qwen/Qwen3-Reranker-8B
```

BGE is smaller, so it can usually use a larger batch size:

```bash
.venv/bin/python survey_reranker_benchmark.py \
  --device mps \
  --batch-size 16 \
  --models BAAI/bge-reranker-v2-m3
```

## Outputs

The benchmark writes:

| Output | Meaning |
|---|---|
| `domain_reranker_benchmark_log.jsonl` | One JSON row per model run |
| `reranker_benchmark_results/latest_summary_ranking.csv` | Latest model ranking |
| `reranker_benchmark_results/<model>_query_details.csv` | Per-query metrics for deeper inspection |

If a model fails to load or runs out of memory, the failure is written to the
JSONL log and the script continues with the next model.

## How To Read Results

For model selection, look in this order:

1. `reranker_score`
2. `answer_to_answer_ndcg_at_10`
3. `answer_to_answer_map_at_10`
4. `cross_language_ndcg_at_10`
5. `cross_language_nl_to_en_ndcg_at_10`
6. `cross_language_en_to_nl_ndcg_at_10`
7. `topic_prompt_ndcg_at_10`
8. `pairs_per_second`

Recommended interpretation:

```text
High reranker_score + high answer_to_answer_ndcg_at_10
= good general reranker for similar survey answers

High cross_language_ndcg_at_10
= good for mixed Dutch/English analysis

High topic_prompt_ndcg_at_10
= good for predefined theme/topic search

High quality but very low pairs_per_second
= possibly useful, but expensive for production
```

## Benchmark Results

The current full benchmark was run on Apple MPS with seed `42`, no instruction
prompt for any model, `266` ranking cases, and `11,956` scored query-candidate
pairs per model.

| Rank | Model | `reranker_score` | Answer NDCG@10 | Answer MAP@10 | Cross-language NDCG@10 | Topic-prompt NDCG@10 | Pairs/sec |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `zeroentropy/zerank-2-reranker` | 0.4166 | 0.4616 | 0.3096 | 0.4378 | 0.4637 | 32.22 |
| 2 | `Qwen/Qwen3-Reranker-8B` | 0.3980 | 0.4405 | 0.2857 | 0.4104 | 0.4968 | 11.88 |
| 3 | `BAAI/bge-reranker-v2-m3` | 0.3977 | 0.4546 | 0.2979 | 0.4184 | 0.3877 | 159.04 |
| 4 | `jinaai/jina-reranker-v3` | 0.3851 | 0.4444 | 0.2935 | 0.4069 | 0.3803 | 72.36 |

The winner is `zeroentropy/zerank-2-reranker`. It has the highest overall
`reranker_score` and leads the most important quality metrics for this use
case: answer-to-answer `NDCG@10`, answer-to-answer `MAP@10`, and cross-language
`NDCG@10`. That means it was best at ranking similar survey answers near the
top and was also strongest at matching Dutch and English answers about the same
topic.

`Qwen/Qwen3-Reranker-8B` was the strongest model for predefined topic-prompt
search, with the best topic-prompt `NDCG@10`, but it was much slower and did
not beat Zerank on the weighted score. `BAAI/bge-reranker-v2-m3` was almost
tied with Qwen on quality and was by far the fastest model, so it is the best
speed-quality compromise. `jinaai/jina-reranker-v3` ran successfully with its
native reranking interface, but it finished last on this specific benchmark.

## Limitations

The benchmark is useful, but it is not perfect.

The CSV is dummy/synthetic data, so the results should be described as
domain-simulation evidence. The labels come from survey columns, which can
contain some noise. Also, the benchmark uses sampled candidate pools instead of
all possible answer pairs, because scoring every pair with a reranker would be
too expensive.

For a thesis or final report, the best validation strategy is:

1. Use this CSV benchmark to compare reranker candidates.
2. Use the embedding benchmark to choose the retrieval model.
3. Manually inspect a small sample of real reranked survey results.

## Suggested Report Wording

You can describe the benchmark like this:

> Reranker models were evaluated using a deterministic domain-specific
> benchmark built from Dutch and English student survey responses. Each response
> inherited the topic label of its survey question. The benchmark tested
> topic-prompt reranking, answer-to-answer reranking, and cross-language
> Dutch-English reranking. Models were compared with ranking metrics including
> NDCG@10, MAP@10, MRR@10, Recall@10, and top-1 topic accuracy. The final
> weighted reranker score prioritized answer-to-answer and cross-language
> ranking quality because these are closest to the intended survey-analysis
> workflow.
