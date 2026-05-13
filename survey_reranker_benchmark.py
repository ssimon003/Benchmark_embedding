"""
Domain benchmark for multilingual student-survey reranker models.

This benchmark uses DummyData_Final 1.csv in the same way as
survey_embedding_benchmark.py: each open-answer survey column becomes a topic
label. Rerankers are then tested on whether they rank same-topic answers above
other-topic answers.

Default run:

    .venv/bin/python survey_reranker_benchmark.py --device mps

Quick smoke test:

    .venv/bin/python survey_reranker_benchmark.py --quick --device mps

Add models later:

    .venv/bin/python survey_reranker_benchmark.py \
      --models Qwen/Qwen3-Reranker-8B BAAI/bge-reranker-v2-m3 another/model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModel
from tqdm.auto import tqdm

from survey_embedding_benchmark import (
    CSV_PATH,
    CSV_SEP,
    TOPICS,
    load_domain_dataset,
    print_device_info,
    resolve_device,
)


DEFAULT_MODELS = [
   # "Qwen/Qwen3-Reranker-8B",
  #  "BAAI/bge-reranker-v2-m3",
   # "zeroentropy/zerank-2-reranker",
    "jinaai/jina-reranker-v3",
]

LOG_PATH = Path("domain_reranker_benchmark_log.jsonl")
OUTPUT_DIR = Path("reranker_benchmark_results")



@dataclass(frozen=True)
class RankingCase:
    benchmark: str
    query_id: str
    query_text: str
    query_topic: str
    query_language: str
    candidate_indices: np.ndarray
    relevance: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reranker models on Dutch/English student survey responses."
    )
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--csv-sep", default=CSV_SEP)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device for reranking. Use auto to prefer Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="CrossEncoder scoring batch size. Large rerankers may need 1-4; smaller models can often use more.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller deterministic sample for a fast smoke test.",
    )
    parser.add_argument(
        "--queries-per-topic",
        type=int,
        default=20,
        help="Answer-to-answer queries sampled per topic.",
    )
    parser.add_argument(
        "--cross-queries-per-topic-language",
        type=int,
        default=8,
        help="Cross-language answer queries sampled per topic and source language.",
    )
    parser.add_argument("--positives-per-query", type=int, default=5)
    parser.add_argument("--negatives-per-query", type=int, default=30)
    parser.add_argument("--cross-positives-per-query", type=int, default=4)
    parser.add_argument("--cross-negatives-per-query", type=int, default=24)
    parser.add_argument(
        "--topic-candidates-per-topic",
        type=int,
        default=40,
        help="Candidates sampled from each topic for topic-prompt reranking.",
    )
    parser.add_argument(
        "--save-query-details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-query metric CSV files.",
    )
    return parser.parse_args()


def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def sample_without_replacement(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or len(values) == 0:
        return np.array([], dtype=np.int64)
    if len(values) <= count:
        return np.array(values, dtype=np.int64)
    return np.array(rng.choice(values, size=count, replace=False), dtype=np.int64)


def balanced_negative_sample(
    pools_by_topic: dict[str, np.ndarray],
    exclude_topic: str,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    topics = [topic.key for topic in TOPICS if topic.key != exclude_topic]
    if total <= 0 or not topics:
        return np.array([], dtype=np.int64)

    rng.shuffle(topics)
    base = total // len(topics)
    remainder = total % len(topics)
    selected = []
    for position, topic in enumerate(topics):
        count = base + int(position < remainder)
        selected.extend(
            sample_without_replacement(
                pools_by_topic.get(topic, np.array([], dtype=np.int64)),
                count,
                rng,
            )
        )
    return np.array(selected, dtype=np.int64)


def build_topic_prompt_cases(
    data: pd.DataFrame,
    rng: np.random.Generator,
    candidates_per_topic: int,
) -> list[RankingCase]:
    cases = []
    all_candidate_indices = []
    for topic in TOPICS:
        topic_indices = data.index[data["topic"] == topic.key].to_numpy(dtype=np.int64)
        all_candidate_indices.extend(sample_without_replacement(topic_indices, candidates_per_topic, rng))

    candidate_indices = np.array(sorted(set(int(i) for i in all_candidate_indices)), dtype=np.int64)
    candidate_topics = data.loc[candidate_indices, "topic"].to_numpy()

    for topic in TOPICS:
        for language, query_text in [("nl", topic.nl_query), ("en", topic.en_query)]:
            relevance = candidate_topics == topic.key
            if not np.any(relevance):
                continue
            cases.append(
                RankingCase(
                    benchmark="topic_prompt",
                    query_id=f"topic_prompt:{language}:{topic.key}",
                    query_text=query_text,
                    query_topic=topic.key,
                    query_language=language,
                    candidate_indices=candidate_indices,
                    relevance=relevance.astype(bool),
                )
            )
    return cases


def build_answer_cases(
    data: pd.DataFrame,
    rng: np.random.Generator,
    queries_per_topic: int,
    positives_per_query: int,
    negatives_per_query: int,
) -> list[RankingCase]:
    cases = []
    pools_by_topic = {
        topic.key: data.index[data["topic"] == topic.key].to_numpy(dtype=np.int64)
        for topic in TOPICS
    }

    for topic in TOPICS:
        query_indices = sample_without_replacement(pools_by_topic[topic.key], queries_per_topic, rng)
        for query_index in query_indices:
            positive_pool = pools_by_topic[topic.key]
            positive_pool = positive_pool[positive_pool != query_index]
            positives = sample_without_replacement(positive_pool, positives_per_query, rng)
            negatives = balanced_negative_sample(pools_by_topic, topic.key, negatives_per_query, rng)
            candidate_indices = np.concatenate([positives, negatives])
            if len(positives) == 0 or len(candidate_indices) == 0:
                continue
            order = rng.permutation(len(candidate_indices))
            candidate_indices = candidate_indices[order]
            relevance = data.loc[candidate_indices, "topic"].to_numpy() == topic.key
            cases.append(
                RankingCase(
                    benchmark="answer_to_answer",
                    query_id=f"answer:{int(query_index)}",
                    query_text=str(data.at[query_index, "text"]),
                    query_topic=topic.key,
                    query_language=str(data.at[query_index, "language"]),
                    candidate_indices=candidate_indices,
                    relevance=relevance.astype(bool),
                )
            )
    return cases


def build_cross_language_cases(
    data: pd.DataFrame,
    rng: np.random.Generator,
    queries_per_topic_language: int,
    positives_per_query: int,
    negatives_per_query: int,
) -> list[RankingCase]:
    cases = []
    known = data[data["language"].isin(["nl", "en"])]
    if known.empty:
        return cases

    pools_by_language_topic: dict[tuple[str, str], np.ndarray] = {}
    for language in ["nl", "en"]:
        for topic in TOPICS:
            pools_by_language_topic[(language, topic.key)] = known.index[
                (known["language"] == language) & (known["topic"] == topic.key)
            ].to_numpy(dtype=np.int64)

    for source_language, target_language in [("nl", "en"), ("en", "nl")]:
        target_pools_by_topic = {
            topic.key: pools_by_language_topic[(target_language, topic.key)]
            for topic in TOPICS
        }
        for topic in TOPICS:
            query_pool = pools_by_language_topic[(source_language, topic.key)]
            query_indices = sample_without_replacement(query_pool, queries_per_topic_language, rng)
            for query_index in query_indices:
                positives = sample_without_replacement(
                    target_pools_by_topic[topic.key],
                    positives_per_query,
                    rng,
                )
                negatives = balanced_negative_sample(
                    target_pools_by_topic,
                    topic.key,
                    negatives_per_query,
                    rng,
                )
                candidate_indices = np.concatenate([positives, negatives])
                if len(positives) == 0 or len(candidate_indices) == 0:
                    continue
                order = rng.permutation(len(candidate_indices))
                candidate_indices = candidate_indices[order]
                relevance = data.loc[candidate_indices, "topic"].to_numpy() == topic.key
                cases.append(
                    RankingCase(
                        benchmark=f"cross_language_{source_language}_to_{target_language}",
                        query_id=f"cross:{source_language}_to_{target_language}:{int(query_index)}",
                        query_text=str(data.at[query_index, "text"]),
                        query_topic=topic.key,
                        query_language=source_language,
                        candidate_indices=candidate_indices,
                        relevance=relevance.astype(bool),
                    )
                )
    return cases


def build_cases(data: pd.DataFrame, args: argparse.Namespace) -> list[RankingCase]:
    rng = np.random.default_rng(args.seed)
    if args.quick:
        queries_per_topic = min(args.queries_per_topic, 4)
        cross_queries = min(args.cross_queries_per_topic_language, 2)
        positives = min(args.positives_per_query, 3)
        negatives = min(args.negatives_per_query, 12)
        cross_positives = min(args.cross_positives_per_query, 2)
        cross_negatives = min(args.cross_negatives_per_query, 10)
        topic_candidates = min(args.topic_candidates_per_topic, 12)
    else:
        queries_per_topic = args.queries_per_topic
        cross_queries = args.cross_queries_per_topic_language
        positives = args.positives_per_query
        negatives = args.negatives_per_query
        cross_positives = args.cross_positives_per_query
        cross_negatives = args.cross_negatives_per_query
        topic_candidates = args.topic_candidates_per_topic

    cases = []
    cases.extend(build_topic_prompt_cases(data, rng, topic_candidates))
    cases.extend(build_answer_cases(data, rng, queries_per_topic, positives, negatives))
    cases.extend(build_cross_language_cases(data, rng, cross_queries, cross_positives, cross_negatives))
    return cases


def average_precision_at_k(relevance: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return math.nan
    relevance = relevance[:k].astype(np.float32)
    denom = min(total_relevant, k)
    if denom == 0:
        return math.nan
    cumulative = np.cumsum(relevance)
    precision_at_rank = cumulative / np.arange(1, len(relevance) + 1)
    return float((precision_at_rank * relevance).sum() / denom)


def ndcg_at_k(relevance: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return math.nan
    relevance = relevance[:k].astype(np.float32)
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    dcg = float((relevance * discounts).sum())
    ideal_count = min(total_relevant, k)
    ideal = float(discounts[:ideal_count].sum())
    return dcg / ideal if ideal > 0 else math.nan


def reciprocal_rank_at_k(relevance: np.ndarray, k: int) -> float:
    relevant_positions = np.flatnonzero(relevance[:k])
    if len(relevant_positions) == 0:
        return 0.0
    return 1.0 / float(relevant_positions[0] + 1)


def case_metrics(case: RankingCase, scores: np.ndarray, k: int = 10) -> dict[str, float | int | str]:
    order = np.argsort(-scores)
    ranked_relevance = case.relevance[order]
    total_relevant = int(np.sum(case.relevance))
    candidate_count = int(len(case.candidate_indices))
    positive_rate = total_relevant / candidate_count if candidate_count else math.nan

    top1 = float(ranked_relevance[0]) if len(ranked_relevance) else math.nan
    recall_denom = total_relevant if total_relevant > 0 else math.nan
    recall_at_k = float(np.sum(ranked_relevance[:k]) / recall_denom) if total_relevant > 0 else math.nan

    return {
        "benchmark": case.benchmark,
        "query_id": case.query_id,
        "query_topic": case.query_topic,
        "query_language": case.query_language,
        "candidate_count": candidate_count,
        "relevant_count": total_relevant,
        "candidate_positive_rate": positive_rate,
        "top1_topic_acc": top1,
        "map_at_10": average_precision_at_k(ranked_relevance, total_relevant, k),
        "ndcg_at_10": ndcg_at_k(ranked_relevance, total_relevant, k),
        "mrr_at_10": reciprocal_rank_at_k(ranked_relevance, k),
        "recall_at_10": recall_at_k,
    }


def aggregate_metric_rows(rows: list[dict[str, float | int | str]], prefix: str) -> dict[str, float | int]:
    if not rows:
        return {
            f"{prefix}_queries": 0,
            f"{prefix}_candidate_positive_rate": math.nan,
            f"{prefix}_top1_topic_acc": math.nan,
            f"{prefix}_map_at_10": math.nan,
            f"{prefix}_ndcg_at_10": math.nan,
            f"{prefix}_mrr_at_10": math.nan,
            f"{prefix}_recall_at_10": math.nan,
            f"{prefix}_top1_lift_vs_candidate_rate": math.nan,
        }

    out: dict[str, float | int] = {f"{prefix}_queries": len(rows)}
    for key in [
        "candidate_positive_rate",
        "top1_topic_acc",
        "map_at_10",
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_10",
    ]:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        out[f"{prefix}_{key}"] = float(np.nanmean(values)) if values else math.nan

    base = out[f"{prefix}_candidate_positive_rate"]
    top1 = out[f"{prefix}_top1_topic_acc"]
    out[f"{prefix}_top1_lift_vs_candidate_rate"] = (
        float(top1 / base) if isinstance(base, float) and base > 0 and isinstance(top1, float) else math.nan
    )
    return out


def weighted_reranker_score(metrics: dict[str, float | int | str]) -> float:
    weights = {
        "answer_to_answer_ndcg_at_10": 0.30,
        "answer_to_answer_map_at_10": 0.25,
        "cross_language_ndcg_at_10": 0.20,
        "topic_prompt_ndcg_at_10": 0.15,
        "answer_to_answer_top1_topic_acc": 0.10,
    }
    total = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            total += float(value) * weight
            used += weight
    return total / used if used else math.nan


def uses_native_jina_reranker(model_name: str) -> bool:
    return model_name == "jinaai/jina-reranker-v3"


def load_native_jina_reranker(model_name: str, device: str, args: argparse.Namespace) -> Any:
    model = AutoModel.from_pretrained(
        model_name,
        dtype="auto",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


def load_reranker(model_name: str, device: str, args: argparse.Namespace) -> Any:
    if uses_native_jina_reranker(model_name):
        return load_native_jina_reranker(model_name, device, args)
    return load_cross_encoder(model_name, device, args)


def load_cross_encoder(model_name: str, device: str, args: argparse.Namespace) -> CrossEncoder:
    return CrossEncoder(
        model_name,
        device=device,
        max_length=args.max_length,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )


def score_with_native_jina(
    model: Any,
    case: RankingCase,
    texts: np.ndarray,
) -> np.ndarray:
    documents = [str(texts[index]) for index in case.candidate_indices]
    results = model.rerank(case.query_text, documents)
    scores = np.full(len(documents), -np.inf, dtype=np.float32)
    for result in results:
        scores[int(result["index"])] = float(result["relevance_score"])
    if np.any(np.isneginf(scores)):
        raise RuntimeError("Native Jina reranker did not return a score for every candidate.")
    return scores


def score_with_cross_encoder(
    model: CrossEncoder,
    case: RankingCase,
    texts: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    pairs = [(case.query_text, str(texts[index])) for index in case.candidate_indices]
    raw_scores = model.predict(
        pairs,
        batch_size=args.batch_size,
        show_progress_bar=False,
        prompt=None,
        convert_to_numpy=True,
    )
    scores = np.asarray(raw_scores, dtype=np.float32)
    if scores.ndim > 1:
        scores = scores[:, -1]
    return scores


def score_cases(
    model: Any,
    model_name: str,
    cases: list[RankingCase],
    texts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], float]:
    per_query_rows = []
    pair_count = sum(len(case.candidate_indices) for case in cases)

    with tqdm(total=pair_count, desc=f"Scoring {safe_name(model_name)}", unit="pair") as progress:
        for case in cases:
            start = time.time()
            if uses_native_jina_reranker(model_name):
                scores = score_with_native_jina(model, case, texts)
            else:
                scores = score_with_cross_encoder(model, case, texts, args)
            elapsed = time.time() - start
            row = case_metrics(case, scores)
            row["score_time_s"] = elapsed
            per_query_rows.append(row)
            progress.update(len(case.candidate_indices))

    return per_query_rows, float(sum(float(row["score_time_s"]) for row in per_query_rows))


def summarize_model(
    model_name: str,
    device: str,
    data: pd.DataFrame,
    cases: list[RankingCase],
    per_query_rows: list[dict[str, float | int | str]],
    score_time_s: float,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    topic_rows = [row for row in per_query_rows if row["benchmark"] == "topic_prompt"]
    answer_rows = [row for row in per_query_rows if row["benchmark"] == "answer_to_answer"]
    cross_rows = [row for row in per_query_rows if str(row["benchmark"]).startswith("cross_language_")]
    nl_to_en_rows = [row for row in cross_rows if row["benchmark"] == "cross_language_nl_to_en"]
    en_to_nl_rows = [row for row in cross_rows if row["benchmark"] == "cross_language_en_to_nl"]

    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "device": device,
        "prompt_mode": "none",
        "prompt_used": "",
        "seed": args.seed,
        "n_responses": int(len(data)),
        "n_topics": int(len(TOPICS)),
        "nl_pct_detected": round(float(np.mean(data["language"].to_numpy() == "nl") * 100), 2),
        "en_pct_detected": round(float(np.mean(data["language"].to_numpy() == "en") * 100), 2),
        "unknown_lang_pct_detected": round(float(np.mean(data["language"].to_numpy() == "unknown") * 100), 2),
        "ranking_cases": int(len(cases)),
        "scored_pairs": int(sum(len(case.candidate_indices) for case in cases)),
        "score_time_s": round(float(score_time_s), 2),
        "pairs_per_second": round(
            float(sum(len(case.candidate_indices) for case in cases) / score_time_s),
            2,
        )
        if score_time_s > 0
        else math.nan,
    }
    metrics.update(aggregate_metric_rows(topic_rows, "topic_prompt"))
    metrics.update(aggregate_metric_rows(answer_rows, "answer_to_answer"))
    metrics.update(aggregate_metric_rows(cross_rows, "cross_language"))
    metrics.update(aggregate_metric_rows(nl_to_en_rows, "cross_language_nl_to_en"))
    metrics.update(aggregate_metric_rows(en_to_nl_rows, "cross_language_en_to_nl"))
    metrics["reranker_score"] = weighted_reranker_score(metrics)
    return metrics


def print_model_summary(metrics: dict[str, float | int | str], log_path: Path) -> None:
    important = [
        "reranker_score",
        "answer_to_answer_ndcg_at_10",
        "answer_to_answer_map_at_10",
        "answer_to_answer_top1_topic_acc",
        "answer_to_answer_top1_lift_vs_candidate_rate",
        "cross_language_ndcg_at_10",
        "cross_language_map_at_10",
        "cross_language_nl_to_en_ndcg_at_10",
        "cross_language_en_to_nl_ndcg_at_10",
        "topic_prompt_ndcg_at_10",
        "topic_prompt_map_at_10",
        "score_time_s",
        "pairs_per_second",
    ]
    print("\nSummary:")
    for key in important:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key:44s} {value:.4f}")
            else:
                print(f"  {key:44s} {value}")
    print(f"\nAppended JSON row to {log_path}")


def print_ranking(results: list[dict[str, float | int | str]]) -> None:
    if not results:
        return
    ranked = sorted(results, key=lambda row: float(row["reranker_score"]), reverse=True)
    print("\n" + "=" * 78)
    print("Ranking by reranker_score")
    print("=" * 78)
    for index, row in enumerate(ranked, start=1):
        print(
            f"{index}. {row['model']}  "
            f"reranker_score={float(row['reranker_score']):.4f}  "
            f"answer_ndcg={float(row['answer_to_answer_ndcg_at_10']):.4f}  "
            f"cross_ndcg={float(row['cross_language_ndcg_at_10']):.4f}  "
            f"topic_ndcg={float(row['topic_prompt_ndcg_at_10']):.4f}"
        )


def write_outputs(
    model_name: str,
    metrics: dict[str, float | int | str],
    per_query_rows: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    if args.save_query_details:
        details_path = args.output_dir / f"{safe_name(model_name)}_query_details.csv"
        pd.DataFrame(per_query_rows).to_csv(details_path, index=False)


def write_failure(model_name: str, device: str, error: BaseException, args: argparse.Namespace) -> None:
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model_name,
        "device": device,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(limit=4),
    }
    with args.log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_current_ranking(results: list[dict[str, float | int | str]], args: argparse.Namespace) -> None:
    if not results:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda row: float(row["reranker_score"]), reverse=True)
    pd.DataFrame(ranked).to_csv(args.output_dir / "latest_summary_ranking.csv", index=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print_device_info(device)

    data = load_domain_dataset(args.csv_path, args.csv_sep, max_rows_per_topic=None).reset_index(drop=True)
    print(f"Loaded {len(data)} responses from {len(TOPICS)} survey topics.")
    print("Topic counts:")
    print(data["topic"].value_counts().sort_index().to_string())
    print("\nDetected language counts:")
    print(data["language"].value_counts().sort_index().to_string())

    cases = build_cases(data, args)
    if not cases:
        raise RuntimeError("No ranking cases could be built from the CSV.")
    pair_count = sum(len(case.candidate_indices) for case in cases)
    print(f"\nBuilt {len(cases)} ranking cases with {pair_count} scored pairs per model.")
    print("Benchmark case counts:")
    print(pd.Series([case.benchmark for case in cases]).value_counts().sort_index().to_string())

    results = []
    texts = data["text"].to_numpy()
    for model_name in args.models:
        print("\n" + "=" * 78)
        print(f"Benchmarking {model_name}")
        print("=" * 78)
        print("Using model without an extra instruction prompt.")

        model = None
        try:
            model = load_reranker(model_name, device, args)
            per_query_rows, score_time_s = score_cases(model, model_name, cases, texts, args)
            metrics = summarize_model(model_name, device, data, cases, per_query_rows, score_time_s, args)
            write_outputs(model_name, metrics, per_query_rows, args)
            print_model_summary(metrics, args.log_path)
            results.append(metrics)
        except Exception as exc:
            print(f"\nFailed to benchmark {model_name}: {type(exc).__name__}: {exc}")
            print(f"Failure details were appended to {args.log_path}")
            write_failure(model_name, device, exc, args)
        finally:
            if model is not None:
                del model
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    write_current_ranking(results, args)
    print_ranking(results)
    if results:
        print(f"\nWrote latest ranking to {args.output_dir / 'latest_summary_ranking.csv'}")
    else:
        print("\nNo successful model runs, so no summary ranking was written.")


if __name__ == "__main__":
    main()
