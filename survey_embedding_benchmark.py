"""
Domain benchmark for multilingual student-survey response embeddings.

This file is intentionally separate from bench.py. It evaluates embedding models
for the actual use case: short Dutch/English survey answers that should group by
the survey topic they answer.

Default benchmark, no MTEB dependency required:

    .venv/bin/python survey_embedding_benchmark.py

Optional MTEB run, after installing mteb:

    .venv/bin/python survey_embedding_benchmark.py --run-mteb --mteb-languages nld

Why this is different from bench.py:
- It uses the original normalized embedding space, not UMAP output.
- It uses known survey-question columns as labels.
- It includes Dutch/English cross-language retrieval checks.
- Silhouette is only a weak diagnostic, not the main score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split


CSV_PATH = Path("DummyData_Final 1.csv")
CSV_SEP = ";"
LOG_PATH = Path("domain_embedding_benchmark_log.jsonl")

DEFAULT_MODELS = [
    "zeroentropy/zembed-1-embedding",
    "BAAI/bge-m3",
]


@dataclass(frozen=True)
class Topic:
    key: str
    column: str
    nl_query: str
    en_query: str


TOPICS = [
    Topic(
        key="content_organisation",
        column="Would you like to give your institution any other feedback on the content and organisation of your course programme?",
        nl_query="Feedback over de inhoud, structuur en organisatie van het studieprogramma.",
        en_query="Feedback about the content, structure and organisation of the course programme.",
    ),
    Topic(
        key="professional_practice",
        column="Would you like to give your institution any other feedback on the link with professional practice / professional careers?",
        nl_query="Feedback over de aansluiting met beroepspraktijk, stages en loopbaanvoorbereiding.",
        en_query="Feedback about links with professional practice, internships and career preparation.",
    ),
    Topic(
        key="teachers",
        column="Would you like to give your institution any other feedback on the teachers on your course programme?",
        nl_query="Feedback over docenten, begeleiding door docenten en de kwaliteit van lessen.",
        en_query="Feedback about teachers, teacher support and the quality of teaching.",
    ),
    Topic(
        key="support_mentoring",
        column="Would you like to give your institution any other feedback on support/mentoring?",
        nl_query="Feedback over studiebegeleiding, mentoring, ondersteuning en persoonlijke hulp.",
        en_query="Feedback about study support, mentoring, guidance and personal help.",
    ),
    Topic(
        key="examination_assessment",
        column="Would you like to give your institution any other feedback on examination and assessment?",
        nl_query="Feedback over toetsen, examens, beoordeling, rubrics en feedback op cijfers.",
        en_query="Feedback about exams, assessment, grading, rubrics and feedback on marks.",
    ),
    Topic(
        key="engagement_contact",
        column="Would you like to give your institution any other feedback on engagement and contact?",
        nl_query="Feedback over betrokkenheid, contactmomenten, communicatie en bereikbaarheid.",
        en_query="Feedback about engagement, contact moments, communication and availability.",
    ),
    Topic(
        key="special_circumstances",
        column="Would you like to give your institution any other feedback on studying under special circumstances?",
        nl_query="Feedback over studeren met bijzondere omstandigheden, flexibiliteit en extra regelingen.",
        en_query="Feedback about studying under special circumstances, flexibility and extra arrangements.",
    ),
]


DUTCH_MARKERS = {
    "aan",
    "als",
    "bij",
    "dat",
    "de",
    "deze",
    "die",
    "dit",
    "docenten",
    "een",
    "en",
    "er",
    "geen",
    "het",
    "hulp",
    "ik",
    "in",
    "is",
    "kan",
    "kunnen",
    "maar",
    "meer",
    "met",
    "niet",
    "nog",
    "om",
    "onderwijs",
    "op",
    "over",
    "te",
    "van",
    "voor",
    "wat",
    "we",
    "zou",
    "zouden",
}

ENGLISH_MARKERS = {
    "about",
    "and",
    "are",
    "be",
    "but",
    "by",
    "can",
    "could",
    "for",
    "from",
    "good",
    "help",
    "i",
    "in",
    "is",
    "it",
    "more",
    "need",
    "not",
    "of",
    "on",
    "should",
    "that",
    "the",
    "this",
    "to",
    "we",
    "with",
    "would",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on Dutch/English student survey responses."
    )
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--csv-sep", default=CSV_SEP)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device for embedding. Use auto to prefer Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--dutch-weight",
        type=float,
        default=0.70,
        help="Business weight for Dutch-heavy mixed-language usage. English gets 1 - this value.",
    )
    parser.add_argument(
        "--max-rows-per-topic",
        type=int,
        default=None,
        help="Optional quick-test limit per survey topic.",
    )
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    parser.add_argument("--skip-linear-probe", action="store_true")
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument("--run-mteb", action="store_true")
    parser.add_argument(
        "--mteb-languages",
        nargs="+",
        default=["nld"],
        help='MTEB ISO 639-3 language codes. Use "nld" for Dutch, or "nld eng" for both.',
    )
    parser.add_argument("--mteb-max-tasks", type=int, default=8)
    parser.add_argument("--mteb-output-folder", type=Path, default=Path("mteb_results"))
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if mps_available:
            return "mps"
        if cuda_available:
            return "cuda"
        return "cpu"

    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "Apple MPS was requested, but PyTorch reports it is not available. "
            f"mps_built={torch.backends.mps.is_built()}, "
            f"mps_available={torch.backends.mps.is_available()}."
        )
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def print_device_info(device: str) -> None:
    print(
        "Device: "
        f"{device} | torch={torch.__version__} | "
        f"mps_built={torch.backends.mps.is_built()} | "
        f"mps_available={torch.backends.mps.is_available()} | "
        f"cuda_available={torch.cuda.is_available()}"
    )


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def detect_language(text: str) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())
    if not tokens:
        return "unknown"
    dutch = sum(token in DUTCH_MARKERS for token in tokens)
    english = sum(token in ENGLISH_MARKERS for token in tokens)
    if dutch >= english + 1:
        return "nl"
    if english >= dutch + 1:
        return "en"
    return "unknown"


def load_domain_dataset(
    csv_path: Path,
    csv_sep: str,
    max_rows_per_topic: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=csv_sep)

    missing = [topic.column for topic in TOPICS if topic.column not in df.columns]
    if missing:
        joined = "\n- ".join(missing)
        raise ValueError(f"Missing expected survey columns:\n- {joined}")

    records = []
    for topic in TOPICS:
        values = df[topic.column].map(clean_text)
        values = values[values != ""]
        if max_rows_per_topic is not None:
            values = values.head(max_rows_per_topic)
        for row_id, text in values.items():
            records.append(
                {
                    "row_id": int(row_id),
                    "topic": topic.key,
                    "text": text,
                    "language": detect_language(text),
                }
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise ValueError("No survey responses found after cleaning.")
    return out


def load_model_and_encode(
    model_name: str,
    texts: Iterable[str],
    batch_size: int,
    device: str,
) -> tuple[SentenceTransformer, np.ndarray, float]:
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    start = time.time()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return model, embeddings.astype(np.float32, copy=False), time.time() - start


def topk_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    n = similarities.shape[0]
    k = min(k, n - 1)
    part = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(similarities, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


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


def ranking_metrics_from_candidates(
    similarities: np.ndarray,
    labels: np.ndarray,
    query_indices: np.ndarray,
    candidates_for_query,
    k: int = 10,
) -> dict[str, float | int]:
    ap_values = []
    ndcg_values = []
    reciprocal_ranks = []
    top1_values = []

    for i in query_indices:
        candidate_indices = candidates_for_query(int(i))
        if len(candidate_indices) == 0:
            continue

        candidate_labels = labels[candidate_indices]
        relevance_all = candidate_labels == labels[i]
        total_relevant = int(np.sum(relevance_all))
        if total_relevant == 0:
            continue

        scores = similarities[i, candidate_indices]
        limit = min(k, len(candidate_indices))
        order = np.argpartition(-scores, kth=limit - 1)[:limit]
        order = order[np.argsort(-scores[order])]
        relevance = relevance_all[order]

        top1_values.append(float(relevance[0]))
        ap_values.append(average_precision_at_k(relevance, total_relevant, k))
        ndcg_values.append(ndcg_at_k(relevance, total_relevant, k))

        relevant_positions = np.flatnonzero(relevance)
        if len(relevant_positions):
            reciprocal_ranks.append(1.0 / float(relevant_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

    return {
        "queries": int(len(ap_values)),
        "top1_topic_acc": float(np.nanmean(top1_values)) if top1_values else math.nan,
        "map_at_10": float(np.nanmean(ap_values)),
        "ndcg_at_10": float(np.nanmean(ndcg_values)),
        "mrr_at_10": float(np.nanmean(reciprocal_ranks)) if reciprocal_ranks else math.nan,
    }


def label_retrieval_metrics(similarities: np.ndarray, labels: np.ndarray, k: int = 10) -> dict[str, float | int]:
    all_indices = np.arange(len(labels))

    def candidates_for_query(i: int) -> np.ndarray:
        return all_indices[all_indices != i]

    metrics = ranking_metrics_from_candidates(similarities, labels, all_indices, candidates_for_query, k)
    metrics.pop("queries", None)
    return metrics


def add_prefix(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def weighted_language_value(
    metrics: dict[str, float | int],
    nl_key: str,
    en_key: str,
    dutch_weight: float,
) -> float:
    english_weight = 1.0 - dutch_weight
    nl_value = metrics.get(nl_key)
    en_value = metrics.get(en_key)
    total = 0.0
    weight = 0.0
    if isinstance(nl_value, (int, float)) and not math.isnan(float(nl_value)):
        total += float(nl_value) * dutch_weight
        weight += dutch_weight
    if isinstance(en_value, (int, float)) and not math.isnan(float(en_value)):
        total += float(en_value) * english_weight
        weight += english_weight
    return total / weight if weight else math.nan


def language_retrieval_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
    dutch_weight: float,
    k: int = 10,
) -> dict[str, float | int]:
    all_indices = np.arange(len(labels))
    known = np.isin(languages, ["nl", "en"])
    known_indices = np.flatnonzero(known)

    def opposite_language_candidates(i: int) -> np.ndarray:
        return np.flatnonzero(known & (languages != languages[i]))

    aggregate = ranking_metrics_from_candidates(
        similarities,
        labels,
        known_indices,
        opposite_language_candidates,
        k,
    )

    out: dict[str, float | int] = {
        "cross_lang_queries": aggregate["queries"],
        "cross_lang_top1_topic_acc": aggregate["top1_topic_acc"],
        "cross_lang_map_at_10": aggregate["map_at_10"],
        "cross_lang_ndcg_at_10": aggregate["ndcg_at_10"],
        "cross_lang_mrr_at_10": aggregate["mrr_at_10"],
    }

    for source, target in [("nl", "en"), ("en", "nl")]:
        query_indices = np.flatnonzero(languages == source)
        target_indices = np.flatnonzero(languages == target)

        def candidates_for_direction(_i: int, target_indices: np.ndarray = target_indices) -> np.ndarray:
            return target_indices

        directional = ranking_metrics_from_candidates(
            similarities,
            labels,
            query_indices,
            candidates_for_direction,
            k,
        )
        out.update(add_prefix(f"cross_lang_{source}_to_{target}", directional))

    for source in ["nl", "en"]:
        query_indices = np.flatnonzero(languages == source)

        def mixed_pool_candidates(i: int) -> np.ndarray:
            return all_indices[all_indices != i]

        mixed_pool = ranking_metrics_from_candidates(
            similarities,
            labels,
            query_indices,
            mixed_pool_candidates,
            k,
        )
        out.update(add_prefix(f"mixed_pool_{source}_query", mixed_pool))

    out["cross_lang_map_at_10_70_30"] = weighted_language_value(
        out,
        "cross_lang_nl_to_en_map_at_10",
        "cross_lang_en_to_nl_map_at_10",
        dutch_weight,
    )
    out["cross_lang_top1_topic_acc_70_30"] = weighted_language_value(
        out,
        "cross_lang_nl_to_en_top1_topic_acc",
        "cross_lang_en_to_nl_top1_topic_acc",
        dutch_weight,
    )
    out["mixed_pool_map_at_10_70_30"] = weighted_language_value(
        out,
        "mixed_pool_nl_query_map_at_10",
        "mixed_pool_en_query_map_at_10",
        dutch_weight,
    )
    out["mixed_pool_top1_topic_acc_70_30"] = weighted_language_value(
        out,
        "mixed_pool_nl_query_top1_topic_acc",
        "mixed_pool_en_query_top1_topic_acc",
        dutch_weight,
    )

    return out


def topic_query_metrics(
    model: SentenceTransformer,
    answer_embeddings: np.ndarray,
    labels: np.ndarray,
    languages: np.ndarray,
    batch_size: int,
    dutch_weight: float,
) -> dict[str, float]:
    topic_keys = np.array([topic.key for topic in TOPICS])
    nl_queries = [topic.nl_query for topic in TOPICS]
    en_queries = [topic.en_query for topic in TOPICS]

    nl_emb = model.encode(
        nl_queries,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    en_emb = model.encode(
        en_queries,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    nl_pred = topic_keys[np.argmax(answer_embeddings @ nl_emb.T, axis=1)]
    en_pred = topic_keys[np.argmax(answer_embeddings @ en_emb.T, axis=1)]
    mixed_scores = np.maximum(answer_embeddings @ nl_emb.T, answer_embeddings @ en_emb.T)
    mixed_pred = topic_keys[np.argmax(mixed_scores, axis=1)]

    metrics = {
        "query_acc_dutch": float(accuracy_score(labels, nl_pred)),
        "query_f1_dutch": float(f1_score(labels, nl_pred, average="macro")),
        "query_acc_english": float(accuracy_score(labels, en_pred)),
        "query_f1_english": float(f1_score(labels, en_pred, average="macro")),
        "query_acc_mixed": float(accuracy_score(labels, mixed_pred)),
        "query_f1_mixed": float(f1_score(labels, mixed_pred, average="macro")),
    }

    for lang in ["nl", "en"]:
        mask = languages == lang
        if not np.any(mask):
            continue
        metrics[f"query_acc_dutch_prompt_on_{lang}_answers"] = float(accuracy_score(labels[mask], nl_pred[mask]))
        metrics[f"query_acc_english_prompt_on_{lang}_answers"] = float(accuracy_score(labels[mask], en_pred[mask]))
        metrics[f"query_acc_mixed_prompt_on_{lang}_answers"] = float(accuracy_score(labels[mask], mixed_pred[mask]))

    metrics["query_acc_same_language_prompt_70_30"] = weighted_language_value(
        metrics,
        "query_acc_dutch_prompt_on_nl_answers",
        "query_acc_english_prompt_on_en_answers",
        dutch_weight,
    )
    metrics["query_acc_cross_language_prompt_70_30"] = weighted_language_value(
        metrics,
        "query_acc_english_prompt_on_nl_answers",
        "query_acc_dutch_prompt_on_en_answers",
        dutch_weight,
    )
    metrics["query_acc_mixed_prompt_70_30"] = weighted_language_value(
        metrics,
        "query_acc_mixed_prompt_on_nl_answers",
        "query_acc_mixed_prompt_on_en_answers",
        dutch_weight,
    )
    return metrics


def linear_probe_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return {
        "linear_probe_acc": float(accuracy_score(y_test, pred)),
        "linear_probe_f1_macro": float(f1_score(y_test, pred, average="macro")),
    }


def kmeans_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    clusters = KMeans(
        n_clusters=len(TOPICS),
        n_init=20,
        random_state=42,
    ).fit_predict(embeddings)
    return {
        "kmeans_ari": float(adjusted_rand_score(labels, clusters)),
        "kmeans_nmi": float(normalized_mutual_info_score(labels, clusters)),
        "kmeans_v_measure": float(v_measure_score(labels, clusters)),
    }


def sampled_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return math.nan
    sample_size = min(2000, len(labels))
    return float(
        silhouette_score(
            embeddings,
            labels,
            metric="cosine",
            sample_size=sample_size,
            random_state=42,
        )
    )


def weighted_domain_score(metrics: dict[str, float | int | str]) -> float:
    weights = {
        "map_at_10": 0.30,
        "query_acc_mixed": 0.25,
        "cross_lang_map_at_10": 0.20,
        "linear_probe_f1_macro": 0.15,
        "kmeans_v_measure": 0.10,
    }
    total = 0.0
    used_weight = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            total += float(value) * weight
            used_weight += weight
    return total / used_weight if used_weight else math.nan


def dutch_heavy_cross_language_score(metrics: dict[str, float | int | str]) -> float:
    weights = {
        "cross_lang_map_at_10_70_30": 0.35,
        "mixed_pool_map_at_10_70_30": 0.25,
        "query_acc_cross_language_prompt_70_30": 0.20,
        "query_acc_same_language_prompt_70_30": 0.10,
        "map_at_10": 0.05,
        "linear_probe_f1_macro": 0.05,
    }
    total = 0.0
    used_weight = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            total += float(value) * weight
            used_weight += weight
    return total / used_weight if used_weight else math.nan


def run_domain_benchmark(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    device = resolve_device(args.device)
    print_device_info(device)

    data = load_domain_dataset(args.csv_path, args.csv_sep, args.max_rows_per_topic)
    labels = data["topic"].to_numpy()
    languages = data["language"].to_numpy()

    print(f"Loaded {len(data)} responses from {len(TOPICS)} survey topics.")
    print("Topic counts:")
    print(data["topic"].value_counts().sort_index().to_string())
    print("\nDetected language counts:")
    print(data["language"].value_counts().sort_index().to_string())
    print("\nChance topic accuracy is about %.3f for balanced 7-way topics." % (1 / len(TOPICS)))

    results = []
    for model_name in args.models:
        print("\n" + "=" * 78)
        print(f"Benchmarking {model_name}")
        print("=" * 78)

        model, embeddings, encode_time = load_model_and_encode(
            model_name,
            data["text"],
            args.batch_size,
            device,
        )
        metrics: dict[str, float | int | str] = {
            "model": model_name,
            "device": device,
            "n_responses": int(len(data)),
            "n_topics": int(len(TOPICS)),
            "dutch_weight": round(float(args.dutch_weight), 3),
            "english_weight": round(float(1.0 - args.dutch_weight), 3),
            "nl_pct_detected": round(float(np.mean(languages == "nl") * 100), 2),
            "en_pct_detected": round(float(np.mean(languages == "en") * 100), 2),
            "unknown_lang_pct_detected": round(float(np.mean(languages == "unknown") * 100), 2),
            "embedding_dim": int(embeddings.shape[1]),
            "embed_time_s": round(float(encode_time), 2),
        }

        similarities = embeddings @ embeddings.T
        np.fill_diagonal(similarities, -np.inf)

        metrics.update(label_retrieval_metrics(similarities, labels, k=10))
        metrics.update(language_retrieval_metrics(similarities, labels, languages, args.dutch_weight, k=10))
        metrics.update(topic_query_metrics(model, embeddings, labels, languages, args.batch_size, args.dutch_weight))

        if not args.skip_linear_probe:
            metrics.update(linear_probe_metrics(embeddings, labels))
        if not args.skip_kmeans:
            metrics.update(kmeans_metrics(embeddings, labels))

        metrics["silhouette_true_topics_cosine"] = sampled_silhouette(embeddings, labels)
        metrics["domain_score"] = weighted_domain_score(metrics)
        metrics["dutch_heavy_cross_language_score"] = dutch_heavy_cross_language_score(metrics)

        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        print_model_summary(metrics, args.log_path)
        results.append(metrics)

    print_ranking(results)
    return results


def print_model_summary(metrics: dict[str, float | int | str], log_path: Path) -> None:
    important = [
        "domain_score",
        "dutch_heavy_cross_language_score",
        "cross_lang_map_at_10_70_30",
        "cross_lang_nl_to_en_map_at_10",
        "cross_lang_en_to_nl_map_at_10",
        "mixed_pool_map_at_10_70_30",
        "mixed_pool_nl_query_map_at_10",
        "mixed_pool_en_query_map_at_10",
        "query_acc_cross_language_prompt_70_30",
        "query_acc_same_language_prompt_70_30",
        "map_at_10",
        "top1_topic_acc",
        "query_acc_mixed",
        "cross_lang_map_at_10",
        "cross_lang_top1_topic_acc",
        "linear_probe_f1_macro",
        "kmeans_v_measure",
        "silhouette_true_topics_cosine",
        "embed_time_s",
    ]
    print("\nSummary:")
    for key in important:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key:32s} {value:.4f}")
            else:
                print(f"  {key:32s} {value}")
    print(f"\nAppended JSON row to {log_path}")


def print_ranking(results: list[dict[str, float | int | str]]) -> None:
    if not results:
        return
    ranked = sorted(results, key=lambda row: float(row["dutch_heavy_cross_language_score"]), reverse=True)
    print("\n" + "=" * 78)
    print("Ranking by dutch_heavy_cross_language_score")
    print("=" * 78)
    for i, row in enumerate(ranked, start=1):
        print(
            f"{i}. {row['model']}  "
            f"dutch_heavy_score={float(row['dutch_heavy_cross_language_score']):.4f}  "
            f"cross_70_30={float(row['cross_lang_map_at_10_70_30']):.4f}  "
            f"mixed_pool_70_30={float(row['mixed_pool_map_at_10_70_30']):.4f}"
        )


def run_optional_mteb(args: argparse.Namespace) -> None:
    try:
        import mteb
    except ImportError:
        print(
            "\nMTEB is not installed. Install it first, for example:\n"
            "  UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python mteb\n"
        )
        return

    task_types = ["STS", "PairClassification", "Retrieval", "Classification", "Clustering"]
    tasks = mteb.get_tasks(
        languages=args.mteb_languages,
        script=["Latn"],
        task_types=task_types,
        exclude_aggregate=True,
    )
    if args.mteb_max_tasks:
        tasks = tasks[: args.mteb_max_tasks]

    print(
        f"\nRunning MTEB for languages={args.mteb_languages} "
        f"on {len(tasks)} task(s). Output: {args.mteb_output_folder}"
    )
    for model_name in args.models:
        print(f"\nMTEB model: {model_name}")
        device = resolve_device(args.device)
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        evaluation = mteb.MTEB(tasks=tasks)
        evaluation.run(
            model,
            output_folder=str(args.mteb_output_folder / safe_name(model_name)),
            encode_kwargs={"batch_size": args.batch_size, "normalize_embeddings": True},
        )


def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def main() -> None:
    args = parse_args()
    run_domain_benchmark(args)
    if args.run_mteb:
        run_optional_mteb(args)


if __name__ == "__main__":
    main()
