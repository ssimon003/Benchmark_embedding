"""
Focused MTEB benchmark for Dutch-heavy Dutch/English embedding usage.

This is an external sanity check for survey_embedding_benchmark.py. The survey
CSV is domain-specific but dummy/labeled; MTEB is public and labeled, so agreement
between both rankings is stronger evidence than either benchmark alone.

Run:

    .venv/bin/python mteb_dutch_english_benchmark.py --device mps

The default task suite is text-only and Dutch-first, with cross-language
Dutch-English bitext/STS tasks where MTEB provides them.
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
from typing import Any

os.environ.setdefault("MTEB_CACHE", ".mteb-cache")
os.environ.setdefault("HF_DATASETS_CACHE", ".hf-datasets-cache")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mteb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODELS = [
    "zeroentropy/zembed-1-embedding",
    "BAAI/bge-m3",
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_subsets: tuple[str, ...] | None = None
    languages: tuple[str, ...] | None = None
    weight: float = 1.0
    note: str = ""


FOCUSED_TASKS = [
    # Dutch semantic similarity / pair matching.
    TaskSpec("SICK-NL-STS", weight=1.2, note="Dutch semantic similarity"),
    TaskSpec("STSBenchmarkMultilingualSTS", hf_subsets=("nl",), weight=1.2, note="Dutch STS"),
    TaskSpec("SICKNLPairClassification", weight=1.0, note="Dutch sentence-pair classification"),
    TaskSpec("XLWICNLPairClassification", weight=0.8, note="Dutch word-in-context pair task"),
    # Cross-language Dutch-English checks.
    TaskSpec("STS17", hf_subsets=("nl-en",), languages=("nld", "eng"), weight=1.5, note="Dutch-English STS"),
    TaskSpec("Tatoeba", hf_subsets=("nld-eng",), languages=("nld", "eng"), weight=1.5, note="Dutch-English bitext"),
    TaskSpec("IWSLT2017BitextMining", hf_subsets=("nl-en", "en-nl"), languages=("nld", "eng"), weight=1.5, note="Dutch-English translation bitext"),
    TaskSpec("WebFAQBitextMiningQuestions", hf_subsets=("eng-nld",), languages=("nld", "eng"), weight=1.2, note="Dutch-English FAQ questions"),
    # Dutch classification / clustering / retrieval.
    TaskSpec("DutchBookReviewSentimentClassification.v2", weight=0.8, note="Dutch sentiment classification"),
    TaskSpec("DutchNewsArticlesClassification", weight=0.8, note="Dutch topic classification"),
    TaskSpec("MultiHateClassification", hf_subsets=("nld", "eng"), languages=("nld", "eng"), weight=0.8, note="Dutch and English classification"),
    TaskSpec("DutchNewsArticlesClusteringS2S", weight=0.8, note="Dutch clustering"),
    TaskSpec("Quora-NL", weight=1.2, note="Dutch duplicate-question retrieval"),
    TaskSpec("DutchNewsArticlesRetrieval", weight=1.0, note="Dutch news retrieval"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Dutch/English focused MTEB benchmark.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--output-folder", type=Path, default=Path("mteb_dutch_english_results"))
    parser.add_argument("--summary-path", type=Path, default=Path("mteb_dutch_english_summary.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller suite first: STS, pair classification, bitext, Quora-NL.",
    )
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


def selected_specs(quick: bool) -> list[TaskSpec]:
    if not quick:
        return FOCUSED_TASKS
    quick_names = {
        "SICK-NL-STS",
        "STSBenchmarkMultilingualSTS",
        "STS17",
        "Tatoeba",
        "IWSLT2017BitextMining",
        "SICKNLPairClassification",
        "Quora-NL",
    }
    return [spec for spec in FOCUSED_TASKS if spec.name in quick_names]


def build_tasks(specs: list[TaskSpec]) -> list[Any]:
    tasks = []
    for spec in specs:
        task = mteb.get_task(
            spec.name,
            languages=list(spec.languages) if spec.languages else ["nld"],
            script=["Latn"],
            hf_subsets=list(spec.hf_subsets) if spec.hf_subsets else None,
        )
        tasks.append(task)
    return tasks


def safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def result_to_score(task_result: Any) -> float:
    if hasattr(task_result, "get_score"):
        score = task_result.get_score()
        if isinstance(score, dict):
            values = [float(v) for v in score.values() if isinstance(v, (int, float))]
            return float(np.mean(values)) if values else math.nan
        return float(score)

    scores = getattr(task_result, "scores", None)
    if isinstance(scores, dict):
        values = []
        for split_scores in scores.values():
            if isinstance(split_scores, list):
                for item in split_scores:
                    if isinstance(item, dict):
                        values.extend(float(v) for v in item.values() if isinstance(v, (int, float)))
            elif isinstance(split_scores, dict):
                values.extend(float(v) for v in split_scores.values() if isinstance(v, (int, float)))
        return float(np.mean(values)) if values else math.nan
    return math.nan


def result_to_name(task_result: Any) -> str:
    return (
        getattr(task_result, "task_name", None)
        or getattr(getattr(task_result, "task", None), "metadata", None).name
        or "unknown"
    )


def result_to_metric(task_result: Any) -> str:
    task = getattr(task_result, "task", None)
    metadata = getattr(task, "metadata", None)
    if metadata is not None and hasattr(metadata, "main_score"):
        return str(metadata.main_score)
    if hasattr(task_result, "main_score"):
        return str(task_result.main_score)
    return "main_score"


def summarize_model(
    model_name: str,
    task_results: list[Any],
    specs_by_name: dict[str, TaskSpec],
    elapsed_s: float,
    device: str,
) -> dict[str, Any]:
    task_rows = []
    weighted_total = 0.0
    weight_total = 0.0
    cross_total = 0.0
    cross_weight = 0.0
    dutch_total = 0.0
    dutch_weight = 0.0

    for result in task_results:
        task_name = result_to_name(result)
        spec = specs_by_name.get(task_name)
        score = result_to_score(result)
        weight = spec.weight if spec else 1.0
        note = spec.note if spec else ""

        row = {
            "task": task_name,
            "metric": result_to_metric(result),
            "score": score,
            "weight": weight,
            "note": note,
        }
        task_rows.append(row)

        if not math.isnan(score):
            weighted_total += score * weight
            weight_total += weight
            if "Dutch-English" in note:
                cross_total += score * weight
                cross_weight += weight
            else:
                dutch_total += score * weight
                dutch_weight += weight

    return {
        "model": model_name,
        "device": device,
        "mteb_version": getattr(mteb, "__version__", "unknown"),
        "n_tasks": len(task_results),
        "weighted_score": weighted_total / weight_total if weight_total else math.nan,
        "cross_language_score": cross_total / cross_weight if cross_weight else math.nan,
        "dutch_text_score": dutch_total / dutch_weight if dutch_weight else math.nan,
        "elapsed_s": round(elapsed_s, 2),
        "tasks": task_rows,
    }


def print_device_info(device: str) -> None:
    print(
        "Device: "
        f"{device} | torch={torch.__version__} | "
        f"mps_built={torch.backends.mps.is_built()} | "
        f"mps_available={torch.backends.mps.is_available()} | "
        f"cuda_available={torch.cuda.is_available()}"
    )
    print(f"MTEB_CACHE={os.environ['MTEB_CACHE']}")
    print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print_device_info(device)

    specs = selected_specs(args.quick)
    tasks = build_tasks(specs)
    specs_by_name = {spec.name: spec for spec in specs}
    print("Tasks:")
    for spec in specs:
        subsets = f" subsets={list(spec.hf_subsets)}" if spec.hf_subsets else ""
        print(f"  - {spec.name}{subsets} | weight={spec.weight} | {spec.note}")

    args.output_folder.mkdir(parents=True, exist_ok=True)
    summaries = []

    for model_name in args.models:
        print("\n" + "=" * 78)
        print(f"MTEB model: {model_name}")
        print("=" * 78)
        start = time.time()
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            model,
            output_folder=str(args.output_folder / safe_name(model_name)),
            overwrite_results=args.overwrite,
            encode_kwargs={
                "batch_size": args.batch_size,
                "normalize_embeddings": True,
            },
        )
        summary = summarize_model(model_name, results, specs_by_name, time.time() - start, device)
        summaries.append(summary)
        with args.summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(
            f"weighted_score={summary['weighted_score']:.4f} | "
            f"cross_language_score={summary['cross_language_score']:.4f} | "
            f"dutch_text_score={summary['dutch_text_score']:.4f} | "
            f"elapsed_s={summary['elapsed_s']:.1f}"
        )

    ranked = sorted(summaries, key=lambda row: float(row["weighted_score"]), reverse=True)
    print("\nRanking by focused Dutch/English MTEB weighted_score")
    for i, row in enumerate(ranked, start=1):
        print(
            f"{i}. {row['model']} "
            f"weighted={row['weighted_score']:.4f} "
            f"cross={row['cross_language_score']:.4f} "
            f"dutch={row['dutch_text_score']:.4f}"
        )


if __name__ == "__main__":
    main()
