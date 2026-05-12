import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import ollama
import json
import time

# ── Config ────────────────────────────────────────────────────────────────────

CSV_PATH        = 'DummyData_Final 1.csv'
CSV_SEP         = ';'
ANSWER_COL      = 'Would you like to give your institution any other feedback on the content and organisation of your course programme?'

"""

    'BAAI/bge-m3',

"""
embeding_models = [
    'zeroentropy/zembed-1-embedding',
    'BAAI/bge-m3'

]

# UMAP
UMAP_N_COMPONENTS  = 15   # dimensions to reduce to before clustering
UMAP_N_NEIGHBORS   = 15   # higher = more global structure

UMAP_MIN_DIST      = 0.0  # 0.0 works best for clustering (points can clump)

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE  = 10  # minimum rows to form a cluster
HDBSCAN_MIN_SAMPLES       = 5   # controls how conservative cluster borders are
                                # increase to get fewer, tighter clusters


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
answers = df[ANSWER_COL].dropna().tolist()
print(f"  {len(answers)} answers loaded")



for EMBEDDING_MODEL in embeding_models:

    # ── Embed ─────────────────────────────────────────────────────────────────────
    
    print(f"\nEmbedding with '{EMBEDDING_MODEL}'...")
    t0 = time.time()
    embedder = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embeddings = embedder.encode(answers, show_progress_bar=True, batch_size=64)
    embed_time = time.time() - t0
    print(f"  Done in {embed_time:.1f}s  |  shape: {embeddings.shape}")

    # ── UMAP ──────────────────────────────────────────────────────────────────────

    print(f"\nReducing dimensions with UMAP ({embeddings.shape[1]}d → {UMAP_N_COMPONENTS}d)...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42,
        verbose=False,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  Done  |  shape: {reduced.shape}")

    # ── HDBSCAN ───────────────────────────────────────────────────────────────────

    print(f"\nClustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean',          # euclidean is fine after UMAP reduction
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    noise_pct  = 100 * n_noise / len(labels)

    df_working = df[df[ANSWER_COL].notna()].copy()
    df_working['Cluster'] = labels

    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({noise_pct:.1f}% of data)")

    # ── Evaluation metrics ────────────────────────────────────────────────────────

    print("\n── Evaluation metrics ──────────────────────────────────────────────────")

    mask       = labels != -1
    clean_emb  = reduced[mask]
    clean_labs = labels[mask]

    if len(set(clean_labs)) > 1:
        sil = silhouette_score(clean_emb, clean_labs, metric='euclidean')
        dbi = davies_bouldin_score(clean_emb, clean_labs)
        chi = calinski_harabasz_score(clean_emb, clean_labs)
        print(f"  Silhouette score       : {sil:.4f}  (higher is better, range -1 to 1)")
        print(f"  Davies-Bouldin index   : {dbi:.4f}  (lower is better)")
        print(f"  Calinski-Harabasz      : {chi:.1f}  (higher is better)")
    else:
        sil, dbi, chi = None, None, None
        print("  Not enough clusters for metrics.")

    print(f"\n  Model     : {EMBEDDING_MODEL}")
    print(f"  Embed time: {embed_time:.1f}s")
    print(f"  Clusters  : {n_clusters}")
    print(f"  Noise %   : {noise_pct:.1f}%")
    print("─────────────────────────────────────────────────────────────────────────\n")

    # ── LLM labeling ─────────────────────────────────────────────────────────────

    cluster_ids = sorted([c for c in set(labels) if c != -1])
    cluster_themes = []
    established_themes = set()


    # ── Output ────────────────────────────────────────────────────────────────────


    # Append noise summary row
    if n_noise > 0:
        noise_row = pd.DataFrame([{"Theme": "(no cluster / noise)", "Sentiment": "—", "Count": n_noise}])



    # Save benchmark row (useful when comparing models)
    benchmark = {
        "model":       EMBEDDING_MODEL,
        "n_clusters":  n_clusters,
        "noise_pct":   round(noise_pct, 2),
        "silhouette":  round(sil, 4) if sil else None,
        "davies_bouldin": round(dbi, 4) if dbi else None,
        "calinski_harabasz": round(chi, 1) if chi else None,
        "embed_time_s": round(embed_time, 1),
    }
    with open('benchmark_log.jsonl', 'a') as f:
        f.write(json.dumps(benchmark) + '\n')
    print("Benchmark row appended → benchmark_log.jsonl")