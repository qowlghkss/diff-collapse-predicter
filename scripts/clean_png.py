#!/usr/bin/env python3
"""
PNG Curation for publication assets.

What this script does:
1. Recursively collects PNGs.
2. Removes low-information frames.
3. Removes near-duplicates (pixel similarity > threshold, default 0.95).
4. Clusters images into collapse / non-collapse / intervention.
5. Selects representative samples (prioritizing key timesteps t=5,10,12,15).
6. Renames and copies selected files to output dir:
   {type}_{model}_{prompt}_{seed}_{timestep}.png

Notes:
- If timestep is missing in filename, timestep defaults to 15 (final-frame proxy).
- collapse/non-collapse labels are inferred from data/multiview/master_results.csv when available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

KEY_TIMESTEPS = [5, 10, 12, 15]
DEFAULT_TIMESTEP = 15


@dataclass
class ImageRecord:
    path: Path
    model: str
    prompt: str
    method: str
    seed: int
    timestep: int
    cluster: str
    low_info: bool
    entropy: float
    edge_density: float
    feature: np.ndarray


def parse_filename(path: Path) -> Tuple[str, str, str, int, int]:
    """
    Parse patterns like:
      model_prompt_method_seed.png
      model_prompt_method_seed_t12.png
      model_prompt_method_seed_step12.png
      ...
    """
    stem = path.stem

    t_match = re.search(r"(?:_t|_step)(\d+)$", stem)
    timestep = int(t_match.group(1)) if t_match else DEFAULT_TIMESTEP
    if t_match:
        stem = stem[: t_match.start()]

    parts = stem.split("_")
    if len(parts) < 4:
        return "unknown", "unknown", "control", -1, timestep

    seed = -1
    if parts[-1].isdigit():
        seed = int(parts[-1])
        parts = parts[:-1]

    method = parts[-1] if parts else "control"
    model = parts[0] if parts else "unknown"
    prompt = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"

    return model, prompt, method, seed, timestep


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float((edges > 0).mean())


def feature_vector(img: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small / 255.0).reshape(-1)


def pixel_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    mse = float(np.mean((f1 - f2) ** 2))
    sim = 1.0 - mse
    return max(0.0, min(1.0, sim))


def infer_cluster(method: str, setting: str, seed: int, collapse_map: Dict[Tuple[str, int], int]) -> str:
    if method == "intervention":
        return "intervention"

    collapsed = collapse_map.get((setting, seed))
    if collapsed is None:
        return "collapse" if method == "shock" else "non-collapse"
    return "collapse" if int(collapsed) == 1 else "non-collapse"


def load_collapse_map(csv_path: Path) -> Dict[Tuple[str, int], int]:
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    required = {"seed", "setting", "method", "collapse"}
    if not required.issubset(df.columns):
        return {}

    ctrl = df[df["method"] == "control"].copy()
    return {(str(r.setting), int(r.seed)): int(r.collapse) for r in ctrl.itertuples(index=False)}


def prompt_to_setting(prompt: str) -> str:
    if prompt == "main":
        return "main"
    if prompt == "mvdream_explicit":
        return "explicit"
    if prompt == "stress_test":
        return "stress"
    return prompt


def curate(args: argparse.Namespace) -> Dict[str, int]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collapse_map = load_collapse_map(Path(args.master_csv))

    records: List[ImageRecord] = []
    all_pngs = sorted([p for p in input_dir.rglob("*.png") if p.is_file()])

    for p in all_pngs:
        img = cv2.imread(str(p))
        if img is None:
            continue

        model, prompt, method, seed, timestep = parse_filename(p)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ent = image_entropy(gray)
        ed = edge_density(gray)
        low_info = ent < args.min_entropy or ed < args.min_edge_density

        feat = feature_vector(img, size=args.feature_size)

        setting = prompt_to_setting(prompt)
        cluster = infer_cluster(method, setting, seed, collapse_map)

        records.append(
            ImageRecord(
                path=p,
                model=model,
                prompt=prompt,
                method=method,
                seed=seed,
                timestep=timestep,
                cluster=cluster,
                low_info=low_info,
                entropy=ent,
                edge_density=ed,
                feature=feat,
            )
        )

    # 1) low-info filtering
    filtered = [r for r in records if not r.low_info]

    # 2) dedup (greedy)
    deduped: List[ImageRecord] = []
    for r in filtered:
        is_dup = False
        for s in deduped:
            if r.cluster != s.cluster:
                continue
            sim = pixel_similarity(r.feature, s.feature)
            if sim >= args.dup_similarity:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)

    # 3) representative selection per cluster
    selected: List[ImageRecord] = []
    by_cluster: Dict[str, List[ImageRecord]] = {"collapse": [], "non-collapse": [], "intervention": []}
    for r in deduped:
        by_cluster.setdefault(r.cluster, []).append(r)

    for cluster, lst in by_cluster.items():
        if not lst:
            continue

        # prioritize key timesteps
        picked: List[ImageRecord] = []
        for t in KEY_TIMESTEPS:
            candidates = sorted(lst, key=lambda x: (abs(x.timestep - t), x.seed if x.seed >= 0 else 10**9))
            for c in candidates:
                if c not in picked:
                    picked.append(c)
                    break

        # fill remaining to max_per_cluster using medoid-ish centrality
        if len(picked) < args.max_per_cluster and len(lst) > 1:
            feats = np.stack([x.feature for x in lst], axis=0)
            dmat = ((feats[:, None, :] - feats[None, :, :]) ** 2).mean(axis=2)
            centrality = dmat.mean(axis=1)
            order = np.argsort(centrality)
            for idx in order:
                c = lst[int(idx)]
                if c not in picked:
                    picked.append(c)
                if len(picked) >= args.max_per_cluster:
                    break

        selected.extend(picked[: args.max_per_cluster])

    # 4) copy + rename + manifest
    manifest = []
    used_names = set()
    for r in selected:
        base_name = f"{r.cluster}_{r.model}_{r.prompt}_{r.seed}_{r.timestep}.png"
        name = base_name
        k = 1
        while name in used_names:
            name = base_name.replace(".png", f"_{k}.png")
            k += 1
        used_names.add(name)

        dst = output_dir / name
        shutil.copy2(r.path, dst)
        manifest.append(
            {
                "src": str(r.path),
                "dst": str(dst),
                "cluster": r.cluster,
                "model": r.model,
                "prompt": r.prompt,
                "seed": r.seed,
                "timestep": r.timestep,
                "entropy": round(r.entropy, 4),
                "edge_density": round(r.edge_density, 6),
            }
        )

    manifest_path = output_dir / "curation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "input_pngs": len(all_pngs),
        "after_low_info_filter": len(filtered),
        "after_dedup": len(deduped),
        "selected": len(selected),
        "cluster_counts": {
            k: sum(1 for x in selected if x.cluster == k)
            for k in ["collapse", "non-collapse", "intervention"]
        },
        "manifest": str(manifest_path),
    }

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curate PNGs for publication")
    p.add_argument("--input-dir", default="experiments/figures/multiview")
    p.add_argument("--output-dir", default="figures/curated")
    p.add_argument("--master-csv", default="data/multiview/master_results.csv")
    p.add_argument("--dup-similarity", type=float, default=0.95)
    p.add_argument("--min-entropy", type=float, default=2.3)
    p.add_argument("--min-edge-density", type=float, default=0.004)
    p.add_argument("--feature-size", type=int, default=64)
    p.add_argument("--max-per-cluster", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    curate(parse_args())
