import os
import glob
import re
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path if needed
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.seed_utils import set_seed

def parse_metadata(filename):
    """
    Parses model and seed from filenames like 'mvdream_baseline_control_42_ci.npy'
    """
    base = os.path.basename(filename)
    # Pattern to match <model>_<condition>_<seed>_<suffix>.npy
    # Capture model (up to baseline/main), condition (control), seed, and suffix
    match = re.search(r"^(.+)_control_(\d+)_(ci|thin)\.npy$", base)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def get_data_records(data_dir):
    """
    Discovers all (model, seed) pairs and extracts labels memory-efficiently.
    """
    ci_files = sorted(glob.glob(os.path.join(data_dir, "*_control_*_ci.npy")))
    temp_records = []
    thin_counts = []
    
    print(f"Scanning {data_dir} for control condition data pairs...")
    for cif in ci_files:
        model, seed = parse_metadata(cif)
        if model is None:
            continue
            
        thin_path = cif.replace("_ci.npy", "_thin.npy")
        if not os.path.exists(thin_path):
            continue
            
        # Memory optimization: Only load the array and take the last element
        # (Trajectories are small, but we avoid storing the whole array in the record)
        try:
            thin_traj = np.load(thin_path)
            final_thin = float(thin_traj[-1])
            
            temp_records.append({
                "model": model,
                "seed": seed,
                "final_thin": final_thin,
                "ci_path": os.path.relpath(cif, os.getcwd()),
                "thin_path": os.path.relpath(thin_path, os.getcwd())
            })
            thin_counts.append(final_thin)
        except Exception as e:
            print(f"Error loading {thin_path}: {e}")
        
    if not temp_records:
        return []
        
    # Determine collapse threshold (25th percentile of final thin-pixel counts)
    threshold = np.percentile(thin_counts, 25)
    print(f"Computed collapse threshold (25th percentile): {threshold:.2f}")
    
    records = []
    for r in temp_records:
        label = 1 if r["final_thin"] < threshold else 0
        r["label"] = label
        # Composite stratification key
        r["strat_key"] = f"{r['model']}_{label}"
        records.append(r)
        
    return records

def main():
    parser = argparse.ArgumentParser(description="Split diffusion data into Train/Val/Test sets.")
    parser.add_argument("--data-dir", type=str, default="experiments/data/multiview", help="Directory containing .npy files")
    parser.add_argument("--output", type=str, default="experiments/data/split_info.json", help="Path to save split info")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of training data")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of validation data")
    
    args = parser.parse_args()
    
    # Task 1 alignment: Set global seed
    set_seed(args.seed)
    
    records = get_data_records(args.data_dir)
    if not records:
        print("No valid data samples found. Check your data directory and filenames.")
        return
        
    df = pd.DataFrame(records)
    print(f"Total samples found: {len(df)}")
    
    # Robust Stratification: Ensure every class has at least 2 members.
    # If a class (model_label) has only 1 member, we fall back to just the label.
    counts = df["strat_key"].value_counts()
    rare_classes = counts[counts < 2].index
    
    if len(rare_classes) > 0:
        print(f"Warning: Found {len(rare_classes)} classes with only 1 member: {rare_classes.tolist()}")
        df["final_strat_key"] = df.apply(
            lambda x: f"label_{x['label']}" if x["strat_key"] in rare_classes else x["strat_key"], 
            axis=1
        )
        # Check again if fallback resulted in 1-member classes (unlikely)
        final_counts = df["final_strat_key"].value_counts()
        still_rare = final_counts[final_counts < 2].index
        if len(still_rare) > 0:
            print(f"Warning: Still have rare classes {still_rare.tolist()}. Falling back to label-only for all.")
            df["final_strat_key"] = df["label"].apply(lambda l: f"label_{l}")
    else:
        df["final_strat_key"] = df["strat_key"]
    
    print("Final stratification distribution:")
    print(df["final_strat_key"].value_counts())
    
    # First split: Train vs (Val + Test)
    val_test_ratio = 1.0 - args.train_ratio
    train_df, val_test_df = train_test_split(
        df, 
        test_size=val_test_ratio, 
        random_state=args.seed, 
        stratify=df["final_strat_key"]
    )
    
    # Second split: Check stratification again for the smaller set
    vt_counts = val_test_df["final_strat_key"].value_counts()
    if vt_counts.min() < 2:
        print("Warning: classes too small in Val+Test set. Falling back to label-only stratification.")
        second_strat = val_test_df["label"]
    else:
        second_strat = val_test_df["final_strat_key"]
    
    # Stratified Split (Val / Test)
    test_target_ratio = 1.0 - args.train_ratio - args.val_ratio
    test_relative_size = test_target_ratio / val_test_ratio
    
    val_df, test_df = train_test_split(
        val_test_df, 
        test_size=test_relative_size, 
        random_state=args.seed, 
        stratify=second_strat
    )
    
    # Finalize labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save as JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_dict = final_df.to_dict(orient="records")
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"\nSaved split info to {args.output}")
    print("\n--- Split Statistics ---")
    stats = final_df.groupby("split").agg({
        "label": ["count", "mean"]
    })
    stats.columns = ["count", "collapse_rate"]
    print(stats)
    
    print("\n--- Model Representation in Splits ---")
    pivot = pd.pivot_table(final_df, index="model", columns="split", values="label", aggfunc="count", fill_value=0)
    print(pivot)

if __name__ == "__main__":
    main()
