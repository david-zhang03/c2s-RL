import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

def sanitize_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)
    # return re.sub(r"[^\w\-_\.]", "_", name)

def compute_meta_gsea_enrichment(ranked_gene_sets, reference_set):
    N = len(ranked_gene_sets)
    hits = np.isin(ranked_gene_sets, list(reference_set)).astype(int)

    hit_score = np.sqrt((N - len(reference_set)) / len(reference_set))
    miss_score = -np.sqrt(len(reference_set) / (N - len(reference_set)))

    running_sum = np.cumsum(np.where(hits == 1, hit_score, miss_score))
    ES = running_sum.max()

    return ES, running_sum

def compute_meta_gsea_pval(ranked_gene_sets, reference_set, n_perm=1000):
    ES_obs, _ = compute_meta_gsea_enrichment(ranked_gene_sets, reference_set)

    perm_ES = []
    for _ in range(n_perm):
        ref_perm = set(np.random.choice(ranked_gene_sets, size=len(reference_set), replace=False))
        es, _ = compute_meta_gsea_enrichment(ranked_gene_sets, ref_perm)
        perm_ES.append(es)

    perm_ES = np.array(perm_ES)
    p_val = np.mean(perm_ES >= ES_obs)
    NES = ES_obs / np.mean(np.abs(perm_ES)) if np.mean(np.abs(perm_ES)) != 0 else np.nan

    return ES_obs, NES, p_val, perm_ES

def compute_recovery_depths(ranked_gene_sets, reference_set, recovery_percents=[0.5, 0.7, 0.9, 0.99]):
    ref_set = set(reference_set)
    n_ref = len(ref_set)
    found = 0
    depths = {}

    for i, term in enumerate(ranked_gene_sets):
        if term in ref_set:
            found += 1
        recovery = found / n_ref
        for target in recovery_percents:
            if target not in depths and recovery >= target:
                depths[target] = i + 1
        if len(depths) == len(recovery_percents):
            break

    for target in recovery_percents:
        if target not in depths:
            depths[target] = -1

    return depths

def run_meta_gsea_all(
    gps_df_dataset, base_dir, datasets, output_path="meta_gsea_all_datasets_summary.csv",
    sort_by="NES", n_perm=1000
):
    results = []

    for dataset_name in tqdm(datasets, desc="Running meta-GSEA on all datasets"):
        dataset_dir = os.path.join(base_dir, dataset_name)

        # Process pseudobulk results for each combination of condition and cell type
        for cond in gps_df_dataset['Disease'].unique():
            for cell_type in gps_df_dataset['Cell Type'].unique():
                # make sure to sanitize the path first
                csv_name = sanitize_filename(f"{cell_type}_{cond}") + ".csv" # remove spaces and slashes
                result_path = os.path.join(dataset_dir, csv_name)

                if not os.path.exists(result_path):
                    print(f"Missing: {result_path}")
                    continue

                # Load pseudobulk results
                df = pd.read_csv(result_path)
                if df.empty or "Term" not in df.columns:
                    print(f"Empty or malformed file: {result_path}")
                    continue

                # Sort the results as per user preference
                if sort_by == "NES":
                    df_sorted = df.sort_values(by=["FDR q-val", "NES"], ascending=[True, False])
                elif sort_by == "FDR":
                    df_sorted = df.sort_values(by="FDR q-val", ascending=True)
                elif sort_by == "neglog10FDR":
                    df["-log10FDR"] = -np.log10(df["FDR q-val"] + 1e-8)
                    df_sorted = df.sort_values(by="-log10FDR", ascending=False)
                else:
                    raise ValueError(f"Invalid sort_by: {sort_by}")

                ranked_gene_sets = df_sorted["Term"].tolist()

                # Reference sets: Find the top gene programs for the current condition and cell type
                reference_set = gps_df_dataset[
                    (gps_df_dataset['Cell Type'] == cell_type) &
                    (gps_df_dataset['Disease'] == cond) &
                    (gps_df_dataset['Rank Type'] == "Top")
                ]['Gene Program'].values.tolist()

                if len(reference_set) == 0:
                    continue

                # Meta-GSEA
                es, nes, pval, perm_dist = compute_meta_gsea_pval(ranked_gene_sets, reference_set, n_perm=n_perm)

                # Recovery depth metrics
                recovery_depths = compute_recovery_depths(ranked_gene_sets, reference_set)

                # Append results
                results.append({
                    "dataset": dataset_name,
                    "cell_type": cell_type,
                    "condition": cond,
                    "ES": es,
                    "NES": nes,
                    "p_value": pval,
                    "n_ref": len(reference_set),
                    "depth_50pct": recovery_depths[0.5],
                    "depth_70pct": recovery_depths[0.7],
                    "depth_90pct": recovery_depths[0.9],
                    "depth_99pct": recovery_depths[0.99]
                })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved final summary to: {output_path}")
    return results_df