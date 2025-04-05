import os
import pandas as pd
import numpy as np
import argparse
import logging

import scanpy as sc
import anndata as ad
import gseapy as gp

import re
from collections import defaultdict
import pickle
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad_path", type=str, required=True)
    parser.add_argument("--genesets_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--groupby", nargs="+", default=["cell_type", "disease"], help="Groupby keys (default: cell_type + disease)")
    return parser.parse_args()


def generate_group_column(adata, groupby):
    adata.obs["group"] = adata.obs[groupby[0]].astype(str)
    for col in groupby[1:]:
        adata.obs["group"] += "_" + adata.obs[col].astype(str)
    return adata


def generate_pseudobulk(adata):
    pseudobulk_df = adata.to_df().groupby(adata.obs["group"]).sum()
    min_expr_genes = (pseudobulk_df > 0).sum(axis=0) >= 1
    return pseudobulk_df.loc[:, min_expr_genes]


def compute_log2fc(target_group, pseudobulk_df):
    expr_target = pseudobulk_df.loc[target_group]
    expr_rest = pseudobulk_df.drop(index=target_group).mean(axis=0)
    log2fc = np.log2(expr_target + 1e-5) - np.log2(expr_rest + 1e-5)
    return log2fc.sort_values(ascending=False)


def load_gmt_files(geneset_dir):
    gmt_files = glob.glob(os.path.join(geneset_dir, "*.symbols.gmt"))
    return [f for f in gmt_files if ".all." in os.path.basename(f)]


def run_gsea(ranked_genes, gene_set_path):
    try:
        enr = gp.prerank(
            rnk=ranked_genes.reset_index(),
            gene_sets=gene_set_path,
            outdir=None,
            permutation_num=100,
            seed=42,
            format=None,
            verbose=False
        )
        return enr.res2d
    except Exception as e:
        logging.info(f"Failed GSEA for group with {os.path.basename(gene_set_path)}: {e}")
        return None


def main():
    args = parse_args()

    os.makedirs(args.output_prefix, exist_ok=True)

    log_path = os.path.join(args.output_prefix, "pseduobulk_gsea_run.log")
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(f"Logging initialized for dataset: {args.dataset_name}")

    logging.info(f"Loading {args.h5ad_path}")
    adata = sc.read_h5ad(args.h5ad_path)

    for key in args.groupby:
        if key not in adata.obs:
            raise ValueError(f"'{key}' not found in adata.obs")

    logging.info(f"Generating composite group column from: {args.groupby}")
    adata = generate_group_column(adata, args.groupby)

    logging.info("Generating pseudobulk profiles...")
    pseudobulk_df = generate_pseudobulk(adata)

    logging.info("Loading gene set collections...")
    gmt_files = load_gmt_files(args.genesets_path)

    all_results = defaultdict(dict)

    for group in pseudobulk_df.index:
        logging.info(f"\nProcessing group: {group}")
        ranked = compute_log2fc(group, pseudobulk_df)

        n_duplicates = (ranked.value_counts() > 1).sum()
        logging.info(f"  Duplicate log2FC scores: {n_duplicates} out of {len(ranked)}")

        for gmt_path in gmt_files:
            base_name = os.path.basename(gmt_path).replace(".symbols.gmt", "")
            logging.info(f"  Running GSEA on: {base_name}")

            result = run_gsea(ranked, gmt_path)
            if result is not None:
                all_results[group][base_name] = result

    # Save results as CSVs, one per group
    for group_name, gsea_dict in all_results.items():
        combined_results = []
        for gmt_name, df in gsea_dict.items():
            df = df.copy()
            df["gene_set_collection"] = gmt_name
            combined_results.append(df)

        if combined_results:
            group_df = pd.concat(combined_results, axis=0)
            group_df.index.name = "gene_set"

            csv_name = re.sub(r"[^\w\-_\. ]", "_", group_name)  # clean for filename
            output_path = os.path.join(args.output_prefix, f"{csv_name}.csv")
            group_df.to_csv(output_path)

            logging.info(f"Saved GSEA CSV for {group_name} -> {output_path}")
        else:
            logging.info(f"No GSEA results to save for {group_name}")


if __name__ == "__main__":
    main()
