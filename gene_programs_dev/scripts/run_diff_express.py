# Import necessary libraries
import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--h5ad_path', type=str,
                    help='Path to h5ad file')

parser.add_argument('--dataset_name', type=str,
                    help='Name of .h5ad dataset')

parser.add_argument('--output_prefix', type=str, default=None,
                    help='Path to the output file')
    
parser.add_argument('--cell_type_column', type=str, required=False, default='cell_type',
                    help='column name in adata.obs that contains cell type information')

parser.add_argument('--disease_column', type=str, required=False, default='disease',
                    help='column name in adata.obs that contains disease information')

parser.add_argument('--norm_method', type=str, required=False, default='auto',
                    help='normalization method to use')

parser.add_argument('--target_sum', type=float, required=False, default=1e4,
                    help='target sum for library size normalization before log1p (default: 10000)')

parser.add_argument('--top_n_genes', type=int, required=False, default=25,
                    help='number of top differentially expressed genes to include in output (default: 25)')

parser.add_argument('--pval_cutoff', type=float, required=False, default=0.05,
                    help='adjusted p-value cutoff for significance (default: 0.05)')

parser.add_argument('--logfc_cutoff', type=float, required=False, default=1.0,
                    help='log fold change cutoff for significance (default: 1.0)')

parser.add_argument('--reference_disease', type=str, required=False, default="normal",
                    help='label for reference disease to use to compare withiin cell types (default: normal)')


args = parser.parse_args()

# Define output directory
output_prefix = os.path.abspath(args.output_prefix)
output_path = os.path.join(output_prefix, args.dataset_name)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# Create directory for summary plots only
plots_dir = os.path.join(output_path, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

# Load the h5ad dataset
print(f"Loading AnnData object {os.path.abspath((args.h5ad_path))}")
adata = sc.read_h5ad(args.h5ad_path)
print(f"Loaded h5ad file: n_cells = {adata.shape[0]:,} , n_genes = {adata.shape[1]:,}")

# Basic exploration of the dataset
print("\nData overview:")
print(adata)

# Check if cell type and disease annotations exist
cell_type_col = args.cell_type_column
disease_col = args.disease_column

if cell_type_col not in adata.obs.columns:
    print(f"Cell type annotations not found in column '{cell_type_col}'. Please make sure your dataset has cell type labels.")
    sys.exit(1)

if disease_col not in adata.obs.columns:
    print(f"Disease annotations not found in column '{disease_col}'. Please make sure your dataset has disease information.")
    print(f"Available columns: {list(adata.obs.columns)}")
    sys.exit(1)

cell_types = sorted(adata.obs[cell_type_col].unique())
disease_states = sorted(adata.obs[disease_col].unique())

print(f"\nCell types in the dataset: {cell_types}")
print(f"\nDisease states in the dataset: {disease_states}")

if len(cell_types) <= 1:
    print(f"Require at least two cell types to perform differential expression analysis")
    sys.exit(1)

# Normalize data if not already normalized
if 'normalization' not in adata.uns or 'log1p' not in adata.uns:
    print("Data is unnormalized. Normalizing...")
    if args.norm_method != 'auto':
        print("Unsupported normalization method")
        sys.exit(1)
    adata.layers['raw'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    print("Normalization complete")

# Function to perform DE analysis and extract dotplot values
def run_de_analysis(adata, group_by, group_value, condition_name=None, filter_condition=None):
    """
    Perform differential expression analysis and extract values for plotting
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    group_by : str
        Column name to group by
    group_value : str
        Value in the group_by column to analyze
    condition_name : str, optional
        Name of additional condition for combined analysis
    filter_condition : tuple, optional
        (column, value) tuple for filtering data before analysis
        
    Returns:
    --------
    Dictionary with analysis results, including:
    - General statistics (num genes, logFC, etc.)
    - Top genes with their statistics
    - Dotplot data for the top genes
    """
    print(f"\nAnalyzing {group_by}={group_value}" + 
          (f" with {filter_condition[0]}={filter_condition[1]}" if filter_condition else ""))
    
    # Create a working copy of adata to avoid modifying the original
    if filter_condition:
        column, value = filter_condition
        mask = adata.obs[column] == value
        # Skip if no cells match the filter
        if sum(mask) == 0:
            print(f"No cells found for {group_by}={group_value} with {column}={value}. Skipping.")
            return None
        working_adata = adata[mask].copy()
    else:
        working_adata = adata.copy()
    
    # Create a mask for the current group
    working_adata.obs['is_group'] = (working_adata.obs[group_by] == group_value).astype('category')
    
    # Perform rank_genes_groups
    sc.tl.rank_genes_groups(working_adata, 'is_group', method='wilcoxon')
    
    # Get the results
    de_genes = sc.get.rank_genes_groups_df(working_adata, group='True')
    
    # Filter for significant genes
    significant_de_genes = de_genes[(de_genes['pvals_adj'] < args.pval_cutoff) & 
                                    (abs(de_genes['logfoldchanges']) > args.logfc_cutoff)]
    
    # Get top N marker genes
    n_genes = min(args.top_n_genes, len(significant_de_genes))
    top_markers = significant_de_genes.head(n_genes)['names'].tolist() if n_genes > 0 else []
    
    # Calculate additional statistics
    mean_logfc = significant_de_genes['logfoldchanges'].mean() if len(significant_de_genes) > 0 else 0
    max_logfc = significant_de_genes['logfoldchanges'].max() if len(significant_de_genes) > 0 else 0
    min_pval = significant_de_genes['pvals'].min() if len(significant_de_genes) > 0 else 1
    
    # Create a basic row for this group with all relevant information
    group_row = {
        f'{group_by}': group_value,
        'Num_DE_Genes': len(significant_de_genes),
        'Num_Cells': sum(working_adata.obs['is_group'] == 'True'),
        'Percentage_Cells': round((working_adata.obs['is_group'] == 'True').mean() * 100, 2),
        'Mean_LogFC': mean_logfc,
        'Max_LogFC': max_logfc,
        'Min_Pval': min_pval
    }
    
    # Add condition info if provided
    if condition_name:
        group_row['Condition'] = condition_name
    
    # Instead of plotting, extract the data that would be plotted for the top genes
    dotplot_data = {}
    if top_markers:
        # Get expression values for top genes in group vs non-group
        for gene in top_markers:
            # Skip genes that aren't in the adata
            if gene not in working_adata.var_names:
                continue
                
            # Get expression values for this gene
            is_group_mask = working_adata.obs['is_group'] == 'True'
            
            # Calculate mean expression in group and non-group
            mean_expr_in_group = working_adata[is_group_mask, gene].X.mean()
            mean_expr_out_group = working_adata[~is_group_mask, gene].X.mean()
            
            # Calculate fraction of cells expressing the gene
            # Assuming gene is expressed if value > 0
            frac_expr_in_group = np.mean(working_adata[is_group_mask, gene].X > 0)
            frac_expr_out_group = np.mean(working_adata[~is_group_mask, gene].X > 0)
            
            # Get gene statistics from DE results
            gene_stats = significant_de_genes[significant_de_genes['names'] == gene].iloc[0]
            
            # Save all relevant data for this gene
            gene_key = f"gene_{gene}"
            group_row[f"{gene_key}_logFC"] = gene_stats['logfoldchanges']
            group_row[f"{gene_key}_pval"] = gene_stats['pvals']
            group_row[f"{gene_key}_pval_adj"] = gene_stats['pvals_adj']
            group_row[f"{gene_key}_mean_expr_in_group"] = mean_expr_in_group
            group_row[f"{gene_key}_mean_expr_out_group"] = mean_expr_out_group
            group_row[f"{gene_key}_frac_expr_in_group"] = frac_expr_in_group
            group_row[f"{gene_key}_frac_expr_out_group"] = frac_expr_out_group
            
            # Add the gene to the list of top genes
            dotplot_data[gene] = {
                'logFC': gene_stats['logfoldchanges'],
                'pval': gene_stats['pvals'],
                'pval_adj': gene_stats['pvals_adj'],
                'mean_expr_in_group': mean_expr_in_group,
                'mean_expr_out_group': mean_expr_out_group,
                'frac_expr_in_group': frac_expr_in_group,
                'frac_expr_out_group': frac_expr_out_group
            }
    
    # Save the top markers list
    for i, gene in enumerate(top_markers):
        group_row[f'Top_DEG_{i+1}'] = gene
    
    # Fill in empty gene slots if less than args.top_n_genes genes
    for i in range(len(top_markers), args.top_n_genes):
        group_row[f'Top_DEG_{i+1}'] = ""
    
    # Save significant DE genes to CSV 
    if len(significant_de_genes) > 0:
        # Create directory structure for CSVs
        degs_dir = os.path.join(output_path, 'DEGs')
        if not os.path.exists(degs_dir):
            os.makedirs(degs_dir, exist_ok=True)
            
        # Create sub-directories if needed
        if condition_name:
            degs_subdir = os.path.join(degs_dir, f"{group_by}_{condition_name}")
            if not os.path.exists(degs_subdir):
                os.makedirs(degs_subdir, exist_ok=True)
            csv_dir = degs_subdir
        else:
            csv_dir = degs_dir
        
        # Create a safe filename
        safe_group = ''.join(c if c.isalnum() else '_' for c in str(group_value))
        safe_condition = ''.join(c if c.isalnum() else '_' for c in str(condition_name)) if condition_name else ""
        filename_prefix = f"{safe_group}"
        if safe_condition:
            filename_prefix += f"_{safe_condition}"
            
        # Save to CSV
        csv_path = os.path.join(csv_dir, f"{filename_prefix}_significant_DEGs.csv")
        significant_de_genes.to_csv(csv_path, index=False)
    
    # Return the results row and the dotplot data
    return group_row, dotplot_data

# 1. Perform DE analysis grouped by cell type (across all disease states)
print("\n=== Performing DE analysis by cell type (across all disease states) ===")
cell_type_results = []
cell_type_dotplot_data = {}

for cell_type in cell_types:
    result = run_de_analysis(adata, cell_type_col, cell_type)
    if result:
        group_row, dotplot_data = result
        cell_type_results.append(group_row)
        cell_type_dotplot_data[cell_type] = dotplot_data

# Save cell type results to CSV
if cell_type_results:
    cell_type_df = pd.DataFrame(cell_type_results)
    cell_type_csv = os.path.join(output_path, f"{args.dataset_name}_cell_type_DEGs_summary.csv")
    cell_type_df.to_csv(cell_type_csv, index=False)
    print(f"Saved cell type DE summary to: {cell_type_csv}")
    
    # Create a more detailed dataframe with dotplot data for all genes
    dotplot_rows = []
    for cell_type, genes_data in cell_type_dotplot_data.items():
        for gene, data in genes_data.items():
            row = {
                'Cell_Type': cell_type,
                'Gene': gene,
                'logFC': data['logFC'],
                'pval': data['pval'],
                'pval_adj': data['pval_adj'],
                'mean_expr_in_group': data['mean_expr_in_group'],
                'mean_expr_out_group': data['mean_expr_out_group'],
                'frac_expr_in_group': data['frac_expr_in_group'],
                'frac_expr_out_group': data['frac_expr_out_group']
            }
            dotplot_rows.append(row)
    
    if dotplot_rows:
        dotplot_df = pd.DataFrame(dotplot_rows)
        dotplot_csv = os.path.join(output_path, f"{args.dataset_name}_cell_type_DEGs_dotplot_data.csv")
        dotplot_df.to_csv(dotplot_csv, index=False)
        print(f"Saved detailed dotplot data to: {dotplot_csv}")
    
    # Summary barplot for cell types
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cell_type_df, x=cell_type_col, y='Num_DE_Genes')
    plt.xticks(rotation=90)
    plt.title('Number of DE Genes by Cell Type')
    plt.tight_layout()
    barplot_path = os.path.join(plots_dir, "cell_type_DE_gene_counts.png")
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()

# 2. Perform DE analysis grouped by disease state (across all cell types)
disease_results = []
disease_dotplot_data = {}

if len(disease_states) < 2:
    print("\nWarning: Found only one disease state. Skipping disease-based DE analysis.")
    print(f"Disease column '{disease_col}' has only one value: {disease_states[0]}")
else:
    print("\n=== Performing DE analysis by disease state (across all cell types) ===")
    for disease in disease_states:
        result = run_de_analysis(adata, disease_col, disease)
        if result:
            group_row, dotplot_data = result
            disease_results.append(group_row)
            disease_dotplot_data[disease] = dotplot_data

# Save disease state results to CSV
if disease_results:
    disease_df = pd.DataFrame(disease_results)
    disease_csv = os.path.join(output_path, f"{args.dataset_name}_disease_DEGs_summary.csv")
    disease_df.to_csv(disease_csv, index=False)
    print(f"Saved disease DE summary to: {disease_csv}")
    
    # Create a more detailed dataframe with dotplot data for all genes
    dotplot_rows = []
    for disease, genes_data in disease_dotplot_data.items():
        for gene, data in genes_data.items():
            row = {
                'Disease': disease,
                'Gene': gene,
                'logFC': data['logFC'],
                'pval': data['pval'],
                'pval_adj': data['pval_adj'],
                'mean_expr_in_group': data['mean_expr_in_group'],
                'mean_expr_out_group': data['mean_expr_out_group'],
                'frac_expr_in_group': data['frac_expr_in_group'],
                'frac_expr_out_group': data['frac_expr_out_group']
            }
            dotplot_rows.append(row)
    
    if dotplot_rows:
        dotplot_df = pd.DataFrame(dotplot_rows)
        dotplot_csv = os.path.join(output_path, f"{args.dataset_name}_disease_DEGs_dotplot_data.csv")
        dotplot_df.to_csv(dotplot_csv, index=False)
        print(f"Saved detailed disease dotplot data to: {dotplot_csv}")

    # Summary barplot for disease states
    plt.figure(figsize=(12, 6))
    sns.barplot(data=disease_df, x=disease_col, y='Num_DE_Genes')
    plt.xticks(rotation=90)
    plt.title('Number of DE Genes by Disease State')
    plt.tight_layout()
    barplot_path = os.path.join(plots_dir, "disease_DE_gene_counts.png")
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    plt.close()

# 3A. Perform DE analysis for each cell type ACROSS disease states (this is more meaningful for disease mechanisms)
print("\n=== Performing DE analysis for each cell type ACROSS disease states ===")
cell_type_by_disease_results = []
cell_type_by_disease_dotplot_data = {}

# Assuming one of the disease states represents a control/healthy condition
# You might need to specify which disease state is the reference/baseline
# For example, often 'normal', 'healthy', or 'control' is used as reference
# For this example, let's assume the first disease state is the reference
if len(disease_states) < 2:
    print("\nWarning: Need at least two disease states to compare across diseases")
elif args.reference_disease not in disease_states:
    print(f"\nWarning: {args.reference_disease} not found in {disease_states}.")
else:
    reference_disease = args.reference_disease  # Assuming first one is reference/control
    print(f"\nUsing '{reference_disease}' as the reference disease state")
    
    # For each cell type, compare each disease vs. reference disease
    for cell_type in cell_types:
        print(f"\nAnalyzing cell type: {cell_type} across disease states")
        
        # Only analyze cell types that exist in the reference disease
        reference_mask = (adata.obs[cell_type_col] == cell_type) & (adata.obs[disease_col] == reference_disease)
        if sum(reference_mask) == 0:
            print(f"\nNo {cell_type} cells found in reference disease {reference_disease}. Skipping.")
            continue
            
        # Compare this cell type in each disease state vs. reference
        for disease in set(disease_states) - {reference_disease}:  # Skip the reference disease
            print(f"\nComparing {disease} vs {reference_disease} in {cell_type} cells")
            
            # Filter for only this cell type
            cell_type_mask = adata.obs[cell_type_col] == cell_type
            cell_type_adata = adata[cell_type_mask].copy()
            
            # Skip if not enough cells in either group
            disease_mask = cell_type_adata.obs[disease_col] == disease
            reference_mask = cell_type_adata.obs[disease_col] == reference_disease
            
            if sum(disease_mask) < 3 or sum(reference_mask) < 3:
                print(f"\nNot enough cells for comparison ({sum(disease_mask)} vs {sum(reference_mask)}). Skipping.")
                continue
                
            # Create disease state indicator for this comparison
            cell_type_adata.obs['is_disease'] = (cell_type_adata.obs[disease_col] == disease).astype('category')
            
            # Run DE analysis
            sc.tl.rank_genes_groups(cell_type_adata, 'is_disease', method='wilcoxon')
            
            # Get results
            de_genes = sc.get.rank_genes_groups_df(cell_type_adata, group='True')
            
            # Filter for significant genes
            significant_de_genes = de_genes[(de_genes['pvals_adj'] < args.pval_cutoff) & 
                                           (abs(de_genes['logfoldchanges']) > args.logfc_cutoff)]
            
            # Get top N marker genes
            n_genes = min(args.top_n_genes, len(significant_de_genes))
            top_markers = significant_de_genes.head(n_genes)['names'].tolist() if n_genes > 0 else []
            
            # Calculate stats
            mean_logfc = significant_de_genes['logfoldchanges'].mean() if len(significant_de_genes) > 0 else 0
            max_logfc = significant_de_genes['logfoldchanges'].max() if len(significant_de_genes) > 0 else 0
            min_pval = significant_de_genes['pvals'].min() if len(significant_de_genes) > 0 else 1
            
            # Create result row
            result_row = {
                'Cell_Type': cell_type,
                'Disease': disease,
                'Reference_Disease': reference_disease,
                'Num_DE_Genes': len(significant_de_genes),
                'Num_Cells_Disease': sum(disease_mask),
                'Num_Cells_Reference': sum(reference_mask),
                'Mean_LogFC': mean_logfc,
                'Max_LogFC': max_logfc,
                'Min_Pval': min_pval,
                'Analysis_Type': 'Cell_Type_Across_Disease'
            }
            
            # Add top markers
            for i, gene in enumerate(top_markers):
                result_row[f'Top_DEG_{i+1}'] = gene
            
            # Fill in empty gene slots
            for i in range(len(top_markers), args.top_n_genes):
                result_row[f'Top_DEG_{i+1}'] = ""
                
            # Extract dotplot data
            dotplot_data = {}
            for gene in top_markers:
                if gene not in cell_type_adata.var_names:
                    continue
                    
                # Get expression values for this gene
                disease_expr = cell_type_adata[disease_mask, gene].X
                reference_expr = cell_type_adata[reference_mask, gene].X
                
                # Calculate statistics
                mean_expr_disease = disease_expr.mean()
                mean_expr_reference = reference_expr.mean()
                frac_expr_disease = np.mean(disease_expr > 0)
                frac_expr_reference = np.mean(reference_expr > 0)
                
                # Get gene stats
                gene_stats = significant_de_genes[significant_de_genes['names'] == gene].iloc[0]
                
                # Save gene data
                dotplot_data[gene] = {
                    'logFC': gene_stats['logfoldchanges'],
                    'pval': gene_stats['pvals'],
                    'pval_adj': gene_stats['pvals_adj'],
                    'mean_expr_disease': mean_expr_disease,
                    'mean_expr_reference': mean_expr_reference,
                    'frac_expr_disease': frac_expr_disease,
                    'frac_expr_reference': frac_expr_reference
                }
            
            # Save the results
            cell_type_by_disease_results.append(result_row)
            cell_type_by_disease_dotplot_data[(cell_type, disease, reference_disease)] = dotplot_data
            
            # Save significant DE genes to CSV
            if len(significant_de_genes) > 0:
                # Create directory structure for CSVs
                degs_dir = os.path.join(output_path, 'DEGs', 'cell_type_across_disease')
                if not os.path.exists(degs_dir):
                    os.makedirs(degs_dir, exist_ok=True)
                
                # Create a safe filename
                safe_cell_type = ''.join(c if c.isalnum() else '_' for c in str(cell_type))
                safe_disease = ''.join(c if c.isalnum() else '_' for c in str(disease))
                safe_reference = ''.join(c if c.isalnum() else '_' for c in str(reference_disease))
                
                filename = f"{safe_cell_type}_{safe_disease}_vs_{safe_reference}_DEGs.csv"
                csv_path = os.path.join(degs_dir, filename)
                significant_de_genes.to_csv(csv_path, index=False)
                print(f"    Saved {len(significant_de_genes)} significant DEGs to: {csv_path}")

# Save cell type by disease results
if cell_type_by_disease_results:
    # Save summary DataFrame
    summary_df = pd.DataFrame(cell_type_by_disease_results)
    summary_csv = os.path.join(output_path, f"{args.dataset_name}_cell_type_across_disease_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved cell type across disease summary to: {summary_csv}")
    
    # Create detailed dotplot data DataFrame
    dotplot_rows = []
    for (cell_type, disease, reference), genes_data in cell_type_by_disease_dotplot_data.items():
        for gene, data in genes_data.items():
            row = {
                'Cell_Type': cell_type,
                'Disease': disease,
                'Reference': reference,
                'Gene': gene,
                'logFC': data['logFC'],
                'pval': data['pval'],
                'pval_adj': data['pval_adj'],
                'mean_expr_disease': data['mean_expr_disease'],
                'mean_expr_reference': data['mean_expr_reference'],
                'frac_expr_disease': data['frac_expr_disease'],
                'frac_expr_reference': data['frac_expr_reference']
            }
            dotplot_rows.append(row)
    
    if dotplot_rows:
        dotplot_df = pd.DataFrame(dotplot_rows)
        dotplot_csv = os.path.join(output_path, f"{args.dataset_name}_cell_type_across_disease_dotplot_data.csv")
        dotplot_df.to_csv(dotplot_csv, index=False)
        print(f"Saved detailed dotplot data to: {dotplot_csv}")
    
    # Create summary visualization
    plt.figure(figsize=(14, 8))
    sns.barplot(data=summary_df, x='Cell_Type', y='Num_DE_Genes', hue='Disease')
    plt.xticks(rotation=90)
    plt.title('Number of DE Genes by Cell Type Across Disease States')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cell_type_across_disease_DE_gene_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Create a heatmap showing the number of DE genes for each cell type and disease
    pivot_df = summary_df.pivot(index='Cell_Type', columns='Disease', values='Num_DE_Genes')
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='g')
    plt.title('Number of DE Genes by Cell Type and Disease State\n(compared to reference)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cell_type_across_disease_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 3B. Also perform the original analysis (comparing cell types within each disease)
print("\n=== Performing DE analysis for each cell type WITHIN each disease state ===")
cell_types_within_disease_results = []
cell_types_within_disease_dotplot_data = {}

for disease in disease_states:
    for cell_type in cell_types:
        # For each cell type within a disease, compare against all other cell types in the same disease
        disease_filter = (disease_col, disease)
        
        # Filter for this disease
        disease_mask = adata.obs[disease_col] == disease
        if sum(disease_mask) == 0:
            print(f"No cells found for disease {disease}. Skipping.")
            continue
        disease_adata = adata[disease_mask].copy()
        
        # Create cell type indicator
        disease_adata.obs['is_cell_type'] = (disease_adata.obs[cell_type_col] == cell_type).astype('category')
        
        # Skip if no cells of this type in this disease
        if sum(disease_adata.obs['is_cell_type'] == 'True') == 0:
            print(f"No {cell_type} cells found in {disease}. Skipping.")
            continue
            
        # Run DE analysis
        sc.tl.rank_genes_groups(disease_adata, 'is_cell_type', method='wilcoxon')
        
        # Get results
        de_genes = sc.get.rank_genes_groups_df(disease_adata, group='True')
        
        # Filter for significant genes
        significant_de_genes = de_genes[(de_genes['pvals_adj'] < args.pval_cutoff) & 
                                       (abs(de_genes['logfoldchanges']) > args.logfc_cutoff)]
        
        # Get top N marker genes
        n_genes = min(args.top_n_genes, len(significant_de_genes))
        top_markers = significant_de_genes.head(n_genes)['names'].tolist() if n_genes > 0 else []
        
        # Calculate stats
        mean_logfc = significant_de_genes['logfoldchanges'].mean() if len(significant_de_genes) > 0 else 0
        max_logfc = significant_de_genes['logfoldchanges'].max() if len(significant_de_genes) > 0 else 0
        min_pval = significant_de_genes['pvals'].min() if len(significant_de_genes) > 0 else 1
        
        # Create result row
        result_row = {
            'Cell_Type': cell_type,
            'Disease': disease,
            'Num_DE_Genes': len(significant_de_genes),
            'Num_Cells_This_Type': sum(disease_adata.obs['is_cell_type'] == 'True'),
            'Num_Cells_Other_Types': sum(disease_adata.obs['is_cell_type'] == 'False'),
            'Mean_LogFC': mean_logfc,
            'Max_LogFC': max_logfc,
            'Min_Pval': min_pval,
            'Analysis_Type': 'Cell_Types_Within_Disease'
        }
        
        # Add top markers
        for i, gene in enumerate(top_markers):
            result_row[f'Top_DEG_{i+1}'] = gene
        
        # Fill in empty gene slots
        for i in range(len(top_markers), args.top_n_genes):
            result_row[f'Top_DEG_{i+1}'] = ""
            
        # Extract dotplot data
        dotplot_data = {}
        for gene in top_markers:
            if gene not in disease_adata.var_names:
                continue
                
            # Get expression values for this gene
            is_cell_type_mask = disease_adata.obs['is_cell_type'] == 'True'
            
            # Calculate statistics
            mean_expr_this_type = disease_adata[is_cell_type_mask, gene].X.mean()
            mean_expr_other_types = disease_adata[~is_cell_type_mask, gene].X.mean()
            frac_expr_this_type = np.mean(disease_adata[is_cell_type_mask, gene].X > 0)
            frac_expr_other_types = np.mean(disease_adata[~is_cell_type_mask, gene].X > 0)
            
            # Get gene stats
            gene_stats = significant_de_genes[significant_de_genes['names'] == gene].iloc[0]
            
            # Save gene data
            dotplot_data[gene] = {
                'logFC': gene_stats['logfoldchanges'],
                'pval': gene_stats['pvals'],
                'pval_adj': gene_stats['pvals_adj'],
                'mean_expr_this_type': mean_expr_this_type,
                'mean_expr_other_types': mean_expr_other_types,
                'frac_expr_this_type': frac_expr_this_type,
                'frac_expr_other_types': frac_expr_other_types
            }
        
        # Save the results
        cell_types_within_disease_results.append(result_row)
        cell_types_within_disease_dotplot_data[(cell_type, disease)] = dotplot_data
        
        # Save significant DE genes to CSV
        if len(significant_de_genes) > 0:
            # Create directory structure for CSVs
            degs_dir = os.path.join(output_path, 'DEGs', 'cell_types_within_disease')
            if not os.path.exists(degs_dir):
                os.makedirs(degs_dir, exist_ok=True)
            
            # Create a safe filename
            safe_cell_type = ''.join(c if c.isalnum() else '_' for c in str(cell_type))
            safe_disease = ''.join(c if c.isalnum() else '_' for c in str(disease))
            
            filename = f"{safe_cell_type}_in_{safe_disease}_DEGs.csv"
            csv_path = os.path.join(degs_dir, filename)
            significant_de_genes.to_csv(csv_path, index=False)

# Save cell types within disease results
if cell_types_within_disease_results:
    # Save summary DataFrame
    summary_df = pd.DataFrame(cell_types_within_disease_results)
    summary_csv = os.path.join(output_path, f"{args.dataset_name}_cell_types_within_disease_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved cell types within disease summary to: {summary_csv}")
    
    # Create detailed dotplot data DataFrame
    dotplot_rows = []
    for (cell_type, disease), genes_data in cell_types_within_disease_dotplot_data.items():
        for gene, data in genes_data.items():
            row = {
                'Cell_Type': cell_type,
                'Disease': disease,
                'Gene': gene,
                'logFC': data['logFC'],
                'pval': data['pval'],
                'pval_adj': data['pval_adj'],
                'mean_expr_this_type': data['mean_expr_this_type'],
                'mean_expr_other_types': data['mean_expr_other_types'],
                'frac_expr_this_type': data['frac_expr_this_type'],
                'frac_expr_other_types': data['frac_expr_other_types']
            }
            dotplot_rows.append(row)
    
    if dotplot_rows:
        dotplot_df = pd.DataFrame(dotplot_rows)
        dotplot_csv = os.path.join(output_path, f"{args.dataset_name}_cell_types_within_disease_dotplot_data.csv")
        dotplot_df.to_csv(dotplot_csv, index=False)
        print(f"Saved detailed within-disease dotplot data to: {dotplot_csv}")
    
    # Create a heatmap showing the number of DE genes for each cell type within each disease
    pivot_df = summary_df.pivot(index='Cell_Type', columns='Disease', values='Num_DE_Genes')
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='g')
    plt.title('Number of DE Genes for Each Cell Type Within Each Disease')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cell_types_within_disease_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("\nAll differential expression analyses completed!")
print(f"Results saved to: {output_path}")