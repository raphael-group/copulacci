import itertools
import sys
import os
import pandas as pd
import scanpy as sc
import spatialdm as sdm
import networkx as nx
#import gseapy_like_plot as pl
print('Loaded external modules', flush=True)
score_pair = list(itertools.combinations(['rho_zero', 'scc','global_I'],2))


# custom package import
sys.path.append("/n/fs/ragr-research/users/hirak/Projects/copulacci/src/copulacci/")
import cci
import spatial
import model
print('Loading internal modules', flush=True)

print(cci.__file__)
print(spatial.__file__)
print(model.__file__)

#sys.exit(0)

EPSILON = 1.1920929e-07
# 
print("Loading spatial data",flush=True)
# adata = sc.read_visium('/Users/hs0424/Workspace/copulacci/data/3.Human_Breast_Cancer/')
# adata.var_names_make_unique()
# annotation_meta = pd.read_csv(
#     '/Users/hs0424/Workspace/copulacci/data/3.Human_Breast_Cancer/metadata.tsv',
#     sep = '\t',
#     index_col=0
# )
# adata.obs = adata.obs.join(annotation_meta)
# adata.raw = adata.copy()
data_dir = "/n/fs/ragr-research/users/hirak/Projects/niche_project/COMMOT_paper_data/orator_paper_notebook/data/human_breast_visium/"
adata = sc.read_h5ad(os.path.join(data_dir,"adata.h5ad"))

# Same parameters as given in SDM tutorial
sdm.extract_lr(adata, 'human', min_cell=20)
adata_sdm = adata.copy()
sdm.weight_matrix(adata_sdm, l=273, single_cell=False)

# Construct the spatial neighborhood graph
G = nx.from_scipy_sparse_array(adata_sdm.obsp['nearest_neighbors'])
adata.obs['celltype'] = adata.obs.annot_type
int_edges_new, int_edges_with_selfloop = spatial.construct_boundary(
    adata,
    G = G,
    weight_mat=adata_sdm.obsp['weight']
)
df_lig_rec_linear = cci.extract_lig_rec_from_sdm(adata, allow_same_lr=True)
chosen_lr = list(set( df_lig_rec_linear.ligand.unique()).union(
    set( df_lig_rec_linear.receptor.unique() )
))
count_df = adata.raw.to_adata().to_df().loc[:,chosen_lr]
lig_list = adata.uns['ligand'].values
rec_list = adata.uns['receptor'].values


# Prepare data for running copulacci
data_list_dict, umi_sums, dist_list_dict = spatial.prepare_data_list(
    count_df,
    int_edges_new,
    heteromeric=True,
    lig_list=lig_list,
    rec_list = rec_list,
    summarization='sum'
)

# Calculate the correlation coefficient using copula
df_lig_rec = pd.concat(
   [ adata.uns['ligand'], adata.uns['receptor']],
    axis = 1
)

cop_df_dict = model.run_copula(
    data_list_dict,
    umi_sums,
    dist_list_dict,
    DT=False,
    cutoff = 0.8,
    type_run='dense',
    num_restarts=1,
    df_lig_rec=df_lig_rec,
    heteronomic=True
)

# Calculate the correlation coefficient using spatial cross correlation
# cop_df_dict = model.run_scc(
#     count_df,
#     None,
#     cop_df_dict,
#     int_edges_new,
#     groups = list(data_list_dict.keys()),
#     heteronomic=True,
#     lig_list=lig_list,
#     rec_list = rec_list,
#     summarization = "sum"
# )

print("Running normal copula",flush=True)

# # Run spatialDM for individual sub-graph 
# sdm_df_dict = model.run_sdm(
#     adata,
#     int_edges_new,
#     groups = list(cop_df_dict.keys()),
#     nproc = 20,
#     heteronomic=True
# )

# # Merge the results
# merged_res = {}
# for gpair in cop_df_dict.keys():
#     tmp1 = sdm_df_dict[gpair].copy()
#     tmp2 = cop_df_dict[gpair]
#     merged_res[gpair] = tmp1.join(tmp2, how = 'inner').copy()
#     merged_res[gpair]['celltype_direction'] = gpair

# # Concatenate the results for individual groups

# merged_df = pd.concat(merged_res.values(), axis = 0)
# merged_df.to_csv(
#     os.path.join(data_dir,'merged_df_3_methods.csv'),
#     sep = ',',
#     index = False
# )

# Calculate pvalues
data_dir_95 = "/n/fs/ragr-research/users/hirak/Projects/niche_project/COMMOT_paper_data/orator_paper_notebook/data/human_breast_visium/pvals_95"
if not os.path.exists(data_dir_95):
    os.makedirs(data_dir_95, exist_ok=True)
for gpair in data_list_dict.keys():
    print('Adding pvalue to ', gpair, flush=True)
    final_res_cop = model.add_copula_pval(
        data_list_dict,
        cop_df_dict,
        umi_sums,
        dist_list_dict,
        int_edges_new,
        count_df,
        percentile_cutoff = 95,
        n = 1000,
        groups = [gpair],
        heteronomic = True
    )
    file_name = 'final_res_copula_' + gpair + '.csv'
    
    final_res_cop.to_csv(
        os.path.join(data_dir_95, file_name),
        sep = ',',
        index = False
    )
