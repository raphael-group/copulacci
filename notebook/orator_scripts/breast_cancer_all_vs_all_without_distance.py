import itertools
import sys
import os
import pandas as pd
import scanpy as sc
import spatialdm as sdm
import networkx as nx

score_pair = list(itertools.combinations(['rho_zero', 'scc','global_I'],2))

sys.path.append("/n/fs/ragr-research/users/hirak/Projects/copulacci/src/copulacci/")
import cci
import spatial
import model

print('Loading internal modules', flush=True)
print("Loading spatial data",flush=True)

data_dir = "/n/fs/ragr-research/users/hirak/Projects/niche_project/COMMOT_paper_data/orator_paper_notebook/data/human_breast_visium/"
adata = sc.read_h5ad(os.path.join(data_dir,"adata.h5ad"))

sdm.extract_lr(adata, 'human', min_cell=20)
adata.obs['celltype'] = adata.obs.annot_type

print("Constructing boundary",flush=True)
int_edges_new, int_edges_with_selfloop = spatial.construct_boundary(
    adata
)

print("Extracting ligand receptor pairs",flush=True)
df_lig_rec_linear = cci.extract_lig_rec_from_sdm(adata, allow_same_lr=True)
chosen_lr = list(set( df_lig_rec_linear.ligand.unique()).union(
    set( df_lig_rec_linear.receptor.unique() )
))
count_df = adata.raw.to_adata().to_df().loc[:,chosen_lr]
lig_list = adata.uns['ligand'].values
rec_list = adata.uns['receptor'].values
df_lig_rec = pd.concat(
   [ adata.uns['ligand'], adata.uns['receptor']],
    axis = 1
)

print("Preparing data for copulacci",flush=True)
data_list_dict, umi_sums, dist_list_dict = spatial.prepare_data_list(
    count_df,
    int_edges_with_selfloop,
    heteromeric=True,
    lig_list=lig_list,
    rec_list = rec_list,
    summarization='sum'
)

print("Running copulacci",flush=True)
cop_df_dict = model.run_copula(
    data_list_dict,
    umi_sums,
    DT=False,
    cutoff = 0.8,
    type_run='dense',
    num_restarts=2,
    df_lig_rec=df_lig_rec,
    heteronomic=True
)


data_dir_90 = "/n/fs/ragr-research/users/hirak/Projects/niche_project/COMMOT_paper_data/orator_paper_notebook/data/human_breast_visium_wo_dist/pvals_90"
if not os.path.exists(data_dir_90):
    os.makedirs(data_dir_90, exist_ok=True)
for gpair in data_list_dict.keys():
    print('Adding pvalue to ', gpair, flush=True)
    final_res_cop = model.add_copula_pval(
        data_list_dict,
        cop_df_dict,
        umi_sums,
        int_edges_new_with_selfloops=int_edges_with_selfloop,
        count_df=count_df,
        percentile_cutoff = 90,
        n = 500,
        groups = [gpair],
        heteronomic = True
    )
    file_name = 'final_res_copula_' + gpair + '.csv'
    final_res_cop.to_csv(
        os.path.join(data_dir_90, file_name),
        sep = ',',
        index = False
    )
    