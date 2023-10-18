import scanpy as sc
import pandas as pd
from anndata import AnnData
import squidpy as sq
import networkx as nx
import tqdm
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Union


# Construct spatial network
def construct_spatial_network(
    adata: AnnData,
    data_type: str = "visium",
    n_neighs: int = 10
) -> AnnData:
    """
    Construct spatial network from spatial data.
    Arguments:
        adata: AnnData object
        network_construction_method: Method to construct the spatial network. Either "deluanay" or "knn"
    Returns:
        AnnData object
    """
    if (data_type == "visium"):
        sq.gr.spatial_neighbors(adata)
    else:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")
    
    # construct network
    G = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])
    adata.uns['spatial_network'] = G

    return(adata)


# Construct boundary between two groups
def construct_boundary(
    adata: AnnData,
    domanin_name: str = "celltype",
    boundary_type: str = "Internal"
) -> tuple:
    """
    Construct boundary from spatial data and annotation.
    Parameters:
        adata: AnnData object
        annotation: Annotation dataframe with cell barcodes as index
    Returns:
        AnnData object
    """
    # check if domain_name is in adata.obs
    if (domanin_name not in adata.obs.columns):
        raise ValueError("{} must be in adata.obs".format(domanin_name))
    # check pandas columns type is categorical
    if (adata.obs[domanin_name].dtype.name != domanin_name):
        adata.obs[domanin_name] = adata.obs[domanin_name].astype("category")
    
    # Go over all combinations
    # for (i,j) in itertools.combinations(cell_types, 2):
    if ("spatial_network" not in adata.uns):
        adata = construct_spatial_network(adata)
    G = adata.uns["spatial_network"]
    # update node names
    node_dict = {}
    print('relabeling nodes')
    G_with_names = G.copy()
    for node in G.nodes():
        node_dict[node] = adata.obs_names[node]
    G_with_names = nx.relabel_nodes(G_with_names, node_dict)
    adata.uns["spatial_network_names"] = G_with_names
    boundary_cell_type = []
    
    for u,v in G.edges():
        boundary_cell_type += [[ 
            adata.obs.iloc[u].name, 
            adata.obs.iloc[v].name, 
            adata.obs.iloc[u][domanin_name],
            adata.obs.iloc[v][domanin_name]  
        ]]
        boundary_cell_type += [[ 
                adata.obs.iloc[v].name, 
                adata.obs.iloc[u].name, 
                adata.obs.iloc[v][domanin_name],
                adata.obs.iloc[u][domanin_name]  
        ]]
        
        # if (adata.obs.iloc[u][domanin_name] != adata.obs.iloc[v][domanin_name]):
        #     boundary_cell_type += [[ 
        #         adata.obs.iloc[v].name, 
        #         adata.obs.iloc[u].name, 
        #         adata.obs.iloc[v][domanin_name],
        #         adata.obs.iloc[u][domanin_name]  
        # ]]
    def determine_boundary(x):
        if (x[2] != x[3]):
            return("External")
        else:
            return("Internal")
    boundary_df = pd.DataFrame(boundary_cell_type, columns=["cell1", "cell2", "celltype1", "celltype2"])
    boundary_df["boundary_type"] = boundary_df.apply(determine_boundary, axis=1)
    boundary_df["interaction"] = boundary_df.apply(lambda x: "{}={}".format(x[2], x[3]), axis=1)
    int_edges = boundary_df
    external_edges = int_edges.loc[int_edges.boundary_type == "External"]
    pivoted_external = pd.concat( [
            external_edges[["cell1","interaction"]].rename(columns = {"cell1": "cell"}), 
            external_edges[["cell2","interaction"]].rename(columns = {"cell2": "cell"}) 
        ], 
        axis = 0, ignore_index=True
    )
    external_cells = pivoted_external.cell.unique()
    internal_edges = int_edges.loc[ 
            ~(int_edges.cell1.isin(external_cells) | int_edges.cell2.isin(external_cells)), 
        :]
    int_edges_new = pd.concat([internal_edges, external_edges], axis=0, ignore_index=True)
    internal_groups = int_edges_new.loc[int_edges_new.boundary_type == 'Internal','interaction'].unique(
        ).tolist()
    
    if boundary_type == "Internal":
        int_edges_new_with_selfloops = int_edges_new.copy()
        for group_pair in internal_groups:
            # get cells within a group
            df = int_edges_new.loc[int_edges_new.interaction == group_pair]
            self_loops = pd.DataFrame(columns = df.columns)
            all_cells = list(set(df.cell1).union(set(df.cell2)))
            self_loops['cell1'] = all_cells
            self_loops['cell2'] = all_cells
            self_loops['celltype1'] = df.iloc[0,2]
            self_loops['celltype2'] = df.iloc[0,3]
            self_loops['boundary_type'] = df.iloc[0,4]
            self_loops['interaction'] = df.iloc[0,5]
            int_edges_new_with_selfloops = pd.concat([int_edges_new_with_selfloops,self_loops.copy()], axis = 0)
        return (int_edges_new, int_edges_new_with_selfloops)
    else:
        # Don't add self loops
        return (int_edges_new, None)


def extract_edge_from_spatial_network(
    adata: AnnData
) -> pd.DataFrame:
    """
    """
    if 'weight' not in adata.obsp.keys():
        raise ValueError("Weight must be in adata.obsp")
    G = nx.from_scipy_sparse_array(adata.obsp["weight"])
    edges = []
    for u,v in G.edges():
        if G[u][v]['weight'] > 0:
            edges += [[ 
                adata.obs.iloc[u].name, 
                adata.obs.iloc[v].name, 
                G[u][v]['weight']
            ]]
            edges += [[ 
                adata.obs.iloc[v].name, 
                adata.obs.iloc[u].name, 
                G[v][u]['weight']
            ]]
    edge_df = pd.DataFrame(edges, columns=["cell1", "cell2", "weight"])
    return(edge_df)

def prepare_data_list_from_spatial_network():
    #TODO create data list from spatial network with distance
    pass



# Prepare data list
def prepare_data_list(
    count_df: pd.DataFrame,
    int_edges_new_with_selfloops: pd.DataFrame,
    groups: list,
    lig_rec_pair_list = None,
    heteromeric = False,
    lig_list = None,
    rec_list = None,
    summarization = "min"
) -> tuple:
    
    if not heteromeric:
        data_list_dict = {}
        umi_sums = {}
        for g1 in tqdm.tqdm(groups):
            g1_dict = {}
            lr_pairs_g1 = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    ["cell1", "cell2"]
                ]
            g11, g12 = g1.split('=')
            _umi_sum_lig = count_df.loc[ lr_pairs_g1.cell1.values, : ].sum(1).values
            _umi_sum_rec = count_df.loc[ lr_pairs_g1.cell2.values, : ].sum(1).values
            g1_dict[g11] = _umi_sum_lig.copy()
            g1_dict[g12] = _umi_sum_rec.copy()
            
            data_list = []
            for i,(lig, rec) in enumerate(lig_rec_pair_list):
                data_list += [
                    (
                        count_df.loc[ lr_pairs_g1.cell1.values, lig ].values.astype('int'),
                        count_df.loc[ lr_pairs_g1.cell2.values, rec ].values.astype('int')
                    )
                    
                ]
            umi_sums[g1] = g1_dict.copy()
            assert(len(g1_dict[g11]) == len(_umi_sum_lig))
            assert(len(g1_dict[g12]) == len(_umi_sum_rec))
            data_list_dict[g1] = data_list.copy()
    else:
        if lig_list is None or rec_list is None:
            raise ValueError("lig_list and rec_list must be provided")
        assert(len(lig_list) == len(rec_list))
        data_list_dict = {}
        umi_sums = {}
        for g1 in tqdm.tqdm(groups):
            g1_dict = {}
            lr_pairs_g1 = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    ["cell1", "cell2"]
                ]
            g11, g12 = g1.split('=')
            _umi_sum_lig = count_df.loc[ lr_pairs_g1.cell1.values, : ].sum(1).values
            _umi_sum_rec = count_df.loc[ lr_pairs_g1.cell2.values, : ].sum(1).values
            g1_dict[g11] = _umi_sum_lig.copy()
            g1_dict[g12] = _umi_sum_rec.copy()
            
            data_list = []
            for i,(lig, rec) in enumerate(zip(lig_list, rec_list)):
                _lig = [l for l in lig if l != None]
                _rec = [r for r in rec if r != None]
                if summarization == "min":
                    data_list += [
                        (
                            count_df.loc[ lr_pairs_g1.cell1.values, _lig ].min(axis=1).values.astype('int'),
                            count_df.loc[ lr_pairs_g1.cell2.values, _rec ].min(axis=1).values.astype('int')
                        )
                        
                    ]
                elif summarization == "max":
                    data_list += [
                        (
                            count_df.loc[ lr_pairs_g1.cell1.values, _lig ].max(axis=1).values.astype('int'),
                            count_df.loc[ lr_pairs_g1.cell2.values, _rec ].max(axis=1).values.astype('int')
                        )
                        
                    ]
                elif summarization == "mean":
                    data_list += [
                        (
                            count_df.loc[ lr_pairs_g1.cell1.values, _lig ].mean(axis=1).values.astype('int'),
                            count_df.loc[ lr_pairs_g1.cell2.values, _rec ].mean(axis=1).values.astype('int')
                        )
                        
                    ]
                elif summarization == "sum":
                    data_list += [
                        (
                            count_df.loc[ lr_pairs_g1.cell1.values, _lig ].sum(axis=1).values.astype('int'),
                            count_df.loc[ lr_pairs_g1.cell2.values, _rec ].sum(axis=1).values.astype('int')
                        )
                        
                    ]
                else:
                    raise ValueError("summarization must be one of min, max, mean, sum")

            umi_sums[g1] = g1_dict.copy()
            assert(len(g1_dict[g11]) == len(_umi_sum_lig))
            assert(len(g1_dict[g12]) == len(_umi_sum_rec))
            data_list_dict[g1] = data_list.copy()


    return (data_list_dict, umi_sums)


def prepare_differential_data_list(
    count_df: pd.DataFrame,
    int_edges_new_with_selfloops: pd.DataFrame,
    lig_rec_pair_list: list,
    groups: list,
):
    # Prepare data list for differential analysis
    data_list_dict = {}
    umi_sums = {}
    g1, g2 = groups[0], groups[1]
    # null model where x and y share the same correlation coeff