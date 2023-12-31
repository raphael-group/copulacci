import scanpy as sc
import pandas as pd
from anndata import AnnData
import squidpy as sq
import networkx as nx
import tqdm
import scipy
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Union


# Construct spatial network
def construct_spatial_network(
    adata: AnnData,
    data_type: str = "vanilla_visium",
    n_neighs: int = 10,
    n_rings: int = 1,
    radius: float = 1
) :
    """
    Construct spatial network from spatial data.
    Arguments:
        adata: AnnData object
        network_construction_method: Method to construct the spatial network. Either "deluanay" or "knn"
    Returns:
        AnnData object
    """
    print("Constructing spatial network with {}".format(data_type), flush=True)
    if (data_type == "vanilla_visium"):
        sq.gr.spatial_neighbors(adata)
    elif (data_type == "visium"):
        sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid")
    else:
        sq.gr.spatial_neighbors(
            adata, radius = radius, 
            n_neighs = n_neighs, coord_type="generic")
    
    # construct network
    G = nx.from_scipy_sparse_array(adata.obsp["spatial_connectivities"])
    adata.uns['spatial_network'] = G



# Construct boundary between two groups
def construct_boundary(
    adata: AnnData,
    G: nx.Graph = None,
    weight_mat: scipy.sparse._csr.csr_matrix = None,
    domanin_name: str = "celltype",
    boundary_type: str = "Internal",
    add_self_loops: bool = True,
    force_recalculate: bool = False,
    n_rings: int = 3,
    n_neighs: int = 10,
    radius: float = 1,
    data_type: str = "vanilla_visium"
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
    G_was_given = True

    if G is None:
        if (("spatial_network" not in adata.uns) or force_recalculate):
            construct_spatial_network(
                adata, 
                data_type = data_type,
                n_neighs = n_neighs,
                n_rings = n_rings,
                radius = radius
            )
        G = adata.uns["spatial_network"]
        G_was_given = False
    
    # update node names
    
    if weight_mat is None:
        weight_mat = nx.adjacency_matrix(
            adata.uns["spatial_network"]
        )
    node_dict = {}
    print('relabeling nodes')
    G_with_names = G.copy()
    for node in G.nodes():
        node_dict[node] = adata.obs_names[node]
    G_with_names = nx.relabel_nodes(G_with_names, node_dict)
    adata.uns["spatial_network_names"] = G_with_names
    boundary_cell_type = []
    
    for u, v in tqdm.tqdm(G.edges()):

        boundary_cell_type += [[ 
            adata.obs.iloc[u].name, 
            adata.obs.iloc[v].name, 
            adata.obs.iloc[u][domanin_name],
            adata.obs.iloc[v][domanin_name],
            weight_mat[u,v]
        ]]
        if u != v:
            boundary_cell_type += [[ 
                    adata.obs.iloc[v].name, 
                    adata.obs.iloc[u].name, 
                    adata.obs.iloc[v][domanin_name],
                    adata.obs.iloc[u][domanin_name],
                    weight_mat[v,u]
            ]] 
        
        # if (adata.obs.iloc[u][domanin_name] != adata.obs.iloc[v][domanin_name]):
        #     boundary_cell_type += [[ 
        #         adata.obs.iloc[v].name, 
        #         adata.obs.iloc[u].name, 
        #         adata.obs.iloc[v][domanin_name],
        #         adata.obs.iloc[u][domanin_name]  
        # ]]

    # Adding self loops
    if ((not G_was_given) and add_self_loops):
        for u in tqdm.tqdm(G.nodes()):
            boundary_cell_type += [[ 
                adata.obs.iloc[u].name, 
                adata.obs.iloc[u].name, 
                adata.obs.iloc[u][domanin_name],
                adata.obs.iloc[u][domanin_name],
                0
            ]]
    def determine_boundary(x):
        if (x[2] != x[3]):
            return("External")
        else:
            return("Internal")
    boundary_df = pd.DataFrame(boundary_cell_type, columns=["cell1", "cell2", "celltype1", "celltype2", "distance"])
    boundary_df["boundary_type"] = boundary_df.apply(determine_boundary, axis=1)
    boundary_df["interaction"] = boundary_df.apply(lambda x: "{}={}".format(x[2], x[3]), axis=1)
    int_edges = boundary_df.copy()

    # Remove self loops
    int_edges = int_edges.loc[ int_edges.cell1 != int_edges.cell2, : ]
    external_edges = int_edges.loc[int_edges.boundary_type == "External"].copy()
    pivoted_external = pd.concat( [
            external_edges[["cell1", "interaction"]].rename(columns = {"cell1": "cell"}), 
            external_edges[["cell2", "interaction"]].rename(columns = {"cell2": "cell"}) 
        ], 
        axis = 0, ignore_index=True
    )
    external_cells = pivoted_external.cell.unique()
    internal_edges = int_edges.loc[ 
            ~(int_edges.cell1.isin(external_cells) | int_edges.cell2.isin(external_cells)), 
        :].copy()
    # just self-loops
    self_loop_edges = boundary_df.loc[ boundary_df.cell1 == boundary_df.cell2, :].copy()
    int_edges_with_selfloops = pd.concat([internal_edges, external_edges, self_loop_edges], axis=0, ignore_index=True)
    int_edges_without_selfloops = pd.concat([internal_edges, external_edges], axis=0, ignore_index=True)

    return (int_edges_without_selfloops, int_edges_with_selfloops)
    
    # internal_groups = int_edges_new.loc[int_edges_new.boundary_type == 'Internal','interaction'].unique(
    #     ).tolist()
    
    # int_edges_new = int_edges_with_selfloops.copy()
    # # remove internal edges
    # int_edges_new = int_edges_new.loc[int_edges_new.boundary_type == 'External',:]
    # # One without self loops
    # int_edges_without_selfloops = int_edges_new_

    # # For each node in the graph add a self loof if it's
    # # not already present.

    # if boundary_type == "Internal":
    #     int_edges_new_with_selfloops = int_edges_new.copy()
    #     for group_pair in internal_groups:
    #         # get cells within a group
    #         df = int_edges_new.loc[int_edges_new.interaction == group_pair]
    #         self_loops = pd.DataFrame(columns = df.columns)
    #         all_cells = list(set(df.cell1).union(set(df.cell2)))
    #         self_loops['cell1'] = all_cells
    #         self_loops['cell2'] = all_cells
    #         self_loops['celltype1'] = df.iloc[0,2]
    #         self_loops['celltype2'] = df.iloc[0,3]
    #         self_loops['boundary_type'] = df.iloc[0,4]
    #         self_loops['interaction'] = df.iloc[0,5]
    #         self_loops['distance'] = 1
    #         int_edges_new_with_selfloops = pd.concat([int_edges_new_with_selfloops,self_loops.copy()], axis = 0)
    #     return (int_edges_new, int_edges_new_with_selfloops)
    # else:
    #     # Don't add self loops
    #     return (int_edges_new, None)


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
    #Drop this
    pass



# Prepare data list
def prepare_data_list(
    count_df: pd.DataFrame,
    int_edges_new_with_selfloops: pd.DataFrame,
    groups: list = None,
    lig_rec_pair_list = None,
    heteromeric = False,
    lig_list = None,
    rec_list = None,
    summarization = "min",
    record_distance = True
) -> tuple:
    
    if groups is None:
        groups = int_edges_new_with_selfloops.interaction.unique().tolist()
    if not heteromeric:
        if lig_rec_pair_list is None:
            raise ValueError("lig_rec_pair_list must be provided")
        data_list_dict = {}
        umi_sums = {}
        dist_list_dict = {}
        for g1 in tqdm.tqdm(groups):
            g1_dict = {}
            lr_pairs_g1 = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    ["cell1", "cell2"]
                ]
            dist_list_dict[g1] = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    "distance"
            ].values
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
        dist_list_dict = {}
        for g1 in tqdm.tqdm(groups):
            g1_dict = {}
            lr_pairs_g1 = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    ["cell1", "cell2"]
                ]
            dist_list_dict[g1] = int_edges_new_with_selfloops.loc[
                    int_edges_new_with_selfloops.interaction == g1,
                    "distance"
            ].values
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

    if record_distance:
        return (data_list_dict, umi_sums, dist_list_dict)
    else:
        return (data_list_dict, umi_sums, None)


# Prepare data list
def prepare_data_list_cellype(
    count_df: pd.DataFrame,
    int_edges_new_with_selfloops: pd.DataFrame,
    source_celltype = None ,
    target_celltype = None ,
    lig_rec_pair_list = None,
    heteromeric = False,
    lig_list = None,
    rec_list = None,
    summarization = "min",
    record_distance = True
) -> tuple:

    if source_celltype is None and target_celltype is None:
        raise ValueError("source_celltype or target_celltype must be provided")

    data_list = []
    umi_sums = {}
    dist_list = []

    if source_celltype is None:
        celltype = target_celltype
        lr_pairs_g1 = int_edges_new_with_selfloops.loc[
            int_edges_new_with_selfloops.celltype2 == celltype,
            ["cell1", "cell2"]
        ].copy()
        
        dist_list  = int_edges_new_with_selfloops.loc[
                int_edges_new_with_selfloops.celltype2 == celltype,
                "distance"
        ].values
        _umi_sum_lig = count_df.loc[ lr_pairs_g1.cell1.values, : ].sum(1).values
        _umi_sum_rec = count_df.loc[ lr_pairs_g1.cell2.values, : ].sum(1).values
    elif target_celltype is None:
        celltype = source_celltype
        lr_pairs_g1 = int_edges_new_with_selfloops.loc[
            int_edges_new_with_selfloops.celltype1 == celltype,
            ["cell1", "cell2"]
        ].copy()
        
        dist_list  = int_edges_new_with_selfloops.loc[
                int_edges_new_with_selfloops.celltype1 == celltype,
                "distance"
        ].values
        _umi_sum_lig = count_df.loc[ lr_pairs_g1.cell1.values, : ].sum(1).values
        _umi_sum_rec = count_df.loc[ lr_pairs_g1.cell2.values, : ].sum(1).values
    else:
        celltype = "{}={}".format(source_celltype, target_celltype)


    umi_sums['source'] = _umi_sum_lig.copy()
    umi_sums['target'] = _umi_sum_rec.copy()

    if not heteromeric:
        if lig_rec_pair_list is None:
            raise ValueError("lig_rec_pair_list must be provided")

        data_list = []
        for i,(lig, rec) in enumerate(lig_rec_pair_list):
            data_list += [
                (
                    count_df.loc[ lr_pairs_g1.cell1.values, lig ].values.astype('int'),
                    count_df.loc[ lr_pairs_g1.cell2.values, rec ].values.astype('int')
                )
                
            ]
    else:
        if lig_list is None or rec_list is None:
            raise ValueError("lig_list and rec_list must be provided")
        assert(len(lig_list) == len(rec_list))

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

    if record_distance:
        return (data_list, umi_sums, dist_list)
    else:
        return (data_list, umi_sums, None)