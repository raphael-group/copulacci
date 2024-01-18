# pylint: disable=C0103, C0114, C0301, R0914, R0915, R0912, R0913, R0911
# from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Union
from collections import namedtuple
import pandas as pd
from anndata import AnnData
import squidpy as sq
import networkx as nx
import tqdm
import scipy


SpatialParams = namedtuple('SpatialParams', ['data_type',  'coord_type', 'n_neighs', 'n_rings', 'radius', 'distance_aware', 'deluanay'])
SpatialParams.__new__.__defaults__ = ('visium', 'grid', 6, 1, None, False, False)


# Construct spatial network
def construct_spatial_network(
    adata,
    spatial_params
) :
    """
    Construct spatial network from spatial data.
    Arguments:
        adata: AnnData object
        network_construction_method: Method to construct the spatial network. Either "deluanay" or "knn"
    Returns:
        AnnData object
    """
    data_type = spatial_params.data_type
    coord_type = spatial_params.coord_type
    n_neighs = spatial_params.n_neighs
    n_rings = spatial_params.n_rings
    radius = spatial_params.radius
    distance_aware = spatial_params.distance_aware
    print(f"Constructing spatial network with {data_type}", flush=True)
    if data_type == "visium" and not distance_aware:
        sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type=coord_type)
    elif data_type == "visium" and distance_aware:
        sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type='generic')
    else:
        sq.gr.spatial_neighbors(
            adata,
            radius = radius,
            n_neighs = n_neighs,
            coord_type="generic"
        )

    # construct network
    G = nx.from_scipy_sparse_array(adata.obsp["spatial_distances"])
    adata.uns['spatial_network'] = G


# Construct boundary between two groups
def construct_boundary(
    adata: AnnData,
    G: nx.Graph = None,
    weight_mat: scipy.sparse._csr.csr_matrix = None,
    domain_name: str = "celltype",
    add_self_loops: bool = True,
    force_recalculate: bool = False,
    distance_aware: bool = False,
    spatial_params: SpatialParams = SpatialParams()
) -> tuple:
    """
    Construct boundary from spatial data and annotation.
    Parameters:
        adata: AnnData object
        G : Networkx graph
        weight_mat: Weight matrix
        domanin_name: Name of the column in adata.obs that contains the domain information
        add_self_loops: Whether to add self loops
        force_recalculate: Whether to force recalculate the spatial network
        spatial_params: SpatialParams object that contains the parameters for spatial
           network construction
    Returns:
        AnnData object
    """
    # check if domain_name is in adata.obs
    if (domain_name not in adata.obs.columns):
        raise ValueError(f"{domain_name} must be in adata.obs")
    # check pandas columns type is categorical
    if (adata.obs[domain_name].dtype.name != domain_name):
        adata.obs[domain_name] = adata.obs[domain_name].astype("category")

    if G is None:
        if weight_mat is not None:
            G = nx.from_scipy_sparse_array(weight_mat)
            adata.uns["spatial_network"] = G
        if (("spatial_network" not in adata.uns) or force_recalculate):
            construct_spatial_network(
                adata,
                spatial_params
            )
        G = adata.uns["spatial_network"]
        # Name the network
        node_dict = {}
        for node in G.nodes():
            node_dict[node] = adata.obs_names[node]
        G = nx.relabel_nodes(G, node_dict)

    if (len(G.nodes()) != len(adata.obs_names)):
        raise ValueError("Graph nodes must be same as number of cells")
    # relabel nodes if needed
    if len(set(list(G.nodes())).intersection(adata.obs_names)) != adata.obs_names.shape[0]:
        for node in G.nodes():
            node_dict[node] = adata.obs_names[node]
        G = nx.relabel_nodes(G, node_dict)

    boundary_cell_type = []
    for cell1, cell2, data in G.edges(data=True):
        if distance_aware and (cell1 != cell2) and data['weight'] == 0:
            continue
        boundary_cell_type += [[
            cell1,
            cell2,
            adata.obs.loc[cell1][domain_name],
            adata.obs.loc[cell2][domain_name],
            data['weight']
        ]]
        # Add reverse edges
        if cell1 != cell2:
            boundary_cell_type += [[
                    cell2,
                    cell1,
                    adata.obs.loc[cell2][domain_name],
                    adata.obs.loc[cell1][domain_name],
                    data['weight']
            ]]

    if add_self_loops:
        for u in tqdm.tqdm(G.nodes()):
            boundary_cell_type += [[
                u,
                u,
                adata.obs.loc[u][domain_name],
                adata.obs.loc[u][domain_name],
                0
            ]]
    def determine_boundary(x):
        if (x[2] != x[3]):
            return("External")
        return("Internal")
    def determine_selfloop(x):
        if x[0] == x[1]:
            return(True)
        return(False)

    boundary_df = pd.DataFrame(boundary_cell_type, columns=["cell1", "cell2", "celltype1",
                                                            "celltype2", "distance"]
                )
    boundary_df["boundary_type"] = boundary_df.apply(determine_boundary, axis=1)
    boundary_df["interaction"] = boundary_df.apply(lambda x: f"{x[2]}={x[3]}", axis=1)
    boundary_df["self_loop"] = boundary_df.apply(determine_selfloop, axis=1)
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


def extract_edge_from_spatial_network(
    adata: AnnData
) -> pd.DataFrame:
    """
    Extract edges from spatial network.
    Parameters:
        adata: AnnData object
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


def heteromeric_subunit_summarization(
    count_df: pd.DataFrame,
    int_edges : pd.DataFrame,
    lig: list,
    rec: list,
    summarization: str = "sum"
)   :
    """
    Summarize the count matrix for heteromeric subunits.
    Parameters:
        count_df: Count matrix
        lr_pairs_g1: Dataframe containing the edges
        lig_list: List of ligands
        rec_list: List of receptors
        summarization: Summarization method
    Returns:
        data tuple
    """
    _lig = [l for l in lig if l is not None]
    _rec = [r for r in rec if r is not None]
    if len(_lig) > 1 or len(_rec) > 1:
        if summarization == "min":
            data_tuple = (
                    count_df.loc[ int_edges.cell1.values, _lig ].min(axis=1).values.astype('int'),
                    count_df.loc[ int_edges.cell2.values, _rec ].min(axis=1).values.astype('int')
            )
        elif summarization == "max":
            data_tuple = (
                    count_df.loc[ int_edges.cell1.values, _lig ].max(axis=1).values.astype('int'),
                    count_df.loc[ int_edges.cell2.values, _rec ].max(axis=1).values.astype('int')
            )
        elif summarization == "mean":
            data_tuple = (
                    count_df.loc[ int_edges.cell1.values, _lig ].mean(axis=1).values.astype('int'),
                    count_df.loc[ int_edges.cell2.values, _rec ].mean(axis=1).values.astype('int')
            )
        elif summarization == "sum":
            data_tuple = (
                    count_df.loc[ int_edges.cell1.values, _lig ].sum(axis=1).values.astype('int'),
                    count_df.loc[ int_edges.cell2.values, _rec ].sum(axis=1).values.astype('int')
            )
        else:
            raise ValueError("summarization must be one of min, max, mean, sum")
    else:
        data_tuple = (
            count_df.loc[ int_edges.cell1.values, _lig ].values.flatten().astype('int'),
            count_df.loc[ int_edges.cell2.values, _rec ].values.flatten().astype('int')
        )
    return data_tuple


def prepare_data_list_from_spatial_network(
    count_df: pd.DataFrame,
    int_edges: pd.DataFrame,
    groups: list = None,
    lig_rec_info_df = None,
    heteromeric = False,
    lig_df = None,
    rec_df = None,
    summarization = "min",
    separate_lig_rec_type = False
):
    """
    Prepare data list from spatial network.
    Parameters:
    -----------
    count_df: Count matrix
    int_edges: dataframe containing the edges and the type of interaction
    groups: list of interaction types
    lig_rec_pair_list: list of ligand receptor pairs along with annotations
    heteromeric: whether to consider heteromeric interactions which mean ligand and receptors
        joined by a underscore
    lig_df: list of ligands
    rec_df: list of receptors
    summarization: summarization method
    seperate_lig_rec_type: whether to separate close cell-cell contact and other interactions
    """

    if groups is None:
        groups = int_edges.interaction.unique().tolist()
    if ('annotation' not in lig_rec_info_df.columns) and separate_lig_rec_type:
        raise ValueError("annotation must be in lig_rec_info_df to \
                         separate ligand and receptor type")
    # There are three type of interactions
        # 1. Cell-cell contact
        # 2. ECM receptor interaction
        # 3. Secreted signialing
    # If we are in visium then we have to consider only cell-cell contact
    # for self loops. For other interactions we have to consider both
    # self loops and other connections
    if separate_lig_rec_type:
        int_edges_selfloop = int_edges.loc[ int_edges.self_loop, : ]
    if heteromeric:
        if lig_df is None or rec_df is None:
            raise ValueError("lig_list and rec_list must be provided")
        lig_list = lig_df.values
        rec_list = rec_df.values
        assert(len(lig_list) == len(rec_list))
        data_list_dict = {}
        data_list_dict_selfloop = {}
        umi_sums = {}
        umi_sums_selfloop = {}
        dist_list_dict = {}
        dist_list_dict_selfloop = {}
        for g1 in tqdm.tqdm(groups):
            g1_dict = {}
            g1_selfloop_dict = {}
            g11, g12 = g1.split('=')
            if separate_lig_rec_type:
                # Add the close contacts first
                # Get edges for this group pair
                int_edges_selfloop_g1 = int_edges_selfloop.loc[ int_edges_selfloop.interaction == g1, : ]
                dist_list_dict_selfloop[g1] = int_edges_selfloop_g1['distance'].values
                g1_selfloop_dict[g11] = count_df.loc[ int_edges_selfloop_g1.cell1.values, : ].sum(1).values
                g1_selfloop_dict[g12] = count_df.loc[ int_edges_selfloop_g1.cell2.values, : ].sum(1).values

            int_edges_g1 = int_edges.loc[ int_edges.interaction == g1, : ]
            dist_list_dict[g1] = int_edges_g1['distance'].values
            g1_dict[g11] = count_df.loc[ int_edges_g1.cell1.values, : ].sum(1).values
            g1_dict[g12] = count_df.loc[ int_edges_g1.cell2.values, : ].sum(1).values

            data_list = []
            data_list_selfloop = []
            # Add the data for ligand receptors with close contact if separate_lig_rec_type is True
            for index, row in lig_rec_info_df.iterrows():
                lig = lig_df.loc[index].values.tolist()
                rec = rec_df.loc[index].values.tolist()
                if separate_lig_rec_type:
                    if row.annotation == 'Cell-Cell Contact':
                        data_list_selfloop += [heteromeric_subunit_summarization(count_df, int_edges_selfloop_g1,
                                                                        lig, rec, summarization)
                                ]
                    else:
                        data_list += [heteromeric_subunit_summarization(count_df, int_edges_g1,
                                                                    lig, rec, summarization)
                                ]
                else:
                    data_list += [heteromeric_subunit_summarization(count_df, int_edges_g1,
                                                                    lig, rec, summarization)
                            ]
            if separate_lig_rec_type:
                umi_sums_selfloop[g1] = g1_selfloop_dict.copy()
                data_list_dict_selfloop[g1] = data_list_selfloop.copy()
            umi_sums[g1] = g1_dict.copy()
            data_list_dict[g1] = data_list.copy()
    if separate_lig_rec_type:
        return (data_list_dict, umi_sums, dist_list_dict,
                data_list_dict_selfloop, umi_sums_selfloop, dist_list_dict_selfloop
            )
    else:
        return (data_list_dict, umi_sums, dist_list_dict, None, None)


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