import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
from copulacci import model


def create_param_grid(
    rho_vec = np.linspace(-0.9, 0.9, 19),
    sparse_fraction = np.array([0.1, 0.3 , 0.5, 1.0]),
    mu_x_vec = np.array([-8, -5, -3]),
    mu_y_vec = np.array([-8, -5, -3])
):
    param_values = [
        sparse_fraction,
        mu_x_vec,
        mu_y_vec,
        rho_vec
    ]

    param_grids = np.meshgrid(*param_values, indexing='ij')
    grid_points = np.column_stack([grid.ravel() for grid in param_grids])
    grid_points_df = pd.DataFrame(
        grid_points,
        columns=['sparse_frac', 'mu_x', 'mu_y' ,'rho']
    )
    grid_points_df.loc[:, 'ind'] = np.array(range(grid_points_df.shape[0])).astype('int')
    return grid_points_df


def create_param_grid_spatial(
    rho_zero_vec = np.linspace(-0.9, 0.9, 19),
    rho_one_vec = np.array([0.01,0.05]), # make it variable later
    sparse_fraction = np.array([0.1, 0.3 , 0.5, 1.0]),
    mu_x_vec = np.array([-8, -5, -3]),
    mu_y_vec = np.array([-8, -5, -3])
):
    param_values = [
        sparse_fraction,
        mu_x_vec,
        mu_y_vec,
        rho_zero_vec,
        rho_one_vec
    ]


    param_grids = np.meshgrid(*param_values, indexing='ij')
    grid_points = np.column_stack([grid.ravel() for grid in param_grids])
    grid_points_df = pd.DataFrame(
        grid_points,
        columns=['sparse_frac', 'mu_x', 'mu_y' ,'rho_zero', 'rho_one']
    )
    grid_points_df.loc[:, 'ind'] = np.array(range(grid_points_df.shape[0])).astype('int')
    return grid_points_df


def sim_non_spatial(
    n_array,
    grid_points_df
):


    pseudo_count = pd.DataFrame()

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho, global_idx = row[1]
        _n_array = (n_array * sparse_frac).astype('int')


        sample = model.sample_from_copula(
            _n_array,
            mu_x,
            mu_y,
            rho
        )
        sample.columns = ['L'+str(int(global_idx)), 'R'+str(int(global_idx))]
        pseudo_count = pd.concat([pseudo_count, sample.copy()], axis = 1)

    return pseudo_count


def create_spatial_grid(nop=100):
    nop = 100
    N = nop ** 2

    # Generate positions
    pos = np.array(np.meshgrid(np.arange(1, int(np.sqrt(N)) + 1),
                            np.arange(1, int(np.sqrt(N)) + 1))).T.reshape(-1, 2)
    pos = np.concatenate((pos, pos[::-1]), axis=0)
    pos = np.unique(pos, axis=0)

    # Assign row and column names
    pos_names = [f'cell{i+1}' for i in range(N)]
    df_pos = pd.DataFrame(pos, columns=['x', 'y'])
    df_pos.index = pos_names
    #pos = np.column_stack((pos, pos_names))

    # Jitter
    pos_j = df_pos + np.random.uniform(-0.2, 0.2, size = (N,2))

    # Induce warping
    #pos_w = 1.01 ** pos_j
    pos_w = pos_j
    return pos_w


def sim_simple_spatial_distance(pos_w,nop):
    start_x_vec = list(range(2500,4500,100))
    gaps = list(range(40,0,-2))
    sim_coords = pos_w[[ 'x', 'y']].values
    sim_edge_list = []
    for i in range(len(gaps)):
        start_x = start_x_vec[i]
        gap = gaps[i]

        sim_edge_list += [(i,j,euclidean_distances(
            sim_coords[np.newaxis, i, :],
            sim_coords[np.newaxis, j, :]
        )[0][0],gap) for (i,j) in zip(list(range(start_x , start_x + nop)), list(range(start_x + gap * nop, start_x + (gap+1) * nop)) ) ]

    sim_edge_list_df = pd.DataFrame(sim_edge_list,
                                    columns = [
                                        'source', 'target',
                                        'distance', 'gap'])
    return sim_edge_list_df


def simulate_simple_spatial(
        n_array_sum,
        grid_points_df,
        sim_edge_list_df,
        n_jobs = 20
    ):
    arg_list = grid_points_df.values.tolist()
    def copula_dist_caller(args):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, global_idx = args
        _n_array = (n_array_sum * sparse_frac).astype('int')
        coeff_list = rho_zero * np.exp(-1 * sim_edge_list_df.distance * rho_one)

        sample = model.sample_from_copula_dist(
            _n_array,
            mu_x,
            mu_y,
            coeff_list
        )
        sample.columns = ['L'+str(int(global_idx)), 'R'+str(int(global_idx))]
        return sample

    samples = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(copula_dist_caller)(
                    args) for args in arg_list)
    pseudo_count = pd.concat(samples, axis = 1)
    return pseudo_count


def simulate_simple_grad_spatial(
        n_array_sum,
        grid_points_df,
        sim_edge_list_df,
        n_jobs = 20
    ):
    def copula_dist_caller(args):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, global_idx = args
        _n_array = (n_array_sum * sparse_frac).astype('int')
        coeff_list = rho_zero * np.exp(-1 * sim_edge_list_df.distance * rho_one)

        sample = model.sample_from_copula_grad(
            _n_array,
            mu_x,
            mu_y,
            coeff_list,
            sim_edge_list_df.gap.values,
            t = 30
        )
        sample.columns = ['L'+str(int(global_idx)), 'R'+str(int(global_idx))]
        return sample

    arg_list = grid_points_df.values.tolist()
    samples = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(copula_dist_caller)(
                    args) for args in arg_list)
    pseudo_count = pd.concat(samples, axis = 1)
    return pseudo_count


def prepare_df(
    opt_res,
    data_list,
    pseudo_count,
    grid_points_df,
    n_array_sum
):

    count_df_norm = pseudo_count.div(pseudo_count.sum(1), axis = 0)
    count_df_norm_log = np.log( count_df_norm + 1 )

    data_list_norm = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_norm += [
            (
                count_df_norm.loc[:, 'L'+str(int(i))],
                count_df_norm.loc[:, 'R'+str(int(i))]
            )
        ]
        i += 1

    data_list_log = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_log += [
            (
                count_df_norm_log.loc[:, 'L'+str(int(i))],
                count_df_norm_log.loc[:, 'R'+str(int(i))]
            )
        ]


    spr = []
    spr2 = []
    spr3 = []
    spr4 = []
    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        spr += [stats.spearmanr(x, y).correlation]

    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr2 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_norm:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr3 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_log:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr4 += [stats.spearmanr(x, y).correlation]


    opt_res_df = pd.DataFrame(opt_res, columns=['pred_corr', 'pred_mu_x','pred_mu_y','method'])
    zero_ratio = (pseudo_count == 0).mean()
    opt_res_df.loc[:,'zero_ratio'] = [(zero_ratio.iloc[i] + zero_ratio.iloc[i + 1]) / 2 for i in range(0, len(zero_ratio) - 1, 2)]

    opt_res_df.loc[:,'zr_cat'] = pd.cut(opt_res_df.zero_ratio, bins=4, labels = ['below_25',
                                                                            '25_50',
                                                                            '50_75',
                                                                             '75_100'
                                                                            ])

    opt_res_df = opt_res_df[['pred_corr', 'pred_mu_x','pred_mu_y','method','zr_cat']].copy()
    opt_res_df = pd.concat([grid_points_df, opt_res_df], axis = 1)
    opt_res_df['spearman'] = spr
    opt_res_df['pearson'] = spr2
    opt_res_df['spearman_log_norm'] = spr4
    opt_res_df['pearson_norm'] = spr3
    opt_res_df = opt_res_df.loc[opt_res_df.method == 'copula'].copy()
    opt_res_df = opt_res_df.drop('method', axis=1)

    opt_res_df['copula_diff'] = opt_res_df.rho - opt_res_df.pred_corr
    opt_res_df['spearman_diff'] = opt_res_df.rho - opt_res_df.spearman
    opt_res_df['pearson_diff'] = opt_res_df.rho - opt_res_df.pearson
    opt_res_df['spearman_log_norm_diff'] = opt_res_df.rho - opt_res_df.spearman_log_norm
    opt_res_df['pearson_norm_diff'] = opt_res_df.rho - opt_res_df.pearson_norm
    #opt_res_df= opt_res_df.fillna(0)



    opt_res_df.loc[:,'orig_index'] = opt_res_df.index
    res_df_melted = pd.melt(opt_res_df,
            id_vars = ['mu_x','mu_y', 'zr_cat', 'sparse_frac','rho','orig_index'],
            value_vars=['copula_diff', 'spearman_diff','pearson_diff','spearman_log_norm_diff','pearson_norm_diff'],
            var_name = 'method', value_name = 'difference')

    opt_res_df['method_diff'] = abs(opt_res_df.copula_diff) - abs(opt_res_df.spearman_diff)

    return (res_df_melted, opt_res_df)


def prepare_df_dist(
        opt_res,
        data_list,
        pseudo_count,
        grid_points_df,
        n_array_sum
    ):

    count_df_norm = pseudo_count.div(pseudo_count.sum(1), axis = 0)
    count_df_norm_log = np.log( count_df_norm + 1 )

    data_list_norm = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_norm += [
            (
                count_df_norm.loc[:, 'L'+str(int(i))],
                count_df_norm.loc[:, 'R'+str(int(i))]
            )
        ]
        i += 1

    data_list_log = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_log += [
            (
                count_df_norm_log.loc[:, 'L'+str(int(i))],
                count_df_norm_log.loc[:, 'R'+str(int(i))]
            )
        ]


    spr = []
    spr2 = []
    spr3 = []
    spr4 = []
    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        spr += [stats.spearmanr(x, y).correlation]

    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr2 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_norm:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr3 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_log:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr4 += [stats.spearmanr(x, y).correlation]


    opt_res_df = pd.DataFrame(opt_res, columns=['pred_corr_zero', 'pred_corr_one' ,'pred_mu_x','pred_mu_y','method'])

    zero_ratio = (pseudo_count == 0).mean()
    opt_res_df.loc[:,'zero_ratio'] = [(zero_ratio.iloc[i] + zero_ratio.iloc[i + 1]) / 2 for i in range(0, len(zero_ratio) - 1, 2)]

    opt_res_df.loc[:,'zr_cat'] = pd.cut(opt_res_df.zero_ratio, bins=4, labels = ['below_25',
                                                                            '25_50',
                                                                            '50_75',
                                                                             '75_100'
                                                                            ])

    opt_res_df = opt_res_df[['pred_corr_zero', 'pred_corr_one', 'pred_mu_x','pred_mu_y','method','zr_cat']].copy()

    opt_res_df = pd.concat([grid_points_df, opt_res_df], axis = 1)
    opt_res_df['spearman'] = spr
    opt_res_df['pearson'] = spr2
    opt_res_df['spearman_log_norm'] = spr4
    opt_res_df['pearson_norm'] = spr3
    opt_res_df = opt_res_df.loc[opt_res_df.method == 'copula'].copy()
    opt_res_df = opt_res_df.drop('method', axis=1)


    opt_res_df['copula_diff'] = opt_res_df.rho_zero - opt_res_df.pred_corr_zero
    opt_res_df['copula_one_diff'] = opt_res_df.rho_zero - opt_res_df.pred_corr_one
    opt_res_df['spearman_diff'] = opt_res_df.rho_zero - opt_res_df.spearman
    opt_res_df['pearson_diff'] = opt_res_df.rho_zero - opt_res_df.pearson
    opt_res_df['spearman_log_norm_diff'] = opt_res_df.rho_zero - opt_res_df.spearman_log_norm
    opt_res_df['pearson_norm_diff'] = opt_res_df.rho_zero - opt_res_df.pearson_norm
    #opt_res_df= opt_res_df.fillna(0)


    opt_res_df.loc[:,'orig_index'] = opt_res_df.index

    res_df_melted = pd.melt(opt_res_df,
            id_vars = ['mu_x','mu_y', 'zr_cat', 'sparse_frac','rho_zero', 'rho_one','orig_index'],
            value_vars=['copula_diff', 'spearman_diff','pearson_diff','spearman_log_norm_diff','pearson_norm_diff'],
            var_name = 'method', value_name = 'difference')



    opt_res_df['method_diff'] = abs(opt_res_df.copula_diff) - abs(opt_res_df.spearman_diff)

    return (res_df_melted, opt_res_df)


def prepare_df_grad(
        opt_res,
        data_list,
        pseudo_count,
        grid_points_df,
        n_array_sum
    ):

    count_df_norm = pseudo_count.div(pseudo_count.sum(1), axis = 0)
    count_df_norm_log = np.log( count_df_norm + 1 )

    data_list_norm = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_norm += [
            (
                count_df_norm.loc[:, 'L'+str(int(i))],
                count_df_norm.loc[:, 'R'+str(int(i))]
            )
        ]
        i += 1

    data_list_log = []

    for row in tqdm.tqdm(grid_points_df.iterrows()):
        sparse_frac, mu_x, mu_y, rho_zero, rho_one, i = row[1]
        _n_array = (n_array_sum * sparse_frac).astype('int')
        data_list_log += [
            (
                count_df_norm_log.loc[:, 'L'+str(int(i))],
                count_df_norm_log.loc[:, 'R'+str(int(i))]
            )
        ]


    spr = []
    spr2 = []
    spr3 = []
    spr4 = []
    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        spr += [stats.spearmanr(x, y).correlation]

    for (x,y,_,_,_n_array) in data_list:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr2 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_norm:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr3 += [stats.pearsonr(x, y).correlation]

    for (x,y) in data_list_log:
        #inds = np.where((x > 0) | (y > 0))[0]
        #spr2 += [stats.spearmanr(( x[inds] / _n_array[inds] ), ( y[inds] / _n_array[inds] )).correlation]
        spr4 += [stats.spearmanr(x, y).correlation]


    opt_res_df = pd.DataFrame(opt_res, columns=['pred_corr_zero', 'pred_corr_one' ,'pred_mu_x','pred_mu_y', 't_param', 'method'])

    zero_ratio = (pseudo_count == 0).mean()
    opt_res_df.loc[:,'zero_ratio'] = [(zero_ratio.iloc[i] + zero_ratio.iloc[i + 1]) / 2 for i in range(0, len(zero_ratio) - 1, 2)]

    opt_res_df.loc[:,'zr_cat'] = pd.cut(opt_res_df.zero_ratio, bins=4, labels = ['below_25',
                                                                            '25_50',
                                                                            '50_75',
                                                                             '75_100'
                                                                            ])

    opt_res_df = opt_res_df[['pred_corr_zero', 'pred_corr_one', 'pred_mu_x','pred_mu_y','method','zr_cat']].copy()

    opt_res_df = pd.concat([grid_points_df, opt_res_df], axis = 1)
    opt_res_df['spearman'] = spr
    opt_res_df['pearson'] = spr2
    opt_res_df['spearman_log_norm'] = spr4
    opt_res_df['pearson_norm'] = spr3
    opt_res_df = opt_res_df.loc[opt_res_df.method == 'copula'].copy()
    opt_res_df = opt_res_df.drop('method', axis=1)


    opt_res_df['copula_diff'] = opt_res_df.rho_zero - opt_res_df.pred_corr_zero
    opt_res_df['copula_one_diff'] = opt_res_df.rho_zero - opt_res_df.pred_corr_one
    opt_res_df['spearman_diff'] = opt_res_df.rho_zero - opt_res_df.spearman
    opt_res_df['pearson_diff'] = opt_res_df.rho_zero - opt_res_df.pearson
    opt_res_df['spearman_log_norm_diff'] = opt_res_df.rho_zero - opt_res_df.spearman_log_norm
    opt_res_df['pearson_norm_diff'] = opt_res_df.rho_zero - opt_res_df.pearson_norm

    #opt_res_df= opt_res_df.fillna(0)


    opt_res_df.loc[:,'orig_index'] = opt_res_df.index

    res_df_melted = pd.melt(opt_res_df,
            id_vars = ['mu_x','mu_y', 'zr_cat', 'sparse_frac','rho_zero', 'rho_one','orig_index'],
            value_vars=['copula_diff', 'spearman_diff','pearson_diff','spearman_log_norm_diff','pearson_norm_diff'],
            var_name = 'method', value_name = 'difference')



    opt_res_df['method_diff'] = abs(opt_res_df.copula_diff) - abs(opt_res_df.spearman_diff)

    return (res_df_melted, opt_res_df)


import matplotlib.pyplot as plt
import seaborn as sns
def show_pattern(pos_w, sim_edge_list_df, pseudo_count, i):
    tmp_source = pos_w.iloc[sim_edge_list_df.source,:].copy()
    tmp_source.loc[:,'gene'] = pseudo_count.loc[:, 'L' + str(i)].values
    tmp_end = pos_w.iloc[sim_edge_list_df.target,:].copy()
    tmp_end.loc[:,'gene'] = pseudo_count.loc[:, 'R' + str(i)].values
    tmp_dist = pd.concat([tmp_source, tmp_end])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pos_w.x, pos_w.y,s=0.01,alpha=1,color='grey');
    tmp_dist.loc[:, 'gene_log'] = np.log( tmp_dist.gene + 1 )
    sns.scatterplot(data= tmp_dist, x="x", y="y", hue="gene", palette="Reds",  s=4, linewidth=0, alpha=1, ax=ax)

    norm = plt.Normalize(tmp_dist['gene'].min(), tmp_dist['gene'].max())
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    cbar = ax.figure.colorbar(sm)
    cbar.outline.set_linewidth(0)
    sns.despine()
    plt.show()