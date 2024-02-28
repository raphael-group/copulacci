# pylint: disable=C0103, C0114, C0301, W0108
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from adjustText import adjust_text
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import os
from copulacci import model2


def draw_pairwise_scatter(
    merged_data_dict,
    gpair,
    score_pair,
    ntop = 5,
    s = 3,
    big_s = 20,
    bimod_filter = False,
    label = True,
    width = 4,
    height = 4,
    only_pos = False,
    figure_parent = '.',
    file_name = None,
    center_plot = False,
    take_diff = False,
    take_sim = False
):
    """
    TODO : Add docstring
    """
    if isinstance(merged_data_dict, pd.DataFrame):
        res = merged_data_dict.copy()
    # if type(merged_data_dict) is pd.DataFrame:
    #     res = merged_data_dict.copy()
    else:
        res = merged_data_dict[gpair].copy()
    if bimod_filter:
        res = res.loc[res.gmm_modality == 1].copy()


    fig,ax=plt.subplots(1,len(score_pair),
                        figsize=(width*len(score_pair),height)
            )
    for i,(x_col, y_col) in enumerate(score_pair):
        xmax = max(res[x_col])
        xmin = min(res[y_col])
        ymax = max(res[x_col])
        ymin = min(res[y_col])
        gmin = min(xmin, ymin)
        gmax = max(xmax, ymax)

        if not label:
           sns.scatterplot(res, x=x_col, y=y_col,s = s, linewidth = 0,ax = ax[i])
        else:
            if only_pos:
                res = res.loc[(res[x_col] > 0) & (res[y_col] > 0)]
            if take_diff or take_sim:
                if take_diff:
                    res['diff1'] = res[x_col] - res[y_col]
                    res['diff2'] = res[y_col] - res[x_col]
                    sig1 = res.sort_values('diff1', ascending=False)[:ntop]
                    sig2 = res.sort_values('diff2', ascending=False)[:ntop]
                else:
                    res['Rank1'] = res[x_col].abs().rank(method='dense', ascending=False)
                    res['Rank2'] = res[y_col].abs().rank(method='dense', ascending=False)
                    intersection_dict = {}
                    len_dict = {}
                    for ind in range(res.shape[0]):
                        lr_intersect = set(res.sort_values('Rank1').index[:ind]).intersection(
                            set(res.sort_values('Rank2').index[:ind])
                        )
                        intersection_dict[ind] = lr_intersect
                        len_dict[len(lr_intersect)] = ind
                    top_index_key = []
                    if ntop in len_dict:
                        top_index_key = len_dict[ntop]
                    else:
                        for k in len_dict:
                            if k > ntop:
                                top_index_key = len_dict[k]
                                break
                    sig1 = res.loc[list(intersection_dict[top_index_key])].loc[res[x_col] > 0].copy()
                    sig2 = res.loc[list(intersection_dict[top_index_key])].loc[res[x_col] < 0].copy()
            else:
                sig1 = res.sort_values(by=x_col,
                                    key=lambda x: abs(x),
                                    ascending=False)[:ntop]
                sig2 = res.sort_values(by=y_col,
                                    key=lambda x: abs(x),
                                    ascending=False)[:ntop]

            sig12 = sig1.join(sig2, rsuffix='_2',how='inner')
            sns.scatterplot(res, x=x_col, y=y_col,s = s, linewidth = 0,ax = ax[i])
            sns.scatterplot(
                            data = sig1,
                            x = x_col,
                            y = y_col,
                            s = big_s, c='r', linewidth = 0,ax = ax[i])
            sns.scatterplot(
                            data = sig2,
                            x = x_col,
                            y = y_col,
                            s = big_s, c='r', linewidth = 0,ax = ax[i])
            text_sig = []
            for j,r in sig1.drop(sig12.index).iterrows():
                text_sig.append(ax[i].text(x=r[x_col], y = r[y_col],
                                            s = j,
                                            color=(1, 0, 0),
                                            fontsize = 8
                                        ))
            for j,r in sig2.drop(sig12.index).iterrows():
                text_sig.append(ax[i].text(x=r[x_col], y = r[y_col],
                                            s = j,
                                            color=(0, 0, 1),
                                            fontsize = 8
                                        ))
            if(len(text_sig) > 0):
                adjust_text(text_sig,
                            arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
                            ax=ax[i])
            if (len(sig12)):
                sns.scatterplot(
                    data = sig12,
                    x = x_col,
                    y = y_col,
                    s = 10,
                    c='black',
                    linewidth = 2,
                    ax = ax[i]
                )
                text_sig12 = []
                for j,r in sig12.iterrows():
                    text_sig12.append(
                        ax[i].text(
                            x=r[x_col],
                            y = r[y_col],
                            s = j,
                            color = (0.5, 0, 0.5),
                            fontsize = 10,
                            weight='bold'
                        )
                    )
                if len(text_sig12) > 0:
                    adjust_text(
                        text_sig12,
                        arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
                        ax=ax[i]
                    )
        if not only_pos:
            ax[i].axhline(0, color='grey', linestyle='--')
            ax[i].axvline(0, color='grey', linestyle='--')
        ax[i].set_title(f'Spearman = { stats.spearmanr(res[x_col].values, res[y_col].values)[0] :.2f}')
        if center_plot:
            ax[i].set_xlim(gmin, gmax)
            ax[i].set_ylim(gmin, gmax)
    plt.suptitle('Interaction '+gpair.replace('=',' → '))
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(os.path.join(figure_parent, file_name), format='pdf', dpi=500)
    plt.show()


def draw_pairwise_difference_scatter(
    merged_data_dict,
    gpair,
    score_pair,
    ntop = 5,
    s = 3,
    big_s = 20,
    bimod_filter = False,
    label = True,
    width = 4,
    height = 4,
    only_pos = False,
    figure_parent = '.',
    file_name = None,
    center_plot = False
):
    """
    TODO : Add docstring
    """
    if isinstance(merged_data_dict, pd.DataFrame):
        res = merged_data_dict.copy()
    # if type(merged_data_dict) is pd.DataFrame:
    #     res = merged_data_dict.copy()
    else:
        res = merged_data_dict[gpair].copy()
    if bimod_filter:
        res = res.loc[res.gmm_modality == 1].copy()


    fig,ax=plt.subplots(1,len(score_pair),
                        figsize=(width*len(score_pair),height)
            )
    for i,(x_col, y_col) in enumerate(score_pair):
        xmax = max(res[x_col])
        xmin = min(res[y_col])
        ymax = max(res[x_col])
        ymin = min(res[y_col])
        gmin = min(xmin, ymin)
        gmax = max(xmax, ymax)

        if not label:
           sns.scatterplot(res, x=x_col, y=y_col,s = s, linewidth = 0,ax = ax[i])
        else:
            if only_pos:
                res = res.loc[(res[x_col] > 0) & (res[y_col] > 0)]
            res['diff'] = np.abs(res[x_col] - res[y_col])
            sig = res.sort_values('diff', ascending=False)[:ntop]

            sns.scatterplot(res, x=x_col, y=y_col,s = s, linewidth = 0,ax = ax[i])

            text_sig = []
            for j,r in sig.iterrows():
                text_sig.append(ax[i].text(x=r[x_col], y = r[y_col],
                                        s = j,
                                        color=(1, 0, 0),
                                        fontsize = 8
                                        #fontweight='bold'
                                    ))
            if(len(text_sig) > 0):
                adjust_text(text_sig,
                            arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
                            ax=ax[i])

        if not only_pos:
            ax[i].axhline(0, color='grey', linestyle='--')
            ax[i].axvline(0, color='grey', linestyle='--')
        ax[i].set_title(f'Spearman = { stats.spearmanr(res[x_col].values, res[y_col].values)[0] :.2f}')
        if center_plot:
            ax[i].set_xlim(-gmax, gmax)
            ax[i].set_ylim(-gmax, gmax)
        x_ax = np.linspace(-gmax, gmax, 100)
        ax[i].plot(x_ax, x_ax+0.2, color='green', linestyle='--')
        ax[i].plot(x_ax, x_ax-0.2, color='green', linestyle='--')
    plt.suptitle('Interaction '+gpair.replace('=',' → '))
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(os.path.join(figure_parent, file_name), format='pdf', dpi=500)
    plt.show()


def get_loc_df(adata):
    loc = adata.obsm['spatial']

    loc_df = pd.DataFrame(loc).rename(columns = {0:"x", 1:"y"})
    loc_df.loc[:,"cell"] = adata.obs_names
    loc_df.set_index('cell', inplace=True)
    return loc_df


def get_data(
    lig_rec_pair,
    gpair,
    other_index,
    close_contact_index,
    data_list_dict,
    data_list_dict_close,
    umi_sums,
    umi_sums_close,
    cont_type='other'
):
    if cont_type == 'other':
        o = np.array(other_index)
        index = np.where(o == lig_rec_pair)[0][0]
        x = data_list_dict[gpair][index][0]
        y = data_list_dict[gpair][index][1]
        us1 = umi_sums[gpair][gpair.split('=')[0]]
        us2 = umi_sums[gpair][gpair.split('=')[1]]
    else:
        o = np.array(close_contact_index)
        index = np.where(o == lig_rec_pair)[0][0]
        x = data_list_dict_close[gpair][index][0]
        y = data_list_dict_close[gpair][index][1]
        us1 = umi_sums_close[gpair][gpair.split('=')[0]]
        us2 = umi_sums_close[gpair][gpair.split('=')[1]]
    return (x, y, us1, us2)


def plot_edges(
    gpair,
    int_edges_new,
    loc_df,
    figsize=(10,10)
):
    lr_pairs_ct = int_edges_new.loc[
            int_edges_new.interaction == gpair,
            :
    ].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.scatter(loc_df['x'], loc_df['y'], c= "grey", s=0.4,alpha = 0.4)

    selected_cells_1 = list(set(lr_pairs_ct.cell1).intersection(set(loc_df.index)))
    selected_cells_2 = list(set(lr_pairs_ct.cell2).intersection(set(loc_df.index)))
    tmp = loc_df.loc[selected_cells_1 + selected_cells_2,:].copy()
    sns.scatterplot(x='x', y='y', s=10, data=tmp,edgecolor='black',alpha=1,ax= ax)

    for ind,row in lr_pairs_ct.iterrows():
        x1, y1 = loc_df.loc[  row.cell1, 'x' ], loc_df.loc[  row.cell1, 'y' ]
        x2, y2 = loc_df.loc[  row.cell2, 'x' ], loc_df.loc[  row.cell2, 'y' ]
        ax.plot([x1, x2], [y1, y2], color='black',
                  linestyle='-', markersize=0.5,linewidth=0.3)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([]);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


from matplotlib.colors import Normalize
def plot_norm_lr_boundary_in_same_plot(
    gpair,
    lr_index,
    loc_df,
    int_edges_new,
    gene1,
    gene2,
    count_df,
    merged_res,
    shrink_fraction=1.0,
    file_name = None,
    figure_parent = '.',
    figsize = (7,5),
    markersize = 10,
    cont_type = 'other',
    copula_params=model2.CopulaParams()
):
    int_type = gpair
    lr_pairs_ct = int_edges_new.loc[
            int_edges_new.interaction == int_type,
            :
    ].copy()

    tmp = loc_df
    tmp = tmp.loc[
        list(set(lr_pairs_ct.cell1).union(lr_pairs_ct.cell2).intersection(tmp.index)),
        :
    ]

    x, y, us1, us2 = get_data(lr_index, gpair, cont_type=cont_type)
    copula_params = copula_params._replace(return_sum=False)
    sx = np.log(x.sum() / us1.sum())
    sy = np.log(y.sum() / us2.sum())
    res = merged_res[gpair].copy()
    coeff = res.loc[lr_index].copula_coeff
    loglikvec = model2.calculate_mahalanobis_distance(
        [coeff, sx, sy],
        x,
        y,
        us1,
        us2,
        copula_params
    )
    lr_pairs_ct['copula_score'] = loglikvec

    selected_cells_1 = list(set(lr_pairs_ct.cell1).intersection(set(tmp.index)))
    selected_cells_2 = list(set(lr_pairs_ct.cell2).intersection(set(tmp.index)))

    max_gene1 = count_df.loc[selected_cells_1, gene1].values.max()
    max_gene2 = count_df.loc[selected_cells_2, gene2].values.max()
    global_max = max(max_gene1, max_gene2)
    global_min = min(
        count_df.loc[selected_cells_1, gene1].values.min(),
        count_df.loc[selected_cells_2, gene2].values.min()
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.scatter(loc_df['x'], loc_df['y'], c= "grey", s=0.4,alpha = 0.4)
    colors = np.array(count_df.loc[selected_cells_1, gene1].values)
    tmp = loc_df.loc[selected_cells_1,:].copy()
    tmp.loc[:, 'gene'] = colors
    sns.scatterplot(x='x', y='y', hue='gene',
                     palette='Reds',s=markersize, data=tmp,edgecolor='black',alpha=1,ax= ax)
    norm = Normalize(tmp['gene'].min(), tmp['gene'].max())
    #norm = Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    cax = ax.figure.colorbar(sm,ax=ax,shrink=shrink_fraction,pad=-0.1)
    cax.set_label(gene1, color='red', rotation=270, labelpad=15)

    colors = np.array(count_df.loc[selected_cells_2, gene2].values)
    tmp = loc_df.loc[selected_cells_2,:].copy()
    tmp.loc[:, 'gene'] = colors
    sns.scatterplot(x='x', y='y', hue='gene',marker="^",
                     palette='Blues',s=markersize, data=tmp,edgecolor='black',alpha=1,ax= ax)
    norm = Normalize(tmp['gene'].min(), tmp['gene'].max())
    #norm = Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    ax.set_title(f"{lr_index} - copula: {coeff:.2f} \n {gpair}")
    #ax.set_title(gene1 + " | " + gene2 + "\n" + gpair)
    cax = ax.figure.colorbar(sm,ax=ax,shrink=shrink_fraction, pad=-0.1)
    cax.set_label(gene2, color='blue', rotation=270, labelpad=15)


    # draw edges
    norm = Normalize(vmin=min(lr_pairs_ct.copula_score),
                     vmax=np.quantile(lr_pairs_ct.copula_score, 0.95))
    cmap = plt.get_cmap('Greys')
    for ind,row in lr_pairs_ct.iterrows():
        x1, y1 = loc_df.loc[  row.cell1, 'x' ], loc_df.loc[  row.cell1, 'y' ]
        x2, y2 = loc_df.loc[  row.cell2, 'x' ], loc_df.loc[  row.cell2, 'y' ]
        dx = x2 - x1
        dy = y2 - y1
        color = cmap(norm(row.copula_score))
        # ax.plot([x1, x2], [y1, y2], color='black',
        #         linestyle='-', markersize=0.5,linewidth=0.3)
        ax.quiver(np.array([x1]), np.array([y1]),
                  np.array([dx]), np.array([dy]),
                  angles='xy', scale_units='xy',
                  color=color,
                  scale=1,
                  headwidth=10,
                  headaxislength=5,
                  headlength=10,
                  width=0.001,
                 )
        # ax.arrow(x1, y1, dx, dy, head_width=0.05,
        #          head_length=0.1, fc='k', ec='k')

    norm = Normalize(vmin=min(lr_pairs_ct.copula_score),
                     vmax=np.quantile(lr_pairs_ct.copula_score,0.95))
    #norm = Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=norm)
    sm.set_array([])
    cax = ax.figure.colorbar(sm,ax=ax,shrink=shrink_fraction,pad=0)
    cax.set_label('mhl', color='blue', rotation=270, labelpad=15)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([]);
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(os.path.join(figure_parent, file_name), format='pdf', dpi=500)
    plt.show()


def plot_liklihood_diagnostic(
    lig_rec_pair,
    gpair,
    other_index,
    close_contact_index,
    data_list_dict,
    data_list_dict_close,
    umi_sums,
    umi_sums_close,
    x_train = np.linspace(-0.99, 0.99, 1000),
    ymin = -1000,
    ymax = 50000,
    cont_type='other',
    copula_params=model2.CopulaParams(),
    opt_params = model2.OptParams()
):
    x, y, us1, us2 = get_data(
        lig_rec_pair,
        gpair,
        other_index,
        close_contact_index,
        data_list_dict,
        data_list_dict_close,
        umi_sums,
        umi_sums_close,
        cont_type=cont_type
    )
    sx = np.log(x.sum() / us1.sum())
    sy = np.log(y.sum() / us2.sum())
    likf = model2.only_log_lik(sx,sy,us1,us2,x,y,copula_params,x_train)
    peaks, _ = find_peaks(-likf)
    if len(peaks) > 1:
        local_min_index = peaks[np.argmax(likf[peaks])]
        local_min = x_train[peaks[np.argmax(likf[peaks])]]
    else:
        local_min_index = peaks[0]
        local_min = x_train[peaks[0]]


    _, d2lik, d1lik = model2.diff_using_num(
        sx,
        sy,
        us1,
        us2,
        x,
        y,
        copula_params,
        opt_params,
        x_train=x_train,
        do_first_order=True
    )

    plt.scatter(x_train, likf, s=3, alpha=0.6, linewidth=0, c='b',label = r'$L$')
    plt.scatter(x_train, d1lik, s=3, alpha=0.6, linewidth=0, c='g',label = r'$\frac{\partial L}{\partial \rho}$' )
    plt.scatter(x_train, d2lik, s=3, alpha=0.6, linewidth=0, c='r',label = r'$\frac{\partial^2 L}{\partial \rho^2}$' )
    plt.axhline(0, color='red', linestyle='--');
    plt.axvline(local_min, color='red', linestyle='--');
    legend = plt.legend()
    for handle in legend.legend_handles:
        handle.set_sizes([30])  #
        handle.set_alpha([1])

    plt.ylim(ymin, ymax);
    #plt.xlim(-0.25, 0.25);
    plt.show()
    return (likf, d2lik, d1lik, local_min_index)
