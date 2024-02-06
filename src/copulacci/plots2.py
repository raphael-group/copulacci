# pylint: disable=C0103, C0114, C0301, W0108
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from adjustText import adjust_text
import seaborn as sns
from scipy import stats
import os

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
