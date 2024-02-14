# import os
import random
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scanpy as sc
import spatialdm as sdm
from sklearn.preprocessing import normalize
from spatialdm.diff_utils import *
from scipy.optimize import minimize
import networkx as nx
import statsmodels.api as sm

from copulacci import spatial
import warnings


# os.environ['USE_PYGEOS'] = '0'
# import geopandas as gpd
# gpd.options.use_pygeos = True

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EPSILON = 1.1920929e-07


def get_dt_cdf(x, lam, DT=True):
    u_x = stats.poisson.cdf(x, lam).clip(EPSILON, 1 - EPSILON)
    u_x2 = stats.poisson.cdf(x-1, lam).clip(EPSILON, 1 - EPSILON)
    if DT:
        v = stats.uniform.rvs(size=len(x))
        r = u_x * v + u_x2 * (1 - v)
    else:
        r = (u_x + u_x2) / 2.0
    #r = (u_x + u_x2) / 2.0
    #r = u_x2
    # remove too low / too big
    idx_adjust = np.where(1 - r < EPSILON)[0]
    r[ idx_adjust ] = r[ idx_adjust ] - EPSILON
    idx_adjust = np.where(r < EPSILON)[0]
    r[ idx_adjust ] = r[ idx_adjust ] + EPSILON

    return stats.norm.ppf(r)


def sample_from_copula(
    n_array,
    mu_x,
    mu_y,
    coeff
):
    cov = linalg.toeplitz([1,coeff])
    lam_x = n_array * np.exp(mu_x)
    lam_y = n_array * np.exp(mu_y)
    samples = np.random.multivariate_normal(
        [0,0],
        cov,
        size = len(n_array)
    )
    samples = pd.DataFrame(samples, columns=['x', 'y'])
    cdf_x = stats.norm.cdf(samples.x)
    cdf_y = stats.norm.cdf(samples.y)
    samples.loc[:, 'x'] = stats.poisson.ppf(cdf_x, lam_x)
    samples.loc[:, 'y'] = stats.poisson.ppf(cdf_y, lam_y)

    return samples


def sample_from_copula_dist(
    n_array,
    mu_x,
    mu_y,
    coeff_list
):
    lam_x = n_array * np.exp(mu_x)
    lam_y = n_array * np.exp(mu_y)

    # samples = Parallel(n_jobs=n_jobs, verbose=0)(
    #                 delayed(np.random.multivariate_normal)(
    #                     [0,0],
    #                     [[1, coeff], [coeff, 1]],
    #                     size = 1) for coeff in coeff_list)
    # samples = np.array(list(chain(*(samples))))

    samples = np.array([np.random.multivariate_normal(
            [0,0],
            [[1,coeff],[coeff,1]],
            size = 1
        ).squeeze() for coeff in coeff_list])

    samples = pd.DataFrame(samples, columns=['x', 'y'])
    cdf_x = stats.norm.cdf(samples.x)
    cdf_y = stats.norm.cdf(samples.y)
    samples.loc[:, 'x'] = stats.poisson.ppf(cdf_x, lam_x)
    samples.loc[:, 'y'] = stats.poisson.ppf(cdf_y, lam_y)

    return samples


def sample_from_copula_grad(
    n_array,
    mu_x,
    mu_y,
    coeff_list,
    dist,
    l = None,
    t = 0
):
    if l is None:
        l = max(dist)
    lam_x = (n_array * np.exp(mu_x)) * np.exp(-(t * (dist/l**2)))
    lam_y = (n_array * np.exp(mu_y)) * np.exp(-(t * (dist/l**2)))

    # samples = Parallel(n_jobs=n_jobs, verbose=0)(
    #                 delayed(np.random.multivariate_normal)(
    #                     [0,0],
    #                     [[1, coeff], [coeff, 1]],
    #                     size = 1) for coeff in coeff_list)
    # samples = np.array(list(chain(*(samples))))

    samples = np.array([np.random.multivariate_normal(
            [0,0],
            [[1,coeff],[coeff,1]],
            size = 1
        ).squeeze() for coeff in coeff_list])

    samples = pd.DataFrame(samples, columns=['x', 'y'])
    cdf_x = stats.norm.cdf(samples.x)
    cdf_y = stats.norm.cdf(samples.y)
    samples.loc[:, 'x'] = stats.poisson.ppf(cdf_x, lam_x)
    samples.loc[:, 'y'] = stats.poisson.ppf(cdf_y, lam_y)

    return samples


def scdesign_copula(x, y, DT=False):
    lam1 = x.mean()
    lam2 = y.mean()
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    z = np.column_stack([r_x, r_y])
    return np.corrcoef(z.T)[0, 1]


def call_pois_single_param_opt(x, y,
                               reg=False,
                               alpha=1.0,
                               DT=False,
                               cutoff=0.6,
                               length_cutoff=20):
    if ( (x.sum() == 0) or (y.sum() == 0) ):
            return ([ 0, x.mean(), y.mean(), 'skip' ])
    if len(x) < length_cutoff:
        return ([ 0, x.mean(), y.mean(), 'skip' ])
    if ( ((x==0).sum() / len(x) > cutoff) | ((y==0).sum() / len(y) > cutoff) ):
        return ([ 0, x.mean(), y.mean(), 'skip' ])

    lam1 = x.mean()
    lam2 = y.mean()
    def copula_coeff_lik(params, _x, _y, reg=False, DT=False, alpha=1.0, cutoff = 0.6, length_cutoff=20):
        x = _x.copy()
        y = _y.copy()

        coeff = params[0]
        # get z
        r_x = get_dt_cdf(x, lam1, DT=DT)
        r_y = get_dt_cdf(y, lam2, DT=DT)
        z = np.column_stack([r_x, r_y])

        # get gaussuan part using z
        cov =linalg.toeplitz(np.array([1,coeff], dtype = 'float'))
        #inv_cov = np.linalg.inv(cov)
        det = 1 - coeff**2
        inv_cov = (1. / det)*linalg.toeplitz(np.array([1,-coeff], dtype = 'float'))
        norm_term = 1 / (2 * np.pi * np.sqrt(det))
        exp_term = -0.5 * np.sum(z @ (inv_cov - np.identity(2)) * z, axis=1)
        lh_vec = np.log(norm_term.clip(EPSILON, 1 - EPSILON) ) + exp_term
        logsum = np.sum(lh_vec)
        if reg:
            #logsum += alpha * np.abs(coeff)
            logsum -= alpha * (coeff**2)
        return -logsum
    res = minimize(
        copula_coeff_lik,
        x0=[0.0],
        method = 'Nelder-Mead',
        bounds = [(-0.9, 0.9)],
        args=(x,y,reg,DT,alpha,cutoff,length_cutoff),
        tol=1e-6
    )
    return ([res.x[0], x.mean(), y.mean(), 'copula'])


def call_pois_multi_param_opt(x,y,DT=False):
    def copula_coeff_lik(params, x,y,DT=False):
        coeff = params[0]
        lam1 = params[1]
        lam2 = params[2]
        # get z
        r_x = get_dt_cdf(x, lam1, DT=DT)
        r_y = get_dt_cdf(y, lam2, DT=DT)
        z = np.column_stack([r_x, r_y])

        # get gaussuan part using z
        cov =linalg.toeplitz(np.array([1,coeff], dtype = 'float'))
        #inv_cov = np.linalg.inv(cov)
        det = 1 - coeff**2
        inv_cov = (1. / det)*linalg.toeplitz(np.array([1,-coeff], dtype = 'float'))
        norm_term = 1 / (2 * np.pi * np.sqrt(det))
        exp_term = -0.5 * np.sum(z @ (inv_cov - np.identity(2)) * z, axis=1)
        lh_vec = np.log(norm_term.clip(EPSILON, 1 - EPSILON) ) + exp_term
        logsum = np.sum(lh_vec)
        # get the pdf for marginals
        logsum += np.sum(np.log(stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) ))
        logsum += np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))

        return -1 * logsum

    lam_x_start = x.mean()
    lam_y_start = y.mean()
    res = minimize(
        copula_coeff_lik,
        x0=[0.0, lam_x_start, lam_y_start],
        method = 'Nelder-Mead',
        bounds = [(-0.98, 0.98), (0, np.max(x)), (0, np.max(y))],
        args=(x,y,DT),
        tol=1e-6
    )
    return res.x


def log_joint_lik_perm_old(params, umi_sum_1, umi_sum_2, x, y, perm=100, DT=True, model = 'copula'):
    # get lam parameters for mu_1
    coeff = params[0]
    mu_1 = params[1]
    mu_2 = params[2]

    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1)
            r_y += get_dt_cdf(y, lam2)
        z = np.column_stack([r_x/perm, r_y/perm])
    else:
        z = np.column_stack([r_x, r_y])

    # get gaussuan part using z
    if model == 'gaussian':
        cov =linalg.toeplitz(np.array([1.0, coeff], dtype = 'float'))
        # Calculate the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov)
        det_cov_matrix = np.linalg.det(cov)
        n = z.shape[0]
        log_likelihood = -0.5 * (n * np.log(2 * np.pi) + n * np.log(det_cov_matrix))
        mahalanobis_distances = np.sum(z.dot(inv_cov_matrix) * z, axis=1)
        log_likelihood -= 0.5 * np.sum(mahalanobis_distances)
        return -log_likelihood

    cov =linalg.toeplitz(np.array([1,coeff], dtype = 'float'))
    #inv_cov = np.linalg.inv(cov)
    det = 1 - coeff**2
    # inv_cov = (1. / det)*linalg.toeplitz(np.array([1,-coeff], dtype = 'float'))
    # norm_term = 1 / (2 * np.pi * np.sqrt(det))
    # exp_term = -0.5 * np.sum(z @ (inv_cov - np.identity(2)) * z, axis=1)
    # lh_vec = np.log(norm_term.clip(EPSILON, 1 - EPSILON) ) + exp_term
    # logsum = np.sum(lh_vec)

    inv_cov = (1. / det)*linalg.toeplitz(np.array([1,-coeff], dtype = 'float'))
    # norm_term = -0.5 * (len(x) * np.log(2 * np.pi) + len(x) * np.log(det+EPSILON))
    norm_term = -0.5 * len(x) * np.log(det+EPSILON)
    exp_term = np.sum(-0.5 * np.sum(z @ (inv_cov - np.identity(2)) * z, axis=1))
    logsum = norm_term + exp_term

    # get the pdf for marginals
    logsum += np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) ))
    logsum += np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
    return -logsum


def log_joint_lik_perm(params, umi_sum_1, umi_sum_2, x, y, perm=100, DT=True, model = 'copula', return_sum = True):
    # get lam parameters for mu_1
    coeff = params[0]
    mu_1 = params[1]
    mu_2 = params[2]

    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1)
            r_y += get_dt_cdf(y, lam2)
        z = np.column_stack([r_x/perm, r_y/perm])
    else:
        z = np.column_stack([r_x, r_y])

    # term1
    det = 1 - coeff**2
    if return_sum:
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )

        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1 + term2 + term3

        return -logsum
    else:
        term1 = -0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1])
        term2 = (
            np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) ) +
            np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) )
        )
        term3 = -0.5 * np.log(det+EPSILON)
        return (term1 + term2 + term3)


def log_joint_lik_perm_dist(params, umi_sum_1, umi_sum_2, x, y, dist_list,
    perm=100, DT=True, model = 'copula', return_sum = True):
    # get lam parameters for mu_1
    rho_zero = params[0]
    rho_one = params[1]
    coeff_list = rho_zero * np.exp(-1 * dist_list * rho_one)
    mu_1 = params[2]
    mu_2 = params[3]

    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1)
            r_y += get_dt_cdf(y, lam2)
        z = np.column_stack([r_x/perm, r_y/perm])
    else:
        z = np.column_stack([r_x, r_y])

    # term1
    det = 1 - coeff_list**2
    if return_sum:
        term1 = np.sum(-0.5 * (((coeff_list**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff_list/det) * z[:,0] * z[:,1]) )

        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )

        term3 = np.sum(-0.5 * np.log(det+EPSILON))
        logsum = term1 + term2 + term3

        return -logsum
    else:
        term1 = -0.5 * (((coeff_list**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff_list/det) * z[:,0] * z[:,1])
        term2 = (
            np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) ) +
            np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) )
        )
        term3 = -0.5 * np.log(det+EPSILON)
        return (term1 + term2 + term3)


def log_joint_lik_perm_grad_dist(params, umi_sum_1, umi_sum_2, x, y, dist_list,  grad_list, perm=100, DT=True, model = 'copula'):
    # get lam parameters for mu_1
    rho_zero = params[0]
    rho_one = params[1]
    coeff_list = rho_zero * np.exp(-1 * dist_list * rho_one)
    mu_1 = params[2]
    mu_2 = params[3]
    t = params[4]
    l = max(grad_list)

    lam1 = (umi_sum_1 * np.exp(mu_1)) * np.exp(-(t * (grad_list / l**2)))
    lam2 = (umi_sum_2 * np.exp(mu_2)) * np.exp(-(t * (grad_list / l**2)))
    #lam1 = umi_sum_1 * np.exp(mu_1)
    #lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1)
            r_y += get_dt_cdf(y, lam2)
        z = np.column_stack([r_x/perm, r_y/perm])
    else:
        z = np.column_stack([r_x, r_y])

    # term1
    det = 1 - coeff_list**2
    term1 = np.sum(-0.5 * (((coeff_list**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff_list/det) * z[:,0] * z[:,1]) )

    term2 = (
        np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
        np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
    )

    term3 = np.sum(-0.5 * np.log(det+EPSILON))
    logsum = term1 + term2 + term3

    return -logsum


def call_optimizer_dense(
    _x,
    _y,
    _umi_sum_1,
    _umi_sum_2,
    method = 'Nelder-Mead',
    perm=10,DT = True,
    cutoff=0.6,
    length_cutoff=20,
    model='copula',
    num_restarts = 50,
    force = False,
    remove_zero = False,
    zero_pair_filter = False,
    zz_ratio_cutoff = 0.7,
    mu_cutoff_filter = False,
    mu_cutoff = -8
):
    if(remove_zero):
        nz_ind = np.where((_x != 0) | (_y != 0))
        _x = _x[nz_ind]
        _y = _y[nz_ind]
        _umi_sum_1 = _umi_sum_1[nz_ind]
        _umi_sum_2 = _umi_sum_2[nz_ind]

    if(zero_pair_filter):
        zz_percent = ((_x == 0) & (_y==0)).sum() / len(_x)
        if zz_percent >= zz_ratio_cutoff:
            return ([ 0, 0, 0 , 'skip_zz' ])

    x = _x.copy()
    y = _y.copy()
    if (np.isnan( stats.spearmanr(x / _umi_sum_1, y / _umi_sum_2).correlation )):
        return ([ 0, 0, 0 , 'skip' ])
    if ( (x.sum() == 0) or (y.sum() == 0) ):
        return ([ 0, 0, 0 , 'skip' ])
    if len(x) < length_cutoff:
        return ([ 0, 0, 0 , 'skip' ])
    umi_sum_1 = _umi_sum_1.copy()
    umi_sum_2 = _umi_sum_2.copy()
    method_type = model

    if not force:
        # If either of the arrays are too sparse we would run
        # gaussian extimate rather than anything else
        inds_zero = np.where((x == 0) & (y == 0))[0]
        if ( ((x==0).sum() / len(x) > cutoff) | ((y==0).sum() / len(y) > cutoff) ):
            method_type = 'skip'


    results = []
    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())

    if mu_cutoff_filter:
        if (mu_x_start < mu_cutoff) | (mu_y_start < mu_cutoff):
            return ([ 0, 0, 0 , 'skip_mu' ])

    if method_type == 'skip':
        return ([ 0, 0, 0 , 'skip' ])
    else:
        for _ in range(num_restarts):
            res = minimize(
                log_joint_lik_perm,
                x0=[0.0,mu_x_start,mu_y_start],
                method = method,
                bounds = [(-0.99, 0.99),
                          (mu_x_start-5, 0),
                          (mu_y_start-5, 0)],
                args=(umi_sum_1,umi_sum_2,x,y,perm,DT,"copula"),
                tol=1e-6
            )
            results += [res.copy()]
        best_result = min(results, key=lambda x: x['fun'])
        return list(best_result['x']) + ['copula']


def call_optimizer_dense_dist(
    _x,
    _y,
    _umi_sum_1,
    _umi_sum_2,
    dist_list,
    rho_one_start = 0.01,
    method = 'Nelder-Mead',
    perm=10,
    DT = True,
    cutoff=0.6,
    length_cutoff=20,
    model='copula',
    num_restarts = 50,
    force = False,
    zero_pair_filter = False,
    zz_ratio_cutoff = 0.7,
    mu_cutoff_filter = False,
    mu_cutoff = -8
):

    x = _x.copy()
    y = _y.copy()
    if(zero_pair_filter):
        zz_percent = ((_x == 0) & (_y==0)).sum() / len(_x)
        if zz_percent >= zz_ratio_cutoff:
            return ([ 0, 0, 0 , 0, 'skip_zz' ])

    if (np.isnan( stats.spearmanr(x / _umi_sum_1, y / _umi_sum_2).correlation )):

        return ([ 0, 0, 0, 0 , 'skip' ])
    if ( (x.sum() == 0) or (y.sum() == 0) ):

        return ([ 0, 0, 0, 0 , 'skip' ])
    if len(x) < length_cutoff:

        return ([ 0, 0, 0, 0 , 'skip' ])
    umi_sum_1 = _umi_sum_1.copy()
    umi_sum_2 = _umi_sum_2.copy()
    method_type = model

    if not force:
        # If either of the arrays are too sparse we would run
        # gaussian extimate rather than anything else

        if ( ((x==0).sum() / len(x) > cutoff) | ((y==0).sum() / len(y) > cutoff) ):
            method_type = 'skip'


    results = []
    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())
    if mu_cutoff_filter:
        if (mu_x_start < mu_cutoff) | (mu_y_start < mu_cutoff):
            return ([ 0, 0, 0 , 0, 'skip_mu' ])

    if method_type == 'skip':
        return ([ 0, 0, 0, 0 , 'skip' ])
    else:
        for _ in range(num_restarts):
            res = minimize(
                log_joint_lik_perm_dist,
                x0=[0.0,rho_one_start,mu_x_start,mu_y_start],
                method = method,
                bounds = [(-0.99, 0.99),
                          (0.0, 0.1),
                          (mu_x_start-5, 0),
                          (mu_y_start-5, 0)],
                args=(umi_sum_1,umi_sum_2,x,y, dist_list,perm,DT,"copula"),
                tol=1e-6
            )
            results += [res.copy()]
        best_result = min(results, key=lambda x: x['fun'])
        return list(best_result['x']) + ['copula']


def call_optimizer_dense_grad(
    _x,
    _y,
    _umi_sum_1,
    _umi_sum_2,
    dist_list,
    grad_list,
    rho_one_start = 0.01,
    method = 'Nelder-Mead',
    perm=10,
    DT = True,
    cutoff=0.6,
    length_cutoff=20,
    model='copula',
    num_restarts = 50,
    force = False
):
    x = _x.copy()
    y = _y.copy()
    if (np.isnan( stats.spearmanr(x / _umi_sum_1, y / _umi_sum_2).correlation )):

        return ([ 0, 0, 0, 0 , 'skip' ])
    if ( (x.sum() == 0) or (y.sum() == 0) ):

        return ([ 0, 0, 0, 0 , 'skip' ])
    if len(x) < length_cutoff:

        return ([ 0, 0, 0, 0 , 'skip' ])
    umi_sum_1 = _umi_sum_1.copy()
    umi_sum_2 = _umi_sum_2.copy()
    method_type = model

    if not force:
        # If either of the arrays are too sparse we would run
        # gaussian extimate rather than anything else

        if ( ((x==0).sum() / len(x) > cutoff) | ((y==0).sum() / len(y) > cutoff) ):

            method_type = 'skip'


    results = []
    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())
    if method_type == 'skip':
        return ([ 0, 0, 0, 0 , 'skip' ])
    else:
        for _ in range(num_restarts):
            res = minimize(
                log_joint_lik_perm_grad_dist,
                x0=[0.0,rho_one_start,mu_x_start,mu_y_start,1],
                method = method,
                bounds = [(-0.99, 0.99),
                          (0.0, 0.1),
                          (mu_x_start-5, 0),
                          (mu_y_start-5, 0),
                          (0,50)
                         ],
                args=(umi_sum_1,umi_sum_2,x,y, dist_list,grad_list,perm,DT,"copula"),
                tol=1e-6
            )
            results += [res.copy()]
        best_result = min(results, key=lambda x: x['fun'])
        return list(best_result['x']) + ['copula']


# Call copula
def call_optimizer_full(_x, _y,
                       _umi_sum_1,
                       _umi_sum_2, method = 'Nelder-Mead',perm=10,DT=True,cutoff=0.6,model='copula',
                       num_restarts = 50,force=False,quick=False):
    x = _x.copy()
    y = _y.copy()
    if ( (x.sum() == 0) or (y.sum() == 0) ):
        return ([ 0, 0, 0 , 'zero' ])
    umi_sum_1 = _umi_sum_1.copy()
    umi_sum_2 = _umi_sum_2.copy()
    method_type = model

    if not force:
        # If either of the arrays are too sparse we would run
        # gaussian extimate rather than anything else
        inds_zero = np.where((x == 0) & (y == 0))[0]
        if ( ((x==0).sum() / len(x) > cutoff) | ((y==0).sum() / len(y) > cutoff) ):
            method_type = 'gaussian'
        elif(len(inds_zero)/len(x) > cutoff):
            method_type = 'gaussian'
        elif( (x.max() == 1) and (y.max() == 1) ):
            method_type = 'gaussian'

    results = []
    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())
    if method_type == 'gaussian':
        if quick:
            r_x = get_dt_cdf(x, x.mean(), DT=DT)
            r_y = get_dt_cdf(y, y.mean(), DT=DT)
            z = np.column_stack([r_x, r_y])
            return ([np.corrcoef(z.T)[0][1],mu_x_start,mu_y_start,'gaussian'])
        else:
            for _ in range(num_restarts):
                res = minimize(
                    log_joint_lik_perm,
                    x0=[0.0,mu_x_start,mu_y_start],
                    method = method,
                    bounds = [(-0.99, 0.99),
                              (mu_x_start-5, 0),
                              (mu_y_start-5, 0)],
                    args=(umi_sum_1,umi_sum_2,x,y,1,True,"gaussian"),
                    tol=1e-6
                )
                results += [res.copy()]
            best_result = min(results, key=lambda x: x['fun'])
            return list(best_result['x']) + ['gaussian']
    else:
        for _ in range(num_restarts):
            res = minimize(
                log_joint_lik_perm,
                x0=[0.0,mu_x_start,mu_y_start],
                method = method,
                bounds = [(-0.99, 0.99),
                          (mu_x_start-5, 0),
                          (mu_y_start-5, 0)],
                args=(umi_sum_1,umi_sum_2,x,y,perm,DT,"copula"),
                tol=1e-6
            )
            results += [res.copy()]
        best_result = min(results, key=lambda x: x['fun'])
        return list(best_result['x']) + ['copula']


# Run optimizer with different threads
def run_poiss_copula(
    data_list_dict: dict,
    lig_rec_pair_list = None,
    n_jobs = 20,
    verbose = 1,
    method = 'Nelder-Mead',
    DT=False,
    cutoff=0.6,
    length_cutoff=20,
    heteronomic = False,
    df_lig_rec = None,
    num_restarts = 1
) -> pd.DataFrame:
    cop_df = pd.DataFrame()
    for g1, data_list in data_list_dict.items():
        g11, g12 = g1.split('=')
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(call_pois_single_param_opt)(
                    x,
                    y,
                    DT=DT,
                    cutoff=cutoff,
                    length_cutoff=length_cutoff
                ) for (x,y) in data_list)
        tmp = pd.DataFrame(res,columns=[g1,g11+'_mu_x',g12+'_mu_y',g1+'_copula_method'])
        cop_df = pd.concat([cop_df, tmp.copy()], axis = 1)
    if heteronomic:
        if df_lig_rec is None:
            raise ValueError('df_lig_rec is None')
        cop_df.index = df_lig_rec.index
    else:
        if lig_rec_pair_list is None:
            raise ValueError('lig_rec_pair_list is None')
        cop_df.index = [l+'_'+r for l,r in lig_rec_pair_list]
        cop_df.loc[:,'ligand'] = cop_df.index.str.split('_').str[0]
        cop_df.loc[:,'receptor'] = cop_df.index.str.split('_').str[1]
    return cop_df


# Run optimizer with different threads
def run_copula(
    data_list_dict: dict,
    umi_sums: dict,
    dist_list_dict: dict = None,
    lig_rec_pair_list = None,
    n_jobs = 20,
    verbose = 1,
    method = 'Nelder-Mead',
    perm = 10,
    DT=True,
    cutoff=0.6,
    length_cutoff=20,
    model='copula',
    num_restarts = 1,
    force=False,
    quick=False,
    type_run = 'full',
    heteronomic = False,
    df_lig_rec = None,
    groups = None,
    zero_pair_filter = False,
    zz_ratio_cutoff = 0.7
) -> dict:
    cop_df_dict = {}
    if groups == None:
        groups = data_list_dict.keys()
    if type_run == 'full':
        for g1 in groups:
            data_list = data_list_dict[g1]
            print(g1)
            g11, g12 = g1.split('=')
            res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(call_optimizer_full)(
                        x,
                        y,
                        umi_sums[g1][g11],
                        umi_sums[g1][g12],
                        method=method,
                        perm=perm,
                        DT=DT,
                        cutoff=cutoff,
                        model=model,
                        num_restarts = num_restarts,
                        quick=quick) for (x,y) in data_list)
            tmp = pd.DataFrame(res,columns=[
                'copula_coeff',
                'mu_x',
                'mu_y',
                'copula_method'
                ])
            cop_df_dict[g1] = tmp.copy()

    else:
        if dist_list_dict is None:
            for g1 in groups:
                data_list = data_list_dict[g1]
                print(g1)
                g11, g12 = g1.split('=')
                res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(call_optimizer_dense)(
                            x,
                            y,
                            umi_sums[g1][g11],
                            umi_sums[g1][g12],
                            method=method,
                            perm=perm,
                            DT=DT,
                            cutoff=cutoff,
                            length_cutoff=length_cutoff,
                            model=model,
                            num_restarts = num_restarts,
                            zero_pair_filter = zero_pair_filter,
                            zz_ratio_cutoff = zz_ratio_cutoff,
                         ) for (x,y) in data_list)
                #tmp = pd.DataFrame(res,columns=[g1,g11+'_mu_x',g12+'_mu_y',g1+'_copula_method'])
                tmp = pd.DataFrame(res,columns=[
                    'copula_coeff',
                    'mu_x',
                    'mu_y',
                    'copula_method'
                ])
                cop_df_dict[g1] = tmp.copy()
                if heteronomic:
                    if df_lig_rec is None:
                        raise ValueError('df_lig_rec is None')
                    cop_df_dict[g1].index = df_lig_rec.index
                else:
                    if lig_rec_pair_list is None:
                        raise ValueError('lig_rec_pair_list is None')
                    cop_df_dict[g1].index = [l+'_'+r for l,r in lig_rec_pair_list]
                    cop_df_dict[g1].loc[:,'ligand'] = cop_df_dict[g1].index.str.split('_').str[0]
                    cop_df_dict[g1].loc[:,'receptor'] = cop_df_dict[g1].index.str.split('_').str[1]
                # cop_df = pd.concat([cop_df, tmp.copy()], axis = 1)
        else:
            for g1 in groups:
                data_list = data_list_dict[g1]
                print(g1)
                g11, g12 = g1.split('=')
                res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                        delayed(call_optimizer_dense_dist)(
                            x,
                            y,
                            umi_sums[g1][g11],
                            umi_sums[g1][g12],
                            dist_list_dict[g1],
                            method=method,
                            perm=perm,
                            DT=DT,
                            cutoff=cutoff,
                            length_cutoff=length_cutoff,
                            model=model,
                            num_restarts = num_restarts
                            ) for (x,y) in data_list)
                # tmp = pd.DataFrame(res,columns=[g1+'_zero', g1+'_one', g11+'_mu_x',
                #                                 g12+'_mu_y', g1+'_copula_method'])
                tmp = pd.DataFrame(res,columns=[
                    'rho_zero',
                    'rho_one',
                    'mu_x',
                    'mu_y',
                    'copula_method'
                ])
                cop_df_dict[g1] = tmp.copy()
                # cop_df = pd.concat([cop_df, tmp.copy()], axis = 1)

                if heteronomic:
                    if df_lig_rec is None:
                        raise ValueError('df_lig_rec is None')
                    cop_df_dict[g1].index = df_lig_rec.index
                else:
                    if lig_rec_pair_list is None:
                        raise ValueError('lig_rec_pair_list is None')
                    cop_df_dict[g1].index = [l+'_'+r for l,r in lig_rec_pair_list]
                    cop_df_dict[g1].loc[:,'ligand'] = cop_df_dict[g1].index.str.split('_').str[0]
                    cop_df_dict[g1].loc[:,'receptor'] = cop_df_dict[g1].index.str.split('_').str[1]
    return cop_df_dict


def merge_data_groups(
    groups,
    data_list_dict: dict,
    umi_sums: dict,
    dist_list_dict: dict = None,
    merged_group_name = None,
):
    if merged_group_name is None:
        merged_group_name = 'group_merged'

    data_list_merged = []

    umi_sum_source = []
    umi_sum_target = []
    dist_list_merged = []

    # num of ligands
    num_of_ligands = len(data_list_dict[groups[0]])
    for i in range(num_of_ligands):
        edge_source = [  data_list_dict[g][i][0] for g in groups ]
        edge_target = [  data_list_dict[g][i][1] for g in groups ]
        data_list_merged += [
            (
                np.concatenate(edge_source),
                np.concatenate(edge_target)
            )
        ]

    for g in groups:
        g1, g2 = g.split('=')
        umi_sum_source += [ umi_sums[g][g1] ]
        umi_sum_target += [ umi_sums[g][g2] ]
        dist_list_merged  += [ dist_list_dict[g] ]



    umi_sums_merged = {
        'ligand' : np.concatenate(umi_sum_source),
        'receptor' : np.concatenate(umi_sum_target)
    }
    dist_list_merged = np.concatenate( dist_list_merged  )
    return (data_list_merged, umi_sums_merged, dist_list_merged)


# TODO The alternate hypotheis has many different in-equality condition
# that we are not testing therefore the llr is not correct
# H_0 : (rho_group1 = rho_group2), but their mu_x and mu_y can be different
#    or same. Which means ideally we should create an (1 + 2 + 2) parameter null model
# H_1 : (rho_group1 != rho_group2), but their mu_x and mu_y can be same or anything
#   which means ideally we should create an (2 + 2 + 2) parameter alternate model
def run_diff_copula(
    data_list_dict: dict,
    umi_sums: dict,
    group_1,
    group_2,
    dist_list_dict: dict = None,
    lig_rec_pair_list = None,
    n_jobs = 20,
    verbose = 1,
    method = 'Nelder-Mead',
    perm = 10,
    DT=True,
    cutoff=0.6,
    length_cutoff=20,
    model_type='copula',
    num_restarts = 1,
    force=False,
    quick=False,
    type_run = 'full',
    heteronomic = False,
    df_lig_rec = None
) -> pd.DataFrame:
    if isinstance(group_1, list) and len(group_1) > 1:
        raise ValueError('groups_1 has more than one group: not implemented yet')
    if isinstance(group_2, list) and len(group_2) > 1:
        raise ValueError('groups_2 has more than one group: not implemented yet')
    # null model
    # Fow all the ligands and receptors the null model
    # assumes that the interaction coefficient is same
    # We merge the groups and run the copula
    print('Running null hypothesis on merged data')
    data_list_merged, umi_sums_merged, dist_list_merged = merge_data_groups(
        [group_1, group_2],
        data_list_dict,
        umi_sums,
        dist_list_dict = dist_list_dict,
    )
    likvalues_null = []
    if dist_list_dict is None:
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(call_optimizer_dense)(
                    x,
                    y,
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    method=method,
                    perm=perm,
                    DT=DT,
                    cutoff=cutoff,
                    length_cutoff=length_cutoff,
                    model=model_type,
                    num_restarts = num_restarts,
            quick=quick) for (x,y) in data_list_merged)
        # Get likelihoods for these values
        tmp = pd.DataFrame(res,columns=['merged_null','null_mu_x',
                                        'null_mu_y','null_copula_method'])
        likvalues_null = []
        for i, row in enumerate(tmp.iterrows()):
            rho, mu_x, mu_y, meth = row[1]
            if meth == 'copula':
                likvalues_null += [log_joint_lik_perm(
                    [rho, mu_x, mu_y],
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    data_list_merged[i][0],
                    data_list_merged[i][1],
                    perm=perm,
                    DT=DT,
                    model=model_type)
                ]
            else:
                likvalues_null += [0]
        likvalues_null = -np.array(likvalues_null)
        tmp['merged_null_lik'] = likvalues_null
    else:
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(call_optimizer_dense_dist)(
                x,
                y,
                umi_sums_merged['ligand'],
                umi_sums_merged['receptor'],
                dist_list_merged,
                method=method,
                perm=perm,
                DT=DT,
                cutoff=cutoff,
                length_cutoff=length_cutoff,
                model=model_type,
                num_restarts = num_restarts
                ) for (x,y) in data_list_merged
        )
        # Get likelihoods for these values
        tmp = pd.DataFrame(res,columns=['merged_null_zero', 'merged_null_one', 'null_mu_x',
                                        'null_mu_y', 'null_copula_method'])
        likvalues_null = []

        for i, row in enumerate(tmp.iterrows()):
            rho_zero, rho_one, mu_x, mu_y, meth = row[1]
            if meth == 'copula':
                likvalues_null += [log_joint_lik_perm_dist(
                    [rho_zero, rho_one, mu_x, mu_y],
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    data_list_merged[i][0],
                    data_list_merged[i][1],
                    dist_list_merged,
                    perm=perm,
                    DT=DT,
                    model=model_type)
                ]
            else:
                likvalues_null += [0]
        likvalues_null = -np.array(likvalues_null)
        tmp['merged_null_lik'] = likvalues_null
    null_df = tmp.copy()
    # alternative model
    # Run for group1
    cop_alt_df = run_copula(
        data_list_dict,
        umi_sums,
        dist_list_dict,
        DT=DT,
        cutoff = cutoff,
        type_run='dense',
        num_restarts=1,
        df_lig_rec=df_lig_rec,
        heteronomic=True,
        groups = [group_1, group_2]
    )
    # Calculate likelihoods

    likvalues_alt = {}
    if dist_list_dict is None:
        for g in [group_1, group_2]:
            print('alternate hypothesis on ', g)
            g1, g2 = g.split('=')
            likvalues = []
            for i, [rho, mu_x, mu_y, meth] in enumerate(cop_alt_df[g].values):
                if meth == 'copula':
                    likvalues += [log_joint_lik_perm(
                        [rho, mu_x, mu_y],
                        umi_sums[g][g1],
                        umi_sums[g][g2],
                        data_list_dict[g][i][0],
                        data_list_dict[g][i][1],
                        perm=perm,
                        DT=DT,
                        model=model_type)
                    ]
                else:
                    likvalues += [0]
            likvalues_alt[g1] = (-np.array(likvalues)).copy()
    else:
        for g in [group_1, group_2]:
            print('alternate hypothesis on ', g)
            g1, g2 = g.split('=')
            likvalues = []
            for i, [rho_zero, rho_one, mu_x, mu_y, meth] in enumerate(cop_alt_df[g].values):
                if meth == 'copula':
                    likvalues += [log_joint_lik_perm_dist(
                        [rho_zero, rho_one, mu_x, mu_y],
                        umi_sums[g][g1],
                        umi_sums[g][g2],
                        data_list_dict[g][i][0],
                        data_list_dict[g][i][1],
                        dist_list_dict[g],
                        perm=perm,
                        DT=DT,
                        model=model_type)
                    ]
                else:
                    likvalues += [0]
            likvalues_alt[g1] = (-np.array(likvalues)).copy()

    likhood_df = np.hstack(
        (np.array(list(likvalues_alt.values())).T, likvalues_null[:, np.newaxis])
    )
    likhood_df = pd.DataFrame(likhood_df, columns=['alt_0', 'alt_1', 'null'])
    lr =  -2*(likhood_df_nz['null'] -
        ( likhood_df_nz['alt_0'] + likhood_df_nz['alt_1'] )
    )
    likhood_df['lr'] = lr
    return likhood_df


def run_diff_copula(
    data_list_dict: dict,
    umi_sums: dict,
    group_1,
    group_2,
    dist_list_dict: dict = None,
    lig_rec_pair_list = None,
    n_jobs = 20,
    verbose = 1,
    method = 'Nelder-Mead',
    perm = 10,
    DT=True,
    cutoff=0.6,
    length_cutoff=20,
    model='copula',
    num_restarts = 1,
    force=False,
    quick=False,
    type_run = 'full',
    heteronomic = False,
    df_lig_rec = None
) -> pd.DataFrame:
    if len(group_1) > 1:
        raise ValueError('groups_1 has more than one group: not implemented yet')
    if len(group_2) > 1:
        raise ValueError('groups_2 has more than one group: not implemented yet')
    # null model
    # Fow all the ligands and receptors the null model
    # assumes that the interaction coefficient is same
    # We merge the groups and run the copula
    data_list_merged, umi_sums_merged, dist_list_merged = merge_data_groups(
        [group_1, group_2],
        data_list_dict,
        umi_sums,
        dist_list_dict = dist_list_dict,
    )
    if dist_list_dict is None:
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(call_optimizer_dense)(
                    x,
                    y,
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    method=method,
                    perm=perm,
                    DT=DT,
                    cutoff=cutoff,
                    length_cutoff=length_cutoff,
                    model=model,
                    num_restarts = num_restarts,
            quick=quick) for (x,y) in data_list_merged)
        # Get likelihoods for these values
        tmp = pd.DataFrame(res,columns=['merged_null','null_mu_x','null_mu_y','null_copula_method'])
        likvalues = []
        for i, rho, mu_x, mu_y, meth in enumerate(res):
            if meth == 'copula':
                likvalues += [log_joint_lik_perm(
                    [rho, mu_x, mu_y],
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    data_list_merged[i][0],
                    data_list_merged[i][1],
                    perm=perm,
                    DT=DT,
                    model='copula')
                ]
            else:
                likvalues += [0]
        tmp['merged_null_lik'] = likvalues
    else:
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(call_optimizer_dense_dist)(
                x,
                y,
                umi_sums_merged['ligand'],
                umi_sums_merged['receptor'],
                dist_list_merged,
                method=method,
                perm=perm,
                DT=DT,
                cutoff=cutoff,
                length_cutoff=length_cutoff,
                model=model,
                num_restarts = num_restarts
                ) for (x,y) in data_list_merged
        )
        # Get likelihoods for these values
        tmp = pd.DataFrame(res,columns=['merged_null_zero', 'merged_null_one', 'null_mu_x',
                                        'null_mu_y', 'null_copula_method'])
        likvalues = []
        for rho_zero, rho_one, mu_x, mu_y, meth in res:
            if meth == 'copula':
                likvalues += [log_joint_lik_perm_dist(
                    [rho_zero, rho_one, mu_x, mu_y],
                    umi_sums_merged['ligand'],
                    umi_sums_merged['receptor'],
                    data_list_merged[i][0],
                    data_list_merged[i][1],
                    dist_list_merged,
                    perm=perm,
                    DT=DT,
                    model='copula')
                ]
            else:
                likvalues += [0]
        tmp['merged_null_lik'] = likvalues
    null_df = tmp.copy()
    # alternative model
    # Run for group1
    cop_alt_df = run_copula(
        data_list_dict,
        umi_sums,
        dist_list_dict,
        DT=False,
        cutoff = 0.8,
        type_run='dense',
        num_restarts=1,
        df_lig_rec=df_lig_rec,
        heteronomic=True,
        groups = [group_1, group_2]
    )
    # Calculate likelihoods
    if dist_list_dict is None:
        for g in groups:
            g1, g2 = g.split('=')
            cop_alt_df_slice = cop_alt_df[[g + '_zero', g + '_one',
                                           g1 + '_mu_x', g2 + '_mu_y',
                                            g + '_copula_method']].copy()
            pass
        pass
    # TODO complete later





def call_spearman(x, y, _umi_sum_x, _umi_sum_y):
    # If we don't get at least 10 pairs of non edges we
    # return spearman
    # return (stats.spearmanr(x / x.sum(), y / y.sum()).correlation)
    return (stats.spearmanr(x / _umi_sum_x,
                            y / _umi_sum_y).correlation)


def linearize_copula_result(
    cop_df: pd.DataFrame,
    groups: list,
    heteronomic = False
):
    cop_df_dict = {}
    if heteronomic:
        for gpair in groups:
            cols_cop = [gpair, gpair+'_copula_method']
            df_chosen = cop_df[cols_cop].copy()
            df_chosen = df_chosen.rename(columns={gpair:'copula_coeff', gpair+'_copula_method':'copula_method'})
            cop_df_dict[gpair] = df_chosen.copy()
    else:
        cop_df = cop_df.reset_index().drop('index', axis = 1)
        for gpair in groups:
            cols_cop = [gpair, 'ligand', 'receptor', gpair+'_copula_method']
            df_chosen = cop_df[cols_cop].copy()
            df_chosen = df_chosen.rename(columns={gpair:'copula_coeff', gpair+'_copula_method':'copula_method'})
            cop_df_dict[gpair] = df_chosen.copy()
    return cop_df_dict

def linearize_copula_result_dist(
    cop_df: pd.DataFrame,
    groups: list,
    heteronomic = False
):
    cop_df_dict = {}
    if heteronomic:
        for gpair in groups:
            cols_cop = [gpair+'_one', gpair+'_zero', gpair+'_copula_method']
            df_chosen = cop_df[cols_cop].copy()
            df_chosen = df_chosen.rename(columns={
                gpair+'_zero':'rho_zero',
                gpair+'_one':'rho_one',
                gpair+'_copula_method':'copula_method'}
            )
            cop_df_dict[gpair] = df_chosen.copy()
    else:
        cop_df = cop_df.reset_index().drop('index', axis = 1)
        for gpair in groups:
            cols_cop = [gpair+'_one', gpair+'_zero', gpair+'_copula_method']
            df_chosen = cop_df[cols_cop].copy()
            df_chosen = df_chosen.rename(columns={
                gpair+'_zero':'rho_zero',
                gpair+'_one':'rho_one',
                gpair+'_copula_method':'copula_method'}
            )
            cop_df_dict[gpair] = df_chosen.copy()
    return cop_df_dict


def merge_copula_results(
    cop_df_dict: dict,
):
    cop_df = pd.DataFrame()
    for gpair, df in cop_df_dict.items():
        df['celltype_direction'] = gpair
        cop_df = pd.concat([cop_df, df.copy()], axis = 0)
    cop_df = cop_df.reset_index().rename(columns={'index':'lig_rec_idx'})
    cop_df.index = range(len(cop_df))
    return cop_df


def add_spearman(
    cop_df_dict: dict,
    data_list_dict: dict,
    umi_sums: dict,
    groups = None
) -> dict:
    if groups is None:
        groups = data_list_dict.keys()

    for g1 in groups:
        data_list = data_list_dict[g1]
        g11, g12 = g1.split('=')
        res = Parallel(n_jobs=20, verbose=1)(
                delayed(call_spearman)(x,y,umi_sums[g1][g11], umi_sums[g1][g12]) for (x,y) in data_list)

        cop_df_dict[g1]['spearman'] = res
    return cop_df_dict


def add_spearman_pval(
    data_list_dict: dict,
    res_dict: dict,
    umi_sums: dict,
    percentile_cutoff = 80,
    groups = None,
    n = 1000,
    n_jobs = 20,
    permute_data = True
) -> pd.DataFrame:
    final_res_cop = pd.DataFrame()
    if groups is None:
        groups = data_list_dict.keys()
    print(permute_data)
    import time
    if permute_data:
        print('The permutation of spatial points for every L-R pair over 1K times will take a while ...')
        for g1 in groups:
            start_time = time.time()
            print('permutation test for ...', g1)
            g11, g12 = g1.split('=')
            res_bg = res_dict[g1].copy().fillna(0.0)
            data_list = data_list_dict[g1]
            if 'spearman' not in res_bg.columns:
                raise ValueError('No spearman values found')
            intersting_pairs = res_bg[res_bg['spearman'] >= np.percentile(res_bg.spearman.values, percentile_cutoff)].index

            res_bg = res_bg.loc[ intersting_pairs ]
            # res_bg['spearman_pval'] = 1.0
            # Do permutation test on them
            print('found ', len(intersting_pairs), ' pairs..')
            for i in intersting_pairs:
                print(".", end='', flush=True)
                x,y = data_list[i]
                I = res_bg.loc[i].copula_coeff
                perm_data_list = []
                for _ in range(n):
                    x1 = x.copy()
                    np.random.shuffle(x1)
                    perm_data_list += [
                        (
                        x1.copy(),
                        y
                        )

                    ]
                res_perm = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(call_spearman)(x,y,umi_sums[g1][g11], umi_sums[g1][g12]) for (x,y) in perm_data_list)
                res_perm += [I]
                pvalue = np.sum(np.array(res_perm) > I) / (n+1)
                res_bg.loc[i, 'pval'] = pvalue
            if(len(intersting_pairs) > 0):
                _,res_bg['qval'], _, _ = sm.stats.multipletests(res_bg.pval.values,alpha=0.05, method='fdr_bh')
                res_bg['celltype_direction'] = g1
                final_res_cop = pd.concat([final_res_cop, res_bg.copy()], axis = 0)
                print('\n')
            print('computing pval for ', g1, ' with ', len(intersting_pairs), ' pairs in ',
                  time.time() - start_time, ' seconds')
    else:
        raise Warning('Not implemented yet')
    return final_res_cop


def draw_copula_permutation(
        x,
        y,
        umi_sum_1,
        umi_sum_2,
        I,
        n = 1000,
        n_jobs = 20,
):
    perm_data_list = []
    for _ in range(n):
        x1 = x.copy()
        np.random.shuffle(x1)
        perm_data_list += [
            (
            x1.copy(),
            y
            )

        ]
    res_perm = Parallel(n_jobs=n_jobs, verbose=1)(
                        delayed(call_optimizer_dense)(
                            x,
                            y,
                            umi_sum_1,
                            umi_sum_2,
                            perm=20,
                            DT=False,
                            cutoff=0.8,
                            model='copula',
                            num_restarts = 1,
                            quick=False) for (x,y) in perm_data_list)
    res_perm = [r[0] for r in res_perm]
    res_perm += [I]
    print('pval -- ',np.sum(np.array(res_perm) > I) / (n+1))
    plt.hist(np.array(res_perm),alpha=0.4);
    plt.axvline( I,c='red')



def add_copula_pval(
    data_list_dict: dict,
    res_dict: dict,
    umi_sums: dict,
    dist_list_dict: dict = None,
    lig_rec_pair_list = None,
    df_lig_rec: pd.DataFrame = None,
    int_edges_new_with_selfloops: pd.DataFrame = None,
    count_df: pd.DataFrame = None,
    percentile_cutoff = 80,
    n_jobs = 20,
    verbose = 0,
    method = 'Nelder-Mead',
    DT=False,
    cutoff=0.8,
    perm = 20,
    length_cutoff=20,
    model='copula',
    num_restarts = 1,
    quick=True,
    n = 1000,
    groups = None,
    heteronomic = True,
    remove_identical = True,
    permute_data = True
) -> pd.DataFrame:

    final_res_cop = pd.DataFrame()
    if groups is None:
        groups = data_list_dict.keys()
    import time

    if permute_data:
        print('The permutation of spatial points for every L-R pair over 1K times will take a while ...')
        for g1 in groups:
            start_time = time.time()
            print('permutation test for ...', g1)
            g11, g12 = g1.split('=')
            res_bg = res_dict[g1].copy()

            # Get the interesting pairs
            if heteronomic:
                res_bg.loc[:, 'lig_rec'] = res_bg.index
                res_bg.index = range(len(res_bg))
                if remove_identical:
                    res_bg = res_bg.loc[res_bg.lig_rec.str.split('_').str[0] != res_bg.lig_rec.str.split('_').str[1]].copy()

            data_list = data_list_dict[g1]
            res_bg = res_bg.loc[res_bg.copula_method == 'copula']
            if len(res_bg) == 0:
                continue

            # if distance based then there is no copula_coeff, but there
            # is rho_zero which is copula_coeff effectively. So take a copy
            if not 'copula_coeff' in res_bg.columns:
                res_bg['copula_coeff'] = res_bg['rho_zero'].copy()


            intersting_pairs = res_bg[res_bg['copula_coeff'] >= np.percentile(res_bg.copula_coeff.values, percentile_cutoff)].index
            res_bg = res_bg.loc[ intersting_pairs ]
            res_bg['pval'] = 1.0


            # Do permutation test on them
            interval = int(len(intersting_pairs) / 10)
            print('found ', len(intersting_pairs), ' pairs..')
            progress = 0
            for i in intersting_pairs:
                #if i%interval == 0:
                print(progress, end='\r', flush=True)
                #print(i, end='\r', flush=True)
                x,y = data_list[i]
                I = res_bg.loc[i].copula_coeff
                perm_data_list = []
                for _ in range(n):
                    x1 = x.copy()
                    y1 = y.copy()
                    np.random.shuffle(x1)
                    perm_data_list += [
                        (
                        x1.copy(),
                        y1.copy()
                        )

                    ]
                if dist_list_dict is None:
                    res_perm = Parallel(n_jobs=n_jobs, verbose=verbose)(
                            delayed(call_optimizer_dense)(
                                x,
                                y,
                                umi_sums[g1][g11],
                                umi_sums[g1][g12],
                                method=method,
                                perm=perm,
                                DT=DT,
                                cutoff=cutoff,
                                length_cutoff=length_cutoff,
                                model=model,
                                num_restarts = num_restarts
                            ) for (x,y) in perm_data_list)
                    res_perm = [r[0] for r in res_perm]
                    res_perm += [I]
                    pvalue = np.sum(np.array(res_perm) > I) / (n+1)
                    res_bg.loc[i, 'pval'] = pvalue
                else:
                    res_perm = Parallel(n_jobs=n_jobs, verbose=verbose)(
                            delayed(call_optimizer_dense_dist)(
                                x,
                                y,
                                umi_sums[g1][g11],
                                umi_sums[g1][g12],
                                dist_list_dict[g1],
                                method=method,
                                perm=perm,
                                DT=DT,
                                cutoff=cutoff,
                                length_cutoff=length_cutoff,
                                model=model,
                                num_restarts = num_restarts
                                ) for (x,y) in perm_data_list)
                    res_perm = [r[0] for r in res_perm]
                    res_perm += [I]
                    pvalue = np.sum(np.array(res_perm) > I) / (n+1)
                    res_bg.loc[i, 'pval'] = pvalue
                progress += 1
            _,res_bg['qval'], _, _ = sm.stats.multipletests(res_bg.pval.values,alpha=0.05, method='fdr_bh')
            res_bg['celltype_direction'] = g1
            final_res_cop = pd.concat([final_res_cop, res_bg.copy()], axis = 0)
            print('\n')
            print('computing pval for ', g1, ' with ', len(intersting_pairs), ' pairs in ',
                  time.time() - start_time, ' seconds')
    else:
        # Make random L-R pairs
        if heteronomic:
            raise ValueError('For heteronomic ligand-receptors the ligand-receptor pairs permutation is not implemented yet.')
        df_lig_rec_rand = pd.DataFrame(columns=df_lig_rec.columns)
        df_lig_rec_rand.loc[:,'ligand'] = random.choices(
            list(df_lig_rec.ligand.values),
            k = n,
        )
        df_lig_rec_rand.loc[:,'receptor'] = random.choices(
            list(df_lig_rec.receptor.values),
            k = n,
        )
        # Prepare data list
        if int_edges_new_with_selfloops is None:
            print('Error -- int_edges_new_with_selfloops is None')
        lig_rec_pair_list = df_lig_rec_rand[['ligand', 'receptor']].values
        rand_data_list_dict, rand_umi_sums = spatial.prepare_data_list(
            count_df,
            int_edges_new_with_selfloops,
            lig_rec_pair_list,
            groups,
        )

        print('The permutation of will only permute LR pairs not spatial points')
        rand_cop_df = run_copula(
            rand_data_list_dict,
            rand_umi_sums,
            lig_rec_pair_list,
            DT=False,
            cutoff = 0.8,
            type_run='dense',
            num_restarts=2
        )
        rand_cop_dict = linearize_copula_result(
            rand_cop_df,
            groups,
        )
        for g1 in groups:
            start_time = time.time()
            print('permutation test for ...', g1)
            res_bg = res_dict[g1].copy()
            res_bg = res_bg.loc[res_bg.copula_method == 'copula']
            rand_values = rand_cop_dict[g1].copula_coeff.values
            def find_pval(num):
                bg = rand_values.copy()
                bg = np.append(bg,num)
                pvalue = np.sum(abs(bg) > abs(num)) / len(bg)
                return pvalue

            res_bg.loc[:,'pval'] = res_bg.copula_coeff.apply(find_pval)
            _,res_bg['qval'], _, _ = sm.stats.multipletests(res_bg.pval.values,alpha=0.05, method='fdr_bh')
            res_bg['celltype_direction'] = g1

            final_res_cop = pd.concat([final_res_cop, res_bg.copy()], axis = 0)
            print('\n')
            print('computing pval for ', g1, ' with ', res_bg.shape[0], ' pairs in ',
                    time.time() - start_time, ' seconds')

    return final_res_cop


# SCC related
def scc_caller(x,y,weight, lig, rec):
    return(
        [
            spatial_cross_correlation(
                x,
                y,
                weight
            ),
            # spatial_corr_test(
            #     x,
            #     y,
            #     weight,
            #     n = 100,
            #     ncores=20,
            #     plot = False
            # ),
            0.1,
            lig,
            rec
        ]
    )


def scc_caller_nonheteronomic(x,y,weight):
    return(
        [
            spatial_cross_correlation(
                x,
                y,
                weight
            )
        ]
    )


def spatial_cross_correlation(
    x,
    y,
    weight
):
    # scale weights
    weight = np.matrix(weight)
    rs = weight.sum(1)
    rs[rs == 0] = 1
    weight = weight / rs

    N = len(x)
    W = int(np.ceil(weight.sum()))
    dx = x - x.mean()
    dy = y - y.mean()
    cv1 = np.outer(dx, dy)
    cv2 = np.outer(dy, dx)
    cv1[np.isnan(cv1)] = 0
    cv2[np.isnan(cv2)] = 0

    cv = np.sum(np.multiply(weight , ( cv1 + cv2 )))
    v = np.sqrt(np.nansum(dx**2) * np.nansum(dy**2))
    SCC = (N/W) * (cv/v) / 2.0
    return(SCC)


def spatial_corr_test(x, y , weight, n = 1000, ncores=1,plot = True):
    I = spatial_cross_correlation(x,y,weight)
    print(I)

    # compute the background distribution
    xbg_corr_tets = []
    for i in range(n):
        xbg = x.copy()
        np.random.shuffle(xbg)
        cor = spatial_cross_correlation(xbg, y, weight)
        xbg_corr_tets += [cor]
    # hypo_res = Parallel(n_jobs=n_jobs, verbose=1)(
    #         delayed(hypothesis_test)(group1, group2) for (group1, group2) in data)
    if(plot):
        fig,ax = plt.subplots(1,1, figsize=(4,4))
        ax.hist(xbg_corr_tets, density=True, alpha = 0.8, label = 'Background');
        ax.axvline(I, color='red', linestyle='--', label='2.5%')

    xbg_corr_tets.append(I)
    bg = np.array(xbg_corr_tets)
    pvalue = np.sum(bg > I) / (n+1)
    return pvalue


# Run SCC
def run_scc(
    count_df: pd.DataFrame,
    lig_rec_pair_list: list,
    cop_df_dict: dict,
    int_edges_new_with_selfloops: pd.DataFrame,
    groups: list=None,
    heteronomic = False,
    lig_list = None,
    rec_list = None,
    summarization = "min"
) -> dict:
    count_df_norm = count_df.div(count_df.sum(1), axis = 0) * 1e6
    count_df_norm_log = np.log( count_df_norm + 1 )
    if groups is None:
        groups = int_edges_new_with_selfloops.interaction.unique()


    for gpair in groups:
        print(gpair)
        g11, g12 = gpair.split('=')

        lr_pairs_g1 = list(
            int_edges_new_with_selfloops.loc[
                int_edges_new_with_selfloops.interaction == gpair,
                ["cell1", "cell2","distance"]
            ].to_records(index=False)
        )


        if g11 == g12:
            G = nx.Graph()
            G.add_weighted_edges_from(lr_pairs_g1)
            self_loops = [(node, node, 1) for node in G.nodes]
            G.add_weighted_edges_from(self_loops)
        else:
            G = nx.DiGraph()
            G.add_weighted_edges_from(lr_pairs_g1)
        print(G)

        # lr_pairs_g1 = int_edges_new_with_selfloops.loc[
        #         int_edges_new_with_selfloops.interaction == gpair,
        #         ["cell1", "cell2"]
        #     ]

        # if g11 == g12:
        #     G = nx.Graph()
        #     G.add_edges_from(lr_pairs_g1.values)
        #     self_loops = [(node, node) for node in G.nodes]
        #     G.add_edges_from(self_loops)
        # else:
        #     G = nx.DiGraph()
        #     G.add_edges_from(lr_pairs_g1.values)
        # print(G)

        weight = nx.adjacency_matrix(G).todense()
        data_list = []
        if heteronomic:
            if (lig_list is None) or (rec_list is None):
                raise ValueError('lig_list or rec_list is None')

            for _,(lig, rec) in enumerate(zip(lig_list, rec_list)):
                _lig = [l for l in lig if l != None]
                _rec = [r for r in rec if r != None]
                if summarization == 'min':
                    data_list += [(
                        count_df_norm_log.loc[ list(G.nodes), _lig].min(axis=1).values,
                        count_df_norm_log.loc[ list(G.nodes), _rec].min(axis=1).values,
                        weight.copy()
                        )
                    ]
                elif summarization == "mean":
                    data_list += [(
                        count_df_norm_log.loc[ list(G.nodes), _lig].max(axis=1).values,
                        count_df_norm_log.loc[ list(G.nodes), _rec].max(axis=1).values,
                        weight.copy()
                        )
                    ]
                elif summarization == "sum":
                    data_list += [(
                        count_df_norm_log.loc[ list(G.nodes), _lig].sum(axis=1).values,
                        count_df_norm_log.loc[ list(G.nodes), _rec].sum(axis=1).values,
                        weight.copy()
                        )
                    ]
                else:
                    raise ValueError('summarization is not recognized')
            hypo_res = Parallel(n_jobs=20, verbose=1)(
                delayed(scc_caller_nonheteronomic)(x,y,w) for (x,y,w) in data_list)
            st_scc_df = pd.DataFrame(hypo_res, columns = ['scc'])
            cop_df_dict[gpair]['scc'] = st_scc_df.scc.values
        else:
            if lig_rec_pair_list is None:
                raise ValueError('lig_rec_pair_list is None')
            for lig,rec in lig_rec_pair_list:
                data_list += [( count_df_norm_log.loc[ list(G.nodes), lig].values,
                            count_df_norm_log.loc[ list(G.nodes), rec ].values,
                            weight.copy(),
                            lig,
                            rec
                            )
                            ]

            hypo_res = Parallel(n_jobs=20, verbose=1)(
                delayed(scc_caller)(x,y,w,l,r) for (x,y,w,l,r) in data_list)

            st_scc_df = pd.DataFrame(hypo_res, columns = ['scc', 'pval', 'ligand', 'receptor'])
            cop_df_dict[gpair]['scc'] = st_scc_df.scc.values

    return cop_df_dict


# Run SDM
def run_sdm(
    adata_orig: sc.AnnData,
    int_edges_new_with_selfloops: pd.DataFrame,
    min_cell: int = 3,
    groups: list=None,
    nproc: int = 10,
    species = 'human',
    heteronomic = False,
    add_self_loops = True
) -> dict:

    sdm_dfs = {}
    adata = adata_orig.copy()
    adata.raw = adata.copy()

    if groups is None:
        groups = int_edges_new_with_selfloops.interaction.unique()


    for gpair in groups:
        st = time.time()
        print(gpair)
        g11, g12 = gpair.split('=')
        lr_pairs_g1 = list(
            int_edges_new_with_selfloops.loc[
                int_edges_new_with_selfloops.interaction == gpair,
                ["cell1", "cell2","distance"]
            ].to_records(index=False)
        )


        if g11 == g12:
            G = nx.Graph()
            G.add_weighted_edges_from(lr_pairs_g1)
            # if add_self_loops:
            #     self_loops = [(node, node, 1) for node in G.nodes]
            #     G.add_weighted_edges_from(self_loops)
        else:
            G = nx.DiGraph()
            G.add_weighted_edges_from(lr_pairs_g1)



        print(G)


        # lr_pairs_g1 = int_edges_new_with_selfloops.loc[
        #         int_edges_new_with_selfloops.interaction == gpair,
        #         ["cell1", "cell2"]
        #     ]
        # if g11 == g12:
        #     G = nx.Graph()
        #     G.add_edges_from(lr_pairs_g1.values)
        #     self_loops = [(node, node) for node in G.nodes]
        #     G.add_edges_from(self_loops)
        # else:
        #     G = nx.DiGraph()
        #     G.add_edges_from(lr_pairs_g1.values)

        # print(G)
        adata_gpair = adata[ list(G.nodes), ].copy()
        # Following the tutorial from SpatialDM we
        # need to take log normalization of the count data
        adata_gpair.raw = adata_gpair.copy()
        sc.pp.normalize_total(adata_gpair, target_sum=None)
        sc.pp.log1p(adata_gpair)
        print(adata_gpair.shape)
        adata_gpair.uns['single_cell'] = False
        adj = nx.adjacency_matrix(G)
        # TODO Not sure if we should normalize
        # to match the the graph from copulacci
        adj = normalize(adj, norm='l1', axis=1)
        adata_gpair.obsp['weight'] = adj.todense()
        adata_gpair.obsp['nearest_neighbors'] = adj.todense()
        sc.pp.normalize_total(adata_gpair, target_sum=None)
        sc.pp.log1p(adata_gpair)
        try:
            sdm.extract_lr(adata_gpair, species, min_cell=min_cell)
        except Exception as e:
            print(f"no effective LR found : {e}")
            continue
        sdm.spatialdm_global(adata_gpair, 1000, specified_ind=None, method='z-score', nproc=nproc)     # global Moran selection
        sdm.sig_pairs(adata_gpair, method='z-score', fdr=True, threshold=0.1)     # select significant pairs
        sdm.spatialdm_local(adata_gpair, n_perm=1000, method='z-score', specified_ind=None, nproc=nproc)     # local spot selection
        sdm.sig_spots(adata_gpair, method='z-score', fdr=False, threshold=0.1)

        ca1 = concat_obj([adata_gpair], ['A1'], species, 'z-score', fdr = True)

        if heteronomic:
            df_tmp = adata_gpair.uns['geneInter'].copy()
            df_tmp['global_I'] = adata_gpair.uns['global_I']
            df_tmp['global_pval'] = ca1.uns['p_df']['A1']
            sdm_dfs[gpair] = df_tmp.copy()
        else:
            lig_rec_list = []
            for i in range(adata_gpair.uns['ligand'].shape[0]):
                ligands =  adata_gpair.uns['ligand'].iloc[i].values
                receptors = adata_gpair.uns['receptor'].iloc[i].values
                ligands = set([l for l in ligands if l is not None])
                receptors  = set([l for l in receptors if l is not None])
                val = adata_gpair.uns['global_I'][i]
                p = ca1.uns['p_df']['A1'][i]
                # Make a datafram with two columns
                for l in ligands:
                    for r in receptors:
                        if l != r:
                            lig_rec_list.append([l, r, val, p])

            global_score = pd.DataFrame(lig_rec_list).rename(columns={
                0:'ligand',1:'receptor',
                2:'global_I',3:'global_pval'}
            )
            sdm_dfs[gpair] = global_score.copy()
        et = time.time()
        print('time taken for ', gpair, ' is ', et - st, ' seconds')
    return sdm_dfs