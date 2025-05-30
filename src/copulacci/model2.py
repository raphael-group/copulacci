# pylint: disable=C0103, C0114, C0301
from collections import namedtuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from joblib import Parallel, delayed
import networkx as nx
import tqdm
import scanpy as sc
import time
from copulacci import spatial
from scipy import integrate
from scipy.signal import find_peaks


EPSILON = 1.1920929e-07
CopulaParams = namedtuple('CopulaParams', [
    'perm', 'DT', 'dist_list',
    'mu_x', 'mu_y',
    'copula_mode' , 'return_sum',
    'rho_one_start'
    ]
)
CopulaParams.__new__.__defaults__ = (1, False, None,
                                     None, None,
                                     'vanilla', True,
                                     0.01)
OptParams = namedtuple('OptParams', ['method', 'tol', 'maxiter', 'num_starts'])
OptParams.__new__.__defaults__ = ('Nelder-Mead', 1e-6, 1e+6, 1)

SpatialParams = namedtuple('SpatialParams', ['data_type',  'coord_type', 'n_neighs', 'n_rings',
                                             'radius', 'distance_aware', 'deluanay']
            )
SpatialParams.__new__.__defaults__ = ('visium', 'grid', 6, 1, None, False, False)



def get_dt_cdf(x, lam, DT=True):
    '''
        This function implements distributional tranformation as described in the
        following papers:
        1. https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-02884-2
        2. https://www.cs.cmu.edu/~pradeepr/paperz/wics1398.pdf
        Patameters:
        -----------
        x: array-like
            The input data
        lam: float
            The lambda parameter of the Poisson distribution
        dt: bool
            Whether to use the distributional transformation with
            uniform random variable or not. If False, the distributional
            transformation is used with the average of the two cdfs.
    '''
    u_x = stats.poisson.cdf(x, lam).clip(EPSILON, 1 - EPSILON)
    u_x2 = stats.poisson.cdf(x-1, lam).clip(EPSILON, 1 - EPSILON)
    if DT:
        v = stats.uniform.rvs(size=len(x))
        r = u_x * v + u_x2 * (1 - v)
    else:
        r = (u_x + u_x2) / 2.0
    idx_adjust = np.where(1 - r < EPSILON)[0]
    r[idx_adjust] = r[idx_adjust] - EPSILON
    idx_adjust = np.where(r < EPSILON)[0]
    r[idx_adjust] = r[idx_adjust] + EPSILON

    return stats.norm.ppf(r)


def sample_from_copula(
        n_array,
        mu_x,
        mu_y,
        **kwargs
):
    '''
        This function samples from a copula distribution to simulate
        bivariate count datasets.
        Parameters:
        -----------
        n_array: array-like
            The array of sample sizes
        mu_x: float
            The mean of the first variable
        mu_y: float
            The mean of the second variable
        **kwargs:
            'coeff' - the correlation coefficient if a float
                      if a list then assumed to contain distance
            'dist' - The distance vector between two count values
            while generating spatial expression.
            'l' - The decay parameter for the exponential decay
            't' - The coefficient of the exponential decay
            The last two attributes are used to simulate spatial
            expression as a function of distance from an axis.
    '''
    mode = None
    coeff_arg = kwargs.get('coeff', 0.0)
    if isinstance(coeff_arg, list):
        coeff_list = np.array(coeff_arg)
        dist = kwargs.get('dist', None)
        l = kwargs.get('l', None)
        t = kwargs.get('t', 0)
        if dist is None:
            mode = 'dist'
        else:
            mode = 'grad'
            if l is None:
                l = max(dist)
    else:
        mode = 'vanilla'
        coeff = coeff_arg
    if mode == 'vanilla' or mode == 'dist':
        lam_x = n_array * np.exp(mu_x)
        lam_y = n_array * np.exp(mu_y)
        if mode == 'vanilla':
            cov = linalg.toeplitz([1,coeff])
            samples = np.random.multivariate_normal(
                [0,0],
                cov,
                size = len(n_array)
            )
        else:
            samples = np.array([np.random.multivariate_normal(
                [0,0],
                [[1,coeff],[coeff,1]],
                size = 1
            ).squeeze() for coeff in coeff_list])
    elif mode == 'grad':
        lam_x = (n_array * np.exp(mu_x)) * np.exp(-(t * (dist/l**2)))
        lam_y = (n_array * np.exp(mu_y)) * np.exp(-(t * (dist/l**2)))
        samples = np.array([np.random.multivariate_normal(
            [0,0],
            [[1,coeff],[coeff,1]],
            size = 1
        ).squeeze() for coeff in coeff_list])
    else:
        raise ValueError('Invalid mode')

    samples = pd.DataFrame(samples, columns=['x', 'y'])
    cdf_x = stats.norm.cdf(samples.x)
    cdf_y = stats.norm.cdf(samples.y)
    samples.loc[:, 'x'] = stats.poisson.ppf(cdf_x, lam_x)
    samples.loc[:, 'y'] = stats.poisson.ppf(cdf_y, lam_y)
    return samples


def log_joint_lik(
        params,
        x,
        y,
        umi_sum_1,
        umi_sum_2,
        copula_params
):
    '''
        This function computes the log joint likelihood of the
        copula model.
        Parameters:
        -----------
        params: array-like
            The parameters of the copula model contains either 3 or 5 values
            depending on the mode.
        x: array-like
            The first variable
        y: array-like
            The second variable
        umi_sum_1: array-like
            The UMI sums of the first variable
        umi_sum_2: array-like
            The UMI sums of the second variable
        copula_params: namedtuple of CopulaParams
            'perm' - Number of permutations
            'DT' - Whether to use the distributional transformation
            with uniform distribution or not
            'dist_list' - The list of distances between two spots
            default to a list of zeros
            'copula_mode' - Either 'dist' or 'vanilla' default to 'vanilla'
            'return_sum' - Whether to return the sum of the log likelihood
            vs individual log likelihoods
    '''
    mode = copula_params.copula_mode
    if len(params) == 3 and mode == 'dist':
        raise ValueError('Invalid number of parameters')
    if mode == 'vanilla':
        coeff = params[0]
        mu_1 = params[1]
        mu_2 = params[2]
    elif mode == 'single_param':
        coeff = params[0]
        mu_1 = copula_params.mu_x
        mu_2 = copula_params.mu_y
        if mu_1 is None or mu_2 is None:
            raise ValueError('mu_1 and mu_2 must be provided for mode single_param. Make sure `quick` is True')
    else:
        rho_zero = params[0]
        rho_one = params[1]
        dist_list = copula_params.dist_list
        coeff_list = rho_zero * np.exp(-1 * dist_list * rho_one)
        mu_1 = params[2]
        mu_2 = params[3]
    if mode == 'dist' and dist_list is None:
        raise ValueError('dist_list must be provided for mode dist')
    perm = copula_params.perm
    DT = copula_params.DT
    return_sum = copula_params.return_sum

    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # Get the distributional transformation
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1, DT=DT)
            r_y += get_dt_cdf(y, lam2, DT=DT)
        z = np.column_stack([r_x/perm, r_y/perm])
    else:
        z = np.column_stack([r_x, r_y])

    # Get likelihood
    if return_sum:
        if mode == 'dist':
            det = 1 - coeff_list**2
            term1 = np.sum(
                -0.5 * (((coeff_list**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) -
                                   2 * (coeff_list/det) * z[:,0] * z[:,1])
            )
            term3 = np.sum(-0.5 * np.log(det+EPSILON))
        else:
            det = 1 - coeff**2
            term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) -
                                   2 * (coeff/det) * z[:,0] * z[:,1])
            )
            term3 = -0.5 * len(x) * np.log(det+EPSILON)
        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON))) +
            np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON)))
        )
        logsum = term1 + term2 + term3
        return -logsum

    if mode == 'dist':
        det = 1 - coeff_list**2
        term1 = -0.5 * (((coeff_list**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) -
                        2 * (coeff_list/det) * z[:,0] * z[:,1])
    else:
        det = 1 - coeff**2
        term1 = -0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) -
                        2 * (coeff/det) * z[:,0] * z[:,1])
    term2 = (
        np.log(stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) ) +
        np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) )
    )
    term3 = -0.5 * np.log(det+EPSILON)
    return term1 + term2 + term3


def only_log_lik(
        mu_1,
        mu_2,
        umi_sum_1,
        umi_sum_2,
        x,
        y,
        copula_params,
        c = np.linspace(-0.99, 0.99, 1000)
):
    """
    Take double derivative of the log likelihood function
    """
    DT = copula_params.DT
    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)

    z = np.column_stack([r_x, r_y])

    def f(coeff):
        det = 1 - coeff**2
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )
        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1+term2+term3
        return -logsum

    val = np.array([f(coeff) for coeff in c])

    return val


def find_local_minima(
        mu_1,
        mu_2,
        umi_sum_1,
        umi_sum_2,
        x,
        y,
        copula_params
):
    """
    Take double derivative of the log likelihood function
    """
    DT = copula_params.DT
    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)

    z = np.column_stack([r_x, r_y])

    def f(coeff):
        det = 1 - coeff**2
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )
        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1+term2+term3
        return -logsum

    left = minimize(f, -0.9, bounds=((-0.99,0.99),))
    right = minimize(f, 0.9, bounds=((-0.99,0.99),))
    return([left.x[0], right.x[0]])


def test_num_local_minima(
    val,
    c = np.linspace(-0.99, 0.99, 1000),
    distance = None
):
    peaks, _ = find_peaks(-val,distance=distance)
    max_peak = 0
    if len(peaks) > 1:
        diff = np.abs(val[peaks[0]] - val[peaks[1]])
        max_peak = c[peaks[np.argmax(val[peaks])]]
    else:
        diff = 0
        max_peak = c[peaks[0]]
    return (diff, max_peak)


def diff_using_num(
        mu_1,
        mu_2,
        umi_sum_1,
        umi_sum_2,
        x,
        y,
        copula_params,
        opt_params,
        x_train = None,
        do_first_order = False
):
    """
    Take double derivative of the log likelihood function
    """
    DT = copula_params.DT
    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)

    z = np.column_stack([r_x, r_y])

    def f(coeff):
        det = 1 - coeff**2
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        # term2 = (
        #     np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
        #     np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        # )
        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1+ term3
        return -logsum
    if x_train is None:
        c = np.linspace(-0.99, 0.99, 1000)
    else:
        c = x_train
    val = np.array([f(coeff) for coeff in c])
    y_spl = UnivariateSpline(c, val,s=0,k=4)
    # Take second derivative
    y_spl_2d = y_spl.derivative(n=2)
    d2l = y_spl_2d(c)
    if do_first_order:
        y_spl_1d = y_spl.derivative(n=1)
        d1l = y_spl_1d(c)
        return val, d2l, d1l
    return val, d2l


def get_peak_distance(
    x,
    y,
    us1,
    us2,
    copula_params,
    opt_params
):
    sx = np.log(x.sum() / us1.sum())
    sy = np.log(y.sum() / us2.sum())
    _,d1lik = diff_using_num(sx,sy,us1,us2,x,y,copula_params,
                                         opt_params,do_first_order=True)

    s = np.sign(d1lik)
    ind_range = np.where(np.r_[s[:-1]!=s[1:], [False]])[0]
    if len(ind_range) > 1:
        val = only_log_lik(sx,sy,us1,us2,x,y,copula_params)
        peaks, _ = find_peaks(-np.array(val))
        if len(peaks) > 0:
            diff = np.abs(val[peaks[0]] - val[peaks[1]])
            return diff
        return 0
    return 0


def get_d2l_values(
        x,
        y,
        umi_sum_1,
        umi_sum_2,
        copula_params,
        opt_params,
        **kwargs
):
    """
    Take double derivative of the log likelihood function
    """
    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())
    d2l = diff_using_num(
        mu_x_start,
        mu_y_start,
        umi_sum_1,
        umi_sum_2,
        x,
        y,
        copula_params,
        opt_params
    )
    return min(d2l)


def calculate_mahalanobis_distance(
    params,
    x,
    y,
    umi_sum_1,
    umi_sum_2,
    copula_params
):
    """
    This function calculates the Mahalanobis distance
    between two variables
    Parameters:
    -----------
    """
    DT = copula_params.DT
    coeff = params[0]
    mu_1 = params[1]
    mu_2 = params[2]
    lam1 = umi_sum_1 * np.exp(mu_1)
    lam2 = umi_sum_2 * np.exp(mu_2)

    # Get the distributional transformation
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = get_dt_cdf(y, lam2, DT=DT)
    z = np.column_stack([r_x, r_y])
    det = 1 - coeff**2
    return (1/det) * ( z[:,0]**2 + z[:,1]**2 - 2 * coeff * z[:,0] * z[:,1] )


def call_optimizer(
        x,
        y,
        umi_sum_1,
        umi_sum_2,
        copula_params,
        opt_params,
        **kwargs
):
    """
        This function calls the likelihood function and find the parameters
        for maxumum likelihood estimation.
        Parameters:
        -----------
        copula_params: namedtuple of CopulaParams
        opt_params: namedtuple of OptParams
            'method' - The optimization method e.g. Nelder-Mead, BFGS etc.
            'tol' - The tolerance level for the optimization
            'maxiter' - The maximum number of iterations
            'num_starts' - The number of random starts for the optimization
        **kwargs:
            'use_zero_cutoff' - Whether to use zero cutoff or not
            'zero_cutoff' - The percengage of zero cutoff e.g. if one of
                the two variables has more than 90% zeros then skip optimization
            'use_length_cutoff' - Whether to use length cutoff or not
            'length_cutoff' - The minimum length of the two variables
            'use_zero_pair_cutoff' - Whether to use zero pair cutoff or not
            'zero_pair_cutoff' - The percengage of zero pair cutoff e.g. if
                the two variables have more than 90% zero-zero pairs then skip
                optimization
            'use_mu_cutoff' - Whether to use mu cutoff or not
            'mu_cutoff' - The minimum GLM mean of the two variables
            'force_opt' - Whether to force optimization or not if set to True
                then optimization is done even if the above cutoffs are not met
    """
    skip_opt = False
    opt_status = 'copula'
    copula_mode = copula_params.copula_mode
    # Get cutoff related parameters
    use_zero_cutoff = kwargs.get('use_zero_cutoff', False)
    zero_cutoff = kwargs.get('zero_cutoff', 0.8)
    use_length_cutoff = kwargs.get('use_length_cutoff', False)
    length_cutoff = kwargs.get('length_cutoff', 20)
    use_zero_pair_cutoff = kwargs.get('use_zero_pair_cutoff', False)
    zero_pair_cutoff = kwargs.get('zero_pair_cutoff', 0.6)
    use_mu_cutoff = kwargs.get('use_mu_cutoff', False)
    mu_cutoff = kwargs.get('mu_cutoff', -8)
    force_opt = kwargs.get('force_opt', False)
    stability_filter = kwargs.get('stability_filter', False)
    stability_cutoff = kwargs.get('stability_cutoff', 0.0)
    local_minima_filter = kwargs.get('local_minima_filter', False)
    quick = kwargs.get('quick', False)
    run_find_peaks = kwargs.get('run_find_peaks', False)
    add_diagonostic = kwargs.get('add_diagonostic', False)
    x_train = kwargs.get('x_train', np.linspace(-0.99, 0.99, 1000))
    if quick:
        # local minima will be global minima
        if not(local_minima_filter or stability_filter):
            raise ValueError('quick mode cannot be used without local_minima_filter or stability_filter')

    # If either of the two variables is empty return with status empty
    return_vec = None
    if x.sum() == 0 or y.sum() == 0:
        if (copula_mode == 'vanilla') or (copula_mode == 'single_param'):
            return_vec = [0, 0, 0, 'all_zero']
        else:
            return_vec = [0, 0, 0, 0, 'all_zero']

    if (len(x) == 0) or (len(y) == 0):
        if (copula_mode == 'vanilla') or (copula_mode == 'single_param'):
            return_vec = [0, 0, 0, 'all_zero']
        else:
            return_vec = [0, 0, 0, 0, 'all_zero']

    if return_vec is not None:
        if add_diagonostic:
            return return_vec + [None]
        return return_vec

    if force_opt is False:
        if use_zero_cutoff:
            if (((x==0).sum() / len(x) > zero_cutoff) or ((y==0).sum() / len(y) > zero_cutoff)):
                skip_opt = True
                opt_status = 'skip_zero_cutoff'
        if use_length_cutoff and (skip_opt is False):
            if (len(x) < length_cutoff) | (len(y) < length_cutoff):
                skip_opt = True
                opt_status = 'skip_length_cutoff'
        if use_zero_pair_cutoff and (skip_opt is False):
            zz_percent = ((x == 0) & (y==0)).sum() / len(x)
            if zz_percent >= zero_pair_cutoff:
                skip_opt = True
                opt_status = 'skip_zero_pair_cutoff'

    mu_x_start = np.log(x.sum() / umi_sum_1.sum())
    mu_y_start = np.log(y.sum() / umi_sum_2.sum())

    if stability_filter:
        val, d2l = diff_using_num(
            mu_x_start,
            mu_y_start,
            umi_sum_1,
            umi_sum_2,
            x,
            y,
            copula_params,
            opt_params,
            x_train = x_train
        )
        # If double derivative is negative at any point then skip optimization
        # as the function is not convex
        area = min(d2l)
        if stability_cutoff < 0.0:
            s = np.sign(d2l)
            ind_range = np.where(np.r_[s[:-1]!=s[1:], [False]])[0]
            if len(ind_range) > 1:
                area = integrate.simpson(
                    d2l[ind_range[0]+1:ind_range[1]+1],
                    dx=0.001
                )

        if area < stability_cutoff:
            skip_opt = True
            opt_status = 'skip_stability_filter'
        elif run_find_peaks:
            peaks, _ = find_peaks(-val)
            if len(peaks) == 1:
                max_peak = x_train[peaks[0]]
                lik_null = val[len(x_train)//2]
                llr = -2 * (val[peaks[0]] - lik_null)
                if add_diagonostic:
                    return [max_peak, mu_x_start, mu_y_start, opt_status, llr]
                return [max_peak, mu_x_start, mu_y_start, opt_status]
            else:
                opt_status = 'skip_local_minima_filter'


    if local_minima_filter:
        lik, d2l, d1l = diff_using_num(
            mu_x_start,
            mu_y_start,
            umi_sum_1,
            umi_sum_2,
            x,
            y,
            copula_params,
            opt_params,
            x_train = x_train,
            do_first_order = True
        )
        # times it crosses zero
        # calculate fisher information of the likelihood
        # E[f''(x)] = -E[log(f(x))]
        # for us it's the negative log likelihood
        # so we drop the negative sign
        # TODO we are not doing fisher but
        # doing the ratio test with null coeff=0
        llr = 0

        peaks, _ = find_peaks(-lik) # find peaks
        if len(peaks) > 1:
            opt_status = 'skip_local_minima_filter'
            return_vec = [0, mu_x_start, mu_y_start, opt_status]
        elif len(peaks) == 1:
            max_peak = x_train[peaks[0]]
            lik_null = lik[len(x_train)//2]
            llr = -2 * (lik[peaks[0]] - lik_null)
            return_vec = [max_peak, mu_x_start, mu_y_start, opt_status]
        else:
            # No minima is weird
            # opt_status = 'skip_no_local_minima'
            return_vec = [0, mu_x_start, mu_y_start, opt_status]
        if add_diagonostic:
            return_vec = return_vec + [llr]
        #if not run_quick_optima:
        if run_find_peaks:
            return return_vec

        # if min(d2l) < 0.0:
        #     s = np.sign(d1l)
        #     ind_range = np.where(np.r_[s[:-1]!=s[1:], [False]])[0]
        #     if len(ind_range) > 1:
        #         skip_opt = True
        #         opt_status = 'skip_local_minima_filter'


    if use_mu_cutoff and (skip_opt is False):
        if (mu_x_start < mu_cutoff) or (mu_y_start < mu_cutoff):
            skip_opt = True
            opt_status = 'skip_mu_cutoff'

    if skip_opt is True:
        if (copula_mode == 'vanilla') or (copula_mode == 'single_param'):
            return_vec = [0, mu_x_start, mu_y_start, opt_status]
        else:
            return_vec = [0, 0, mu_x_start, mu_y_start, opt_status]
        if add_diagonostic:
            return return_vec + [llr]
        return return_vec

    # Get optimization parameters
    method = opt_params.method
    tol = opt_params.tol
    num_starts = opt_params.num_starts

    # Get likelihood parameters
    dist_list = copula_params.dist_list
    if (copula_mode == 'dist' and dist_list is None):
        raise ValueError('dist_list must be provided for mode dist')

    if quick:
        # Make is single parameter that is optimize for $\rho$
        # switch to single_parameter mode
        if copula_mode == 'dist':
            raise ValueError('quick mode is not supported with distances \n as it\'s two parameters')

        if not run_find_peaks:
            copula_params = copula_params._replace(copula_mode='single_param')
            # set copula param's mu_x to the start estimate
            copula_params = copula_params._replace(mu_x=mu_x_start)
            copula_params = copula_params._replace(mu_y=mu_y_start)
            start_params = np.array([0.0])
            res = minimize(
                log_joint_lik,
                x0=start_params,
                method=method,
                bounds=[(-0.99, 0.99)],
                args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                tol=tol
            )
            return_vec = list(res['x']) + [mu_x_start, mu_y_start, opt_status]
        else:
            # Given it's single parameter optimization we can use the
            # find_peaks method to find the local peak
            # this will be a problem if there are more than one peak
            # in the likelihood function and in which case we will
            # have to abort the optimization
            lik = only_log_lik(mu_x_start,mu_y_start,umi_sum_1,umi_sum_2,x,y,copula_params,x_train)
            peaks, _ = find_peaks(-lik)
            if len(peaks) == 1:
                max_peak = x_train[peaks[0]]
                lik_null = lik[len(x_train)//2]
                llr = -2 * (lik[peaks[0]] - lik_null)
                return_vec = [max_peak, mu_x_start, mu_y_start, opt_status]
            else:
                opt_status = 'skip_local_minima_filter'
                return_vec = [0, mu_x_start, mu_y_start, opt_status]

        if add_diagonostic:
            return_vec = return_vec + [llr]
        return return_vec


    if num_starts > 1:
        # Take uniform random samples from the parameter space
        # as different starting points for the optimization
        if copula_mode == 'vanilla':
            coeff_start = np.random.uniform(-0.99, 0.99, num_starts)
            mu_1_start = np.random.uniform(mu_x_start - 2, 0, num_starts)
            mu_2_start = np.random.uniform(mu_y_start - 2, 0, num_starts)
            start_params = np.column_stack([coeff_start, mu_1_start, mu_2_start])
            results = []
            for i in range(num_starts):
                res = minimize(
                    log_joint_lik,
                    x0 = start_params[i],
                    method=method,
                    bounds=[(-0.99, 0.99), (start_params[i][1] - 5, 0), (start_params[i][2] - 5, 0)],
                    args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                    tol=tol
                )
                results += [res.copy()]
            # Take the converged result with the lowest log likelihood
            results = [res for res in results if res['success'] is True]
            best_result = min(results, key=lambda x: x['fun'])
            return_vec = list(best_result['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec
        elif copula_mode == 'single_param':
            coeff_start = np.random.uniform(-0.99, 0.99, num_starts)
            start_params = np.column_stack([coeff_start])
            results = []
            for i in range(num_starts):
                res = minimize(
                    log_joint_lik,
                    x0 = start_params[i],
                    method=method,
                    bounds=[(-0.99, 0.99)],
                    args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                    tol=tol
                )
                results += [res.copy()]
            # Take the converged result with the lowest log likelihood
            results = [res for res in results if res['success'] is True]
            best_result = min(results, key=lambda x: x['fun'])
            return_vec = list(best_result['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec
        else:
            rho_zero_start = np.random.uniform(-0.99, 0.99, num_starts)
            rho_one_start = np.random.uniform(0.0, 1.0, num_starts)
            mu_1_start = np.random.uniform(mu_x_start - 5, 0, num_starts)
            mu_2_start = np.random.uniform(mu_y_start - 5, 0, num_starts)
            start_params = np.column_stack([rho_zero_start, rho_one_start, mu_1_start, mu_2_start])
            results = []
            for i in range(num_starts):
                res = minimize(
                    log_joint_lik,
                    x0 = start_params[i],
                    method=method,
                    bounds=[
                        (-0.99, 0.99),
                        (0.0, 10.0),
                        (mu_x_start-5, 0),
                        (mu_y_start-5, 0)
                    ],
                    args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                    tol=tol
                )
                results += [res.copy()]
            # Take the converged result with the lowest log likelihood
            results = [res for res in results if res['success'] is True]
            best_result = min(results, key=lambda x: x['fun'])
            return_vec = list(best_result['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec
    else:
        if copula_mode == 'vanilla':
            start_params = np.array([0.0, mu_x_start, mu_y_start])
            res = minimize(
                log_joint_lik,
                x0=start_params,
                method=method,
                bounds=[(-0.99, 0.99), (mu_x_start-5, 0), (mu_y_start - 5, 0)],
                args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                tol=tol
            )
            return_vec = list(res['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec
        elif copula_mode == 'single_param':
            start_params = np.array([0.0])
            res = minimize(
                log_joint_lik,
                x0=start_params,
                method=method,
                bounds=[(-0.99, 0.99)],
                args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                tol=tol
            )
            return_vec = list(res['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec
        else:
            start_params = np.array([0.0, copula_params.rho_one_start, mu_x_start, mu_y_start])
            res = minimize(
                log_joint_lik,
                start_params,
                method=method,
                bounds=[
                    (-0.99, 0.99),
                    (0.0, 10.0),
                    (mu_x_start-5, 0),
                    (mu_y_start-5, 0)
                ],
                args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                tol=tol
            )
            return_vec = list(res['x']) + [opt_status]
            if add_diagonostic:
                return return_vec + [None]
            return return_vec


def run_copula(
    data_list_dict,
    umi_sums,
    dist_list_dict = None,
    lig_rec_pair_list = None,
    n_jobs = 2,
    verbose = 1,
    groups = None,
    df_lig_rec_index = None,
    heteronomic = False,
    copula_params = CopulaParams(),
    opt_params = OptParams(),
    **kwargs
):
    """
        This function wraps the copula model and runs it on the data.
        Parameters:
        -----------
        data_list_dict: dict
            The dictionary of dataframes containing the count data
            for each group So a dictionary of lists
        umi_sums: dict
            The dictionary of UMI sums for each group divided
            into nested dictionaries
        TODO - Add more documentation
    """
    cop_df_dict = {}
    add_diagonostic = kwargs.get('add_diagonostic', False)
    if groups is None:
        groups = list(data_list_dict.keys())
    if dist_list_dict is None and copula_params.copula_mode == 'dist':
        warnings.warn('dist_list_dict must be provided for mode dist falling')
        copula_params = copula_params._replace(copula_mode='vanilla')
    # Run the copula model
    for g1 in groups:
        data_list = data_list_dict[g1]
        print(g1)
        g11, g12 = g1.split('=')
        if copula_params.copula_mode == 'dist':
            copula_params = copula_params._replace(
                dist_list = dist_list_dict[g1]
            )
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(call_optimizer)(
                x,
                y,
                umi_sums[g1][g11],
                umi_sums[g1][g12],
                copula_params,
                opt_params,
                **kwargs
            ) for x, y in data_list
        )
        if copula_params.copula_mode == 'vanilla':
            columns_names = [
                'copula_coeff',
                'mu_x',
                'mu_y',
                'copula_method'
            ]
            if add_diagonostic:
                columns_names += ['fisher_info']
        else:
            cop_df_dict[g1] = pd.DataFrame(res, columns=[
                    'rho_zero',
                    'rho_one',
                    'mu_x',
                    'mu_y',
                    'copula_method'
            ])
            if add_diagonostic:
                columns_names += ['fisher_info']

        cop_df_dict[g1] = pd.DataFrame(res, columns=columns_names)
        # Count the number of results that are empty
        tmp = cop_df_dict[g1].loc[
            cop_df_dict[g1].copula_method == 'copula']
        print(f"Number of non empty results: {tmp.shape[0]}")

        if heteronomic:
            if df_lig_rec_index is None:
                raise ValueError('df_lig_rec is None')
            cop_df_dict[g1].index = df_lig_rec_index
        else:
            if lig_rec_pair_list is None:
                raise ValueError('lig_rec_pair_list is None')
            cop_df_dict[g1].index = [l+'_'+r for l,r in lig_rec_pair_list]
            cop_df_dict[g1].loc[:,'ligand'] = cop_df_dict[g1].index.str.split('_').str[0]
            cop_df_dict[g1].loc[:,'receptor'] = cop_df_dict[g1].index.str.split('_').str[1]

    return cop_df_dict


def permutation_pval(
    data_list_dict,
    umi_sums,
    lig_rec_pair_list,
    lig_rec_pair_index,
    n_jobs = 2,
    verbose = 1,
    groups = None,
    n = 1000,
    copula_params = CopulaParams(),
    opt_params = OptParams()
):
    """
        This function adds p-values to the copula coefficients
        Parameters:
        -----------
        data_list_dict: dict
            The dictionary of dataframes containing the count data
            for each group So a dictionary of lists
        umi_sums: dict
        lig_rec_pair_list: list
            The permutation will be run for this list of ligand receptor pairs
        lig_rec_pair_index: list
            This has the list of ligand receptor pairs, this is relevant when
            we have short vs long range itneractions
        n_jobs: int
            The number of jobs to run in parallel
        verbose: int
            The verbosity level
        groups: list
            The list of groups to run the permutation test
        n: int (default 1000)
            The number of permutations
        copula_params: namedtuple of CopulaParams
        opt_params: namedtuple of OptParams
        TODO - Add more documentation
    """
    if groups is None:
        groups = data_list_dict.keys()
    import time
    # Assume that the data has the exact size of ligand receptor pairs
    bg_coeffs_dict = {}
    for gpair in groups:
        print(gpair)
        data_list = data_list_dict[gpair]
        umi_sums_1 = umi_sums[gpair][gpair.split('=')[0]]
        umi_sums_2 = umi_sums[gpair][gpair.split('=')[1]]
        bg_group_dict = {}
        do_for_all = False
        if len(lig_rec_pair_list) == len(lig_rec_pair_index):
            do_for_all = True
        if do_for_all:
            raise ValueError('Not implemented')
        else:
            # loop over each ligand receptor
            for lig_rec_pair in lig_rec_pair_list:
                ind_array = np.array(lig_rec_pair_index)
                if np.where(ind_array == lig_rec_pair)[0].shape[0] == 0:
                    continue
                index = np.where(ind_array == lig_rec_pair)[0][0]
                x, y = data_list[index]
                # Get permuted index
                perm_data_list = []
                for _ in range(n):
                    perm_index = np.random.permutation(len(x))
                    perm_data_list += [(x[perm_index], umi_sums_1[perm_index])]
                perm_copula_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                    delayed(call_optimizer)(
                        x,
                        y,
                        us1,
                        umi_sums_2,
                        copula_params,
                        opt_params
                    ) for x, us1 in perm_data_list
                )
                copula_coeffs = np.array([res[0] for res in perm_copula_results])
                bg_group_dict[lig_rec_pair] = copula_coeffs
        bg_coeffs_dict[gpair] = bg_group_dict.copy()
    return bg_coeffs_dict


def graph_permutation_pval(
    count_df,
    int_edges,
    lig_rec_index,
    lig_df,
    rec_df,
    groups,
    close_group=None,
    heteromeric = True,
    n = 1000,
    summarization='sum',
    n_jobs = 2,
    verbose = 1,
    copula_params = CopulaParams(),
    opt_params = OptParams(),
    **kwargs
):
    """
        This function adds p-values to the copula coefficients
        Parameters:
        -----------
        data_list_dict: dict
            The dictionary of dataframes containing the count data
            for each group So a dictionary of lists
        umi_sums: dict
        TODO - Add more documentation
    """
    # Assume that the data has the exact size of ligand receptor pairs
    bg_coeffs_dict = {}
    start_time = time.time()
    do_checkpoint = kwargs.get('do_checkpoint', False)
    if do_checkpoint:
        checkpoint_dir = kwargs.get('checkpoint_dir', '.')
        checkpoint_interval = kwargs.get('checkpoint_interval', 50)
        checkpoint_counter = 0
    for gpair in groups:
        # Create a the subgraph
        # for this interaction
        G = nx.Graph()
        G.add_weighted_edges_from(
            int_edges.loc[int_edges.interaction == gpair,
                ['cell1', 'cell2', 'distance']
            ].to_records(index=False)
        )
        self_group = False
        if gpair.split('=')[0] == gpair.split('=')[1]:
            self_group = True
        connection_df_list = []
        umi_sum_1_list = []
        umi_sum_2_list = []
        print(f"Constructing {n} random graphs with {len(G.nodes())} nodes and {len(G.edges())} edges",
              flush=True)
        for _ in range(n):
            nodes = list(G.nodes())
            np.random.shuffle(nodes)
            node_mapping = dict(zip(G.nodes(), nodes))
            G_perm = nx.relabel_nodes(G, node_mapping)
            # Make a datalist for each ligand receptor pair
            connection_df = pd.DataFrame(
                 [(u, v, d['weight']) for u, v, d in G_perm.edges(data=True)],
                 columns = ['cell1', 'cell2', 'distance']
            )
            connection_df_list += [connection_df.copy()]
            umi_sum_1_list += [count_df.loc[connection_df.cell1, :].sum(1).values]
            umi_sum_2_list += [count_df.loc[connection_df.cell2, :].sum(1).values]
        print(f"Time taken to construct {n} random graphs: {time.time() - start_time:.2f} seconds",
              flush=True)
        # Now create data_list with for each ligand receptor pair
        bg_group_dict = {}
        for index in tqdm.tqdm(lig_rec_index):
            if close_group is not None:
                if (index in close_group) and (not self_group):
                    continue
            lig = lig_df.loc[index].values.tolist()
            rec = rec_df.loc[index].values.tolist()
            data_lig_rec = []
            # record the permutations
            for i in range(n):
                connection_df = connection_df_list[i]
                if heteromeric:
                    data_lig_rec += [
                        spatial.heteromeric_subunit_summarization(
                            count_df,
                            int_edges=connection_df,
                            lig=lig,
                            rec=rec,
                            summarization=summarization
                        )
                    ]
                else:
                    lig, rec = index.split('_')
                    data_lig_rec += [(
                        count_df.loc[connection_df.cell1, lig].values,
                        count_df.loc[connection_df.cell2, rec].values
                    )]
            # Now estimate copula coeff
            perm_copula_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(call_optimizer)(
                    data_lig_rec[i][0],
                    data_lig_rec[i][1],
                    umi_sum_1_list[i],
                    umi_sum_2_list[i],
                    copula_params,
                    opt_params,
                    **kwargs
                ) for i in range(n)
            )
            copula_coeffs = np.array([res[0] for res in perm_copula_results if res[3] == 'copula'])
            bg_group_dict[index] = copula_coeffs.copy()
        bg_coeffs_dict[gpair] = bg_group_dict.copy()
    return bg_coeffs_dict


# Other methods
# SCC
def spatial_cross_correlation(
    x,
    y,
    weight
):
    # scale weights
    """
        This function calculates the spatial cross correlation
        between two variables.
        Parameters:
        -----------
        x: array-like
            The first variable
        y: array-like
            The second variable
        weight: array-like
            The weight matrix
    """
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
    return SCC


def spatial_corr_test(
    x,
    y,
    weight,
    n = 500
):
    """
        This function calculates the p-value for the spatial cross correlation
        between two variables.
        Parameters:
        -----------
        x: array-like
            The first variable
        y: array-like
            The second variable
        weight: array-like
            The weight matrix
        n: int
            The number of permutations
    """
    I = spatial_cross_correlation(x,y,weight)
    xbg_corr_tets = []
    for _ in range(n):
        xbg = x.copy()
        np.random.shuffle(xbg)
        xbg_corr_tets += [spatial_cross_correlation(xbg, y, weight)]
    xbg_corr_tets.append(I)
    bg = np.array(xbg_corr_tets)
    pvalue = np.sum(abs(bg) >= abs(I)) / (n+1)
    return list([I, pvalue])


def scc_caller(
        x,
        y,
        weight,
        add_pval=False,
        n = 500,
):
    """
    This function is called from multiple threads
    calculate SCC and add p-value if required
    Parameters:
    -----------
    x: array-like
        The first variable
    TODO - Add more documentation
    """
    if add_pval:
        return spatial_corr_test(x, y, weight, n)
    return spatial_cross_correlation(x, y, weight)


def run_scc(
    adata,
    lig_rec_info_df,
    int_edges,
    groups,
    heteronomic = True,
    lig_df = None,
    rec_df = None,
    summarization = 'sum',
    n_jobs = 2,
    verbose = 1,
    distance_aware = False,
    seperate_lig_rec_type = True,
    data_type = 'visium',
    add_pval = False,
    n = 500,
    use_spatialdm = False,
    global_norm = False,
    add_self_loops = False
):
    """
    Parameters:
    -----------
    count_df: pd.DataFrame
        The count matrix
    lig_rec_info_df: pd.DataFrame
    int_edges: pd.DataFrame
        The edge dataframe
    groups: list
        The list of groups
    heteronomic: bool
        Whether to run for heteromeric interactions
    lig_df: pd.DataFrame
        The ligand dataframe (heteronomic only)
    rec_df: pd.DataFrame
        The receptor dataframe (heteronomic only)
    summarization: str
        The summarization method
    n_jobs: int
        The number of jobs to run in parallel
    verbose: int
        The verbosity level for Parallel
    distance_aware: bool
        Whether to consider distance (not implemented yet)
    seperate_lig_rec_type: bool
        Whether to consider ligand receptor type
    data_type: str
        The data type e.g. visium
    add_pval: bool
        Whether to add p-value
    n: int
        The number of permutations
    TODO - Add more documentation
    """
    if global_norm:
        if use_spatialdm:
            sc.pp.normalize_total(adata, target_sum=None)
            sc.pp.log1p(adata["X"])
            count_df_norm_log = adata.to_df()
        else:
            count_df = adata.to_df()
            count_df_norm = count_df.div(count_df.sum(1), axis = 0) * 1e6
            count_df_norm_log = np.log( count_df_norm + 1 )

    if groups is None:
        groups = int_edges.interaction.unique().tolist()

    scc_dict = {}
    for gpair in groups:
        print(gpair)
        g11, g12 = gpair.split('=')
        G = nx.Graph()
        # Ignore the distance for now
        if not distance_aware:
            # if this is across cell-types then remove self loops
            # if exists !
            # TODO should not exist
            connection_df = int_edges.loc[int_edges.interaction == gpair,
                ['cell1', 'cell2']
            ]
            if g11 != g12:
                connection_df = connection_df.loc[
                    connection_df.cell1 != connection_df.cell2
                ]
            G.add_edges_from(
                int_edges.loc[int_edges.interaction == gpair,
                    ['cell1', 'cell2']
                ].to_records(index=False)
            )
            # If this is a self interaction make sure to add self loops
            # let's add them again even if they should be added
            if g11 == g12 and add_self_loops:
                G.add_edges_from(
                    [(i, i) for i in connection_df.cell1]
                )
        else:
            raise ValueError('Not implemented')
        adata_gpair = adata[list(G.nodes()),].copy()
        if not global_norm:
            if use_spatialdm:
                sc.pp.normalize_total(adata_gpair, target_sum=None)
                sc.pp.log1p(adata_gpair)
                count_df_norm_log = adata_gpair.to_df()
            else:
                count_df = adata_gpair.to_df()
                count_df_norm = count_df.div(count_df.sum(1), axis = 0) * 1e6
                count_df_norm_log = np.log( count_df_norm + 1 )
        weight = nx.adjacency_matrix(G).todense()
        data_list = []
        lig_rec_list = []
        if heteronomic:
            for index, row in lig_rec_info_df.iterrows():
                lig = lig_df.loc[index].values.tolist()
                rec = rec_df.loc[index].values.tolist()
                if seperate_lig_rec_type:
                    if row.annotation == 'Cell-Cell Contact' and \
                    data_type == 'visium' and \
                    g11 != g12:
                        continue
                if use_spatialdm:
                    data_list += [
                        spatial.heteromeric_subunit_summarization(
                            count_df_norm_log,
                            G=G,
                            lig=lig,
                            rec=rec,
                            summarization=summarization
                        )
                    ]
                else:
                    data_list += [
                        spatial.heteromeric_subunit_summarization(
                            count_df_norm_log,
                            G=G,
                            lig=lig,
                            rec=rec,
                            summarization=summarization
                        )
                    ]
                lig_rec_list += [index]

            res = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(scc_caller)(
                    x,
                    y,
                    weight,
                    add_pval=add_pval,
                    n=n
                ) for x, y in data_list)
            # res = []
            # for (x,y) in data_list:
            #     res += [scc_caller(x, y, weight, add_pval=add_pval, n=n)]

            if add_pval:
                res_df = pd.DataFrame(np.array(res), columns=['scc', 'scc_pval'],
                                      index=lig_rec_list
                        )
            else:
                res_df = pd.DataFrame(res, columns=['scc'], index=lig_rec_list)
            scc_dict[gpair] = res_df.copy()
        else:
            raise ValueError('Not implemented')
    return scc_dict