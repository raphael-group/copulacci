import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from joblib import Parallel, delayed
import scanpy as sc
import time
from scipy import integrate
from scipy.signal import find_peaks
from collections import namedtuple
from datetime import datetime

def log(message):
    """Prints message with a timestamp."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}")


EPSILON = 1.1920929e-07
CopulaParams = namedtuple('CopulaParams', [
    'perm', 'DT', 'dist_list',
    'mu_x', 'mu_y','lam_x', 'sigma_y',
    'copula_mode' , 'return_sum',
    'rho_one_start'
    ]
)
CopulaParams.__new__.__defaults__ = (1, False, None,
                                     None, None, None,None,
                                     'vanilla', True,
                                     0.01)
OptParams = namedtuple('OptParams', ['method', 'tol', 'maxiter', 'num_starts'])
OptParams.__new__.__defaults__ = ('Nelder-Mead', 1e-6, 1e+6, 1)


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


def sample_from_normal_poisson_copula(
        n_array,
        mu_x,
        mu_y,
        sigma_y,
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
    coeff = kwargs.get('coeff', 0.0)
    lam_x = kwargs.get('lam_x',None)
    if lam_x is None:
        lam_x = n_array * np.exp(mu_x)
    cov = linalg.toeplitz([1,coeff])
    samples = np.random.multivariate_normal(
        [0,0],
        cov,
        size = len(n_array)
    )
    samples = pd.DataFrame(samples, columns=['x', 'y'])
    cdf_x = stats.norm.cdf(samples.x)
    cdf_y = stats.norm.cdf(samples.y)
    if isinstance(lam_x, np.ndarray):
        print(f"lam_x - {lam_x[0]}")
    else:
        print(f"lam_x - {lam_x}")
    samples.loc[:, 'x'] = stats.poisson.ppf(cdf_x, lam_x)
    #samples.loc[:, 'y'] = stats.poisson.ppf(cdf_y, lam_y)
    samples.loc[:, 'y'] = norm.ppf(cdf_y, loc=mu_y, scale=sigma_y) 
    return samples


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


def only_log_pois_norm_lik(
        lam1,
        mu_2,
        sigma_2,
        x,
        y,
        copula_params,
        c = np.linspace(-0.99, 0.99, 1000)
):
    """
    Take double derivative of the log likelihood function
    """
    DT = copula_params.DT
    
    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = (y - mu_2) / sigma_2
    #r_y = get_dt_cdf(y, lam2, DT=DT)

    z = np.column_stack([r_x, r_y])

    def f(coeff):
        det = 1 - coeff**2
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(norm.pdf(y, mu_2, sigma_2).clip(EPSILON, 1 - EPSILON)))
            #np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )
        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1+term2+term3
        return -logsum

    val = np.array([f(coeff) for coeff in c])

    return val


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


from scipy.stats import poisson, norm

def log_joint_lik_poisson_normal(
        params,
        x,
        y,
        umi_sum_1,
        umi_sum_2,
        copula_params
):
    '''
        This function computes the log joint likelihood of the
        copula model with one Poisson and one Normal distribution.
        Parameters:
        -----------
        params: array-like
            The parameters of the copula model contains either 3 or 5 values
            depending on the mode.
        x: array-like
            The first variable (Poisson-distributed)
        y: array-like
            The second variable (Normal-distributed)
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
        sigma_2 = params[3]  # Standard deviation for the Normal distribution
    elif mode == 'single_param':
        coeff = params[0]
        mu_1 = copula_params.mu_x
        mu_2 = copula_params.mu_y
        sigma_2 = copula_params.sigma_y
        if mu_1 is None or mu_2 is None or sigma_2 is None:
            raise ValueError('mu_1, mu_2, and sigma_2 must be provided for mode single_param.')
    else:
        rho_zero = params[0]
        rho_one = params[1]
        dist_list = copula_params.dist_list
        coeff_list = rho_zero * np.exp(-1 * dist_list * rho_one)
        mu_1 = params[2]
        mu_2 = params[3]
        sigma_2 = params[4]
    if mode == 'dist' and dist_list is None:
        raise ValueError('dist_list must be provided for mode dist')
    perm = copula_params.perm
    DT = copula_params.DT
    return_sum = copula_params.return_sum

    if copula_params.lam_x is None:
        lam1 = umi_sum_1 * np.exp(mu_1)
    else:
        lam1 = copula_params.lam_x

    # Get the distributional transformation
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = (y - mu_2) / sigma_2  # Standardized Normal variable
    if DT:
        for _ in range(perm-1):
            r_x += get_dt_cdf(x, lam1, DT=DT)
        z = np.column_stack([r_x/perm, r_y])
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
            np.sum(np.log(poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON))) +
            np.sum(np.log(norm.pdf(y, mu_2, sigma_2).clip(EPSILON, 1 - EPSILON)))
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
        np.log(poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON)) +
        np.log(norm.pdf(y, mu_2, sigma_2).clip(EPSILON, 1 - EPSILON))
    )
    term3 = -0.5 * np.log(det+EPSILON)
    return term1 + term2 + term3



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

    # Assume GLM-PC model
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