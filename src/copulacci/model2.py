# pylint: disable=C0103, C0114, C0301
from collections import namedtuple
import warnings
import numpy as np
import pandas as pd
from scipy import stats, linalg
from scipy.optimize import minimize
from joblib import Parallel, delayed


EPSILON = 1.1920929e-07
CopulaParams = namedtuple('CopulaParams', ['perm', 'DT', 'dist_list',
                                           'copula_mode' , 'return_sum', 'rho_one_start']
            )
CopulaParams.__new__.__defaults__ = (1, False, None, 'vanilla', True, 0.01)
OptParams = namedtuple('OptParams', ['method', 'tol', 'maxiter', 'num_starts'])
OptParams.__new__.__defaults__ = ('Nelder-Mead', 1e-6, 1e+6, 1)


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
    length_cutoff = kwargs.get('length_cutoff', 50)
    use_zero_pair_cutoff = kwargs.get('use_zero_pair_cutoff', False)
    zero_pair_cutoff = kwargs.get('zero_pair_cutoff', 0.6)
    use_mu_cutoff = kwargs.get('use_mu_cutoff', False)
    mu_cutoff = kwargs.get('mu_cutoff', -8)
    force_opt = kwargs.get('force_opt', False)

    # If either of the two variables is empty return with status empty
    if (len(x) == 0) | (len(y) == 0):
        if copula_mode == 'vanilla':
            return [0, 0, 0, 'empty']
        else:
            return [0, 0, 0, 0, 'empty']

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

    if use_mu_cutoff and (skip_opt is False):
        if (mu_x_start < mu_cutoff) | (mu_y_start < mu_cutoff):
            skip_opt = True
            opt_status = 'skip_mu_cutoff'

    if skip_opt is True:
        if copula_mode == 'vanilla':
            return [0, mu_x_start, mu_y_start, opt_status]
        return [0, 0, mu_x_start, mu_y_start, opt_status]

    # Get optimization parameters
    method = opt_params.method
    tol = opt_params.tol
    num_starts = opt_params.num_starts

    # Get likelihood parameters
    dist_list = copula_params.dist_list
    if (copula_mode == 'dist' and dist_list is None):
        raise ValueError('dist_list must be provided for mode dist')

    if num_starts > 1:
        # Take uniform random samples from the parameter space
        # as different starting points for the optimization
        if copula_mode == 'vanilla':
            coeff_start = np.random.uniform(-0.99, 0.99, num_starts)
            mu_1_start = np.random.uniform(mu_x_start - 5, 0, num_starts)
            mu_2_start = np.random.uniform(mu_y_start - 5, 0, num_starts)
            start_params = np.column_stack([coeff_start, mu_1_start, mu_2_start])
            results = []
            for i in range(num_starts):
                res = minimize(
                    log_joint_lik,
                    x0 = start_params[i],
                    method=method,
                    bounds=[(-0.99, 0.99), (mu_x_start-5, 0), (mu_y_start - 5, 0)],
                    args=(x, y, umi_sum_1, umi_sum_2, copula_params,),
                    tol=tol
                )
                results += [res.copy()]
            best_result = min(results, key=lambda x: x['fun'])
            return list(best_result['x']) + [opt_status]
        else:
            rho_zero_start = np.random.uniform(-0.99, 0.99, num_starts)
            rho_one_start = np.random.uniform(0.0, 10.0, num_starts)
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
            best_result = min(results, key=lambda x: x['fun'])
            return list(best_result['x']) + [opt_status]
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
            return list(res['x']) + [opt_status]
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
            return list(res['x']) + [opt_status]


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
        print(res[10])
        if copula_params.copula_mode == 'vanilla':
            cop_df_dict[g1] = pd.DataFrame(res, columns=[
                    'copula_coeff',
                    'mu_x',
                    'mu_y',
                    'copula_method'
            ])
        else:
            cop_df_dict[g1] = pd.DataFrame(res, columns=[
                    'rho_zero',
                    'rho_one',
                    'mu_x',
                    'mu_y',
                    'copula_method'
            ])
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
