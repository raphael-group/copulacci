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
from scipy.stats import poisson, norm

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


def quick_find_vectorized(
        x,
        y,
        DT=False,
        c=np.linspace(-0.99, 0.99, 1000),
        skip_local_min=False,
        piggy_back=True
):
    emp = np.corrcoef(x, y)[0, 1]
    if x.sum() == 0 or y.sum() == 0:
        return np.nan, False, emp

    lam1 = x.mean()
    mu_2, sigma_2 = y.mean(), y.std()

    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = (y - mu_2) / sigma_2
    z = np.column_stack([r_x, r_y])

    # Precompute term2 once
    term2_poisson = np.log(stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON)).sum()
    term2_normal = np.log(norm.pdf(y, mu_2, sigma_2).clip(EPSILON, 1 - EPSILON)).sum()
    term2 = term2_poisson + term2_normal

    # Precompute sums for term1
    sum_z0z1_sq = np.sum(z[:, 0]**2 + z[:, 1]**2)
    sum_z0z1_product = np.sum(z[:, 0] * z[:, 1])
    n = len(x)

    # Vectorized computation for all coefficients
    coeffs = c
    denom = 1 - coeffs**2
    term1 = -0.5 * (coeffs**2 * sum_z0z1_sq - 2 * coeffs * sum_z0z1_product) / denom
    term3 = -0.5 * n * np.log(denom + EPSILON)
    logsum = term1 + term2 + term3
    val = -logsum  # Negative log likelihood

    # Rest of the code remains the same
    if skip_local_min:
        peaks, _ = find_peaks(-val)
        if peaks.size == 0:
            return np.nan, False, emp
        max_peak = c[peaks[0]]
        return (max_peak, True, emp) if peaks.size == 1 else (np.nan, False, emp)
    
    y_spl = UnivariateSpline(c, val, s=0, k=4)
    y_spl_2d = y_spl.derivative(n=2)
    d2l = y_spl_2d(c)
    
    if np.min(d2l) < 0:
        return np.nan, False, emp
    else:
        peaks, _ = find_peaks(-val)
        if peaks.size == 0:
            return np.nan, False, emp
        max_peak = c[peaks[0]]
        return max_peak, True, emp


def quick_find(
        x,
        y,
        DT=False,
        c = np.linspace(-0.99, 0.99, 1000),
        skip_local_min = False,
        piggy_back = True
):
    """
    Take double derivative of the log likelihood function
    """
    emp = np.corrcoef(x,y)[0,1]
    if ((x.sum() == 0) or (y.sum() == 0)):
        return np.nan, False, emp
    
    lam1 = x.mean()
    # # mu_1 = copula_params.mu_x
    # mu_2 = copula_params.mu_y
    # sigma_2 = copula_params.sigma_y
    mu_2, sigma_2 = y.mean(), y.std()
    # get z
    r_x = get_dt_cdf(x, lam1, DT=DT)
    r_y = (y - mu_2) / sigma_2
    #r_y = get_dt_cdf(y, lam2, DT=DT)

    z = np.column_stack([r_x, r_y])
    
    term2 = (
            np.sum(np.log( stats.poisson.pmf(x, lam1).clip(EPSILON, 1 - EPSILON) )) +
            np.sum(np.log(norm.pdf(y, mu_2, sigma_2).clip(EPSILON, 1 - EPSILON)))
            #np.sum(np.log(stats.poisson.pmf(y, lam2).clip(EPSILON, 1 - EPSILON) ))
        )
    def f(coeff):
        det = 1 - coeff**2
        term1 = np.sum(-0.5 * (((coeff**2)/det) * ((z[:,0]**2) + (z[:,1] ** 2)) - 2 * (coeff/det) * z[:,0] * z[:,1]) )
        
        term3 = -0.5 * len(x) * np.log(det+EPSILON)
        logsum = term1+term2+term3
        return -logsum

    val = np.array([f(coeff) for coeff in c])
    if skip_local_min:
        peaks, _ = find_peaks(-val)
        max_peak = c[peaks[0]]
        if len(peaks) == 1:
            return max_peak, True, emp
        else:
            return np.nan, False, emp
    y_spl = UnivariateSpline(c, val,s=0,k=4)
    y_spl_2d = y_spl.derivative(n=2)
    d2l = y_spl_2d(c)
    min_point = min(d2l)
    if min_point < 0:
        return np.nan, False, emp
    else:
        # Find local minima
        peaks, _ = find_peaks(-val)
        max_peak = c[peaks[0]]
        return max_peak, True, emp