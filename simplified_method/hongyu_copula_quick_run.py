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
import pickle

xen_data = sc.read_h5ad("/n/fs/ragr-data/users/hz7140/external/0111-copulacci/xenium.h5ad")
cod_data = sc.read_h5ad("/n/fs/ragr-data/users/hz7140/external/0111-copulacci/codex.h5ad")

import standalone_copula
import copula_pois_norm as cpn

copula_params = standalone_copula.CopulaParams()
opt_params = standalone_copula.OptParams()
xen_array = xen_data.to_df().sum(1).values
cod_array = cod_data.to_df().sum(1).values
xen_counts = xen_data.to_df().values
cod_counts = cod_data.to_df().values

xen_gene_count = xen_counts.shape[1]
cod_gene_count = cod_counts.shape[1]
data_pairs = [(xen_counts[:, i], cod_counts[:, j], xen_array, cod_array) for i in range(xen_gene_count) for j in range(cod_gene_count)]


# opt_res_quick = Parallel(n_jobs=20, verbose=1)(
#     delayed(standalone_copula.call_optimizer)(
#         x,
#         y,
#         xen_array,
#         cod_array,
#         copula_params,
#         opt_params,
#         use_zero_cutoff=True,
#         stability_filter=True,
#         quick=True,
#         run_find_peaks = True,
#         zero_cutoff=0.8
#     ) for (x, y, xen_array, cod_array) in data_pairs)

opt_res_quick = Parallel(n_jobs=20, verbose=1)(
    delayed(cpn.quick_find)(
        x,
        y,
        skip_local_min=True
    ) for (x, y, _, _) in data_pairs)

opt_res_df = pd.DataFrame(opt_res_quick, columns=['cop_coeff', 'cop_status', 'ps'])
with open('/n/fs/ragr-data/users/hirak/Workspace/hongyu_analysis/HZ_copula_norm.pkl', 'wb') as fp:
    pickle.dump(opt_res_df, fp)
