import numpy as np
import scipy as s
import pandas as pd
from svca.models.model1 import Model1
from svca.simulations.from_real import FromRealSimulation
from svca.models.io import *
from svca.util_functions import utils
import sys
from limix.utils.preprocess import covar_rescaling_factor_efficient
from copy import deepcopy


def run(data_dir, protein_index, output_dir, bootstrap_index,
        normalisation='standard', permute=False):
    # reading all data
    ####################################################################
    expression_file = data_dir + 'exp.txt'
    position_file = data_dir+'pos.txt'
    # protein_name, phenotype, X = pr names, pr expr values, positions
    db = pd.read_csv('/CellChatDB_db2.csv',
                     index_col=0)
    protein_names, phenotypes, X = utils.read_data(expression_file,
                                                   position_file)

  # import pdb; pdb.set_trace()
  # protein_name = protein_names[protein_index, :]
    protein_name = db.index[protein_index]
    print(protein_name)
    phenotype = phenotypes[:, (protein_names==protein_name.split("_")[0])[:,0]]
    #import pdb; pdb.set_trace()
    sel = np.arange(phenotypes.shape[1])
    sel =list(sel[pd.Series(protein_names[:,0]).isin(db.loc[protein_name,
                                                    ['R1', 'R2']].dropna().values).values])
    kin_from = phenotypes[:, sel] # remove the selected pr
    N_samples = X.shape[0]

    # permuting cells
    if permute:
        perm = np.random.permutation(X.shape[0])
        X = X[perm, :]

    # do simulations
    ####################################################################
    sim = FromRealSimulation(X, phenotype[:,0], kin_from)
    file_prefix = protein_name
    Y_sim = sim.simulate(interactions_size = None)
    np.save(output_dir+ 'nul/'+file_prefix, Y_sim)
    Y_sim = sim.simulate(interactions_size=0.25)
    np.save(output_dir+ 'rescale25/' + file_prefix, Y_sim)
    Y_sim = sim.simulate(interactions_size=0.99)
    np.save(output_dir+ 'full/' + file_prefix, Y_sim)
    Y_sim = sim.simulate(interactions_size=0.75)
    np.save(output_dir+ 'rescale75/' + file_prefix, Y_sim)
    Y_sim = sim.simulate(interactions_size=0.5)
    np.save(output_dir+ 'half/' + file_prefix, Y_sim)
    Y_sim = sim.simulate(interactions_size = 0.99)
    np.save(output_dir+ 'full/'+file_prefix, Y_sim)

    # run model on simulated data
    ####################################################################
    # intrinsic and environmental term
    ####################################################################
    cterms = ['intrinsic', 'environmental']
    model = Model1(Y_sim, X, norm=normalisation, oos_predictions=0., cov_terms=cterms, kin_from=kin_from)
    model.reset_params()
    model.train_gp(grid_size=10)

    file_prefix = protein_name[0] + '_' + str(bootstrap_index) + '_local'
    write_variance_explained(model, output_dir, file_prefix) # local_effects
    write_LL(model, output_dir, file_prefix) # LL

    int_param = model.intrinsic_cov.getParams()
    env_param = model.environmental_cov.getParams()
    noise_param = model.noise_cov.getParams()

    ####################################################################
    # add cell-cell interactions
    ####################################################################
    model.add_cov(['interactions'])

    LL = np.Inf
    for i in range(5):
        if i == 0:
            int_bk = int_param
            local_bk = env_param
            noise_bk = noise_param
            scale_interactions = True
        else:
            int_bk = int_param * s.random.uniform(0.8, 1.2, len(int_param))
            local_bk = env_param * s.random.uniform(0.8, 1.2, len(env_param))
            noise_bk = noise_param * s.random.uniform(0.8, 1.2, len(noise_param))
            scale_interactions = False
        model.set_initCovs({'intrinsic': int_bk,
                        'noise': noise_bk,
                        'environmental':local_bk})
        if scale_interactions:
            model.set_scale_down(['interactions'])
        else:
            model.use_scale_down = False

        model.reset_params()
        model.train_gp(grid_size=10)
        if model.gp.LML() < LL:
            LL = model.gp.LML()
            saved_params = model.gp.getParams()

    model.gp.setParams(saved_params)
    file_prefix = protein_name[0] + '_' + str(bootstrap_index) + '_interactions'
    write_variance_explained(model, output_dir, file_prefix)
    write_r2(model, output_dir, file_prefix)
    write_LL(model, output_dir, file_prefix)
    # write_Ks(model, output_dir, file_prefix)


    if __name__ == '__main__':
        data_dir = sys.argv[1]
        output_dir = sys.argv[2]
        protein_index = int(0)
        bootstrap_index = 1
        normalisation =  'quantile'

        run(data_dir, protein_index, output_dir, bootstrap_index, normalisation)