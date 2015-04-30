# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("/home/kerem/build/pyhsmm/examples/example-data.txt")
plt.plot(data)

# <codecell>

from pyhsmm.util.plot import pca_project_data

plt.plot(pca_project_data(data)[:, 0])

# <codecell>

import pyhsmm
import pyhsmm.basic.distributions as distributions

obs_dim = 2
Nmax = 25

obs_hypparams = {'mu_0': np.zeros(obs_dim),
                 'sigma_0': np.eye(obs_dim),
                 'kappa_0': 0.3,
                 'nu_0': obs_dim + 5}
dur_hypparams = {'alpha_0': 2 * 30,
                 'beta_0': 2}

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
    alpha=6., gamma=6.,  # better to sample over these; see concentration-resampling.py
    init_state_concentration=6.,  # pretty inconsequential
    obs_distns=obs_distns,
    dur_distns=dur_distns)

# <codecell>

posteriormodel.add_data(data, trunc=60)

# <codecell>

from copy import deepcopy

models = []
for idx in range(150):
    posteriormodel.resample_model()
    if (idx + 1) % 10 == 0:
        models.append(deepcopy(posteriormodel))

# <codecell>

fig = plt.figure()
for idx, model in enumerate(models):
    plt.clf()
    model.plot()
    plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10 * (idx + 1)))
    plt.savefig('iter_%.3d.png' % (10 * (idx + 1)))

# <codecell>

files = !ls *.png
for f in files:
    open(f)

