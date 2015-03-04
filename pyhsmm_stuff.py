# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import urllib2
import numpy as np
data = urllib2.urlopen("https://raw.githubusercontent.com/mattjj/pyhsmm/master/examples/example-data.txt")
data = np.loadtxt(data)
# plt.plot(data[:,0],data[:,1],'kx')

# <codecell>

%matplotlib inline
from pyhsmm.util.plot import pca_project_data, pca, project_data
component = pca(data,2)
plt.plot(np.dot(data, component.T)[:,0])
# pca_project_data(data,1)

# <codecell>

import pyhsmm
import pyhsmm.basic.distributions as distributions

obs_dim = 2
Nmax = 25

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}
dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]


# <codecell>

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
posteriormodel.add_data(data,trunc=60)
from copy import deepcopy
models = []
for idx in range(150):
    posteriormodel.resample_model()
    print "Resample #%i done" % idx
    if (idx+1) % 50 == 0:
        models.append(deepcopy(posteriormodel))

# <codecell>

%matplotlib qt
for idx, model in enumerate(models):
    print "**************"
    fig = plt.figure()
    plt.clf()
    model.plot()
    print model.states_list[0].trans_matrix
    plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (50*(idx+1)))
#     plt.gcf().suptitle('HDP-HSMM sampled after 150 iterations')

# <codecell>


# <codecell>


