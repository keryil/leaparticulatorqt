
# coding: utf-8

# In[1]:

import ghmm
import numpy as np
import pandas as pd
from random import random, randint, choice
import itertools
from sklearn.cluster import KMeans


# In[2]:

# domain = ghmm.Float()

class GHMMWrapper(object):
    """
    This wrapper exists solely to enable pickling.
    """
    hmm = None
    def __get_state__():
        pass
    def __set_state__():
        pass

def get_range_of_multiple_traj(data):
    """
    Use this for raw data, not SequenceSets and stuff.
    Also, returns integers for more efficient search of
    state space, as this is used to initialize means of 
    states.
    """
    if hasattr(data[0][0], "__iter__"):
        xs = []
        ys = []
        for seq in data:
            for x,y in seq:
                xs.append(x)
                ys.append(y)
        return ((int(min(xs)),int(max(xs))),(int(min(ys)),int(max(ys))))
    else:
        xs = []
        for seq in data:
            xs.extend(seq)
        return ((int(min(xs)),int(max(xs))),(int(min(xs)),int(max(xs))))

def flatten_to_emission(arr, domain=ghmm.Float()):
    arr = np.array(arr).flatten().tolist()
    return ghmm.EmissionSequence(domain, arr)

def nest(arr, domain=ghmm.Float()):
    assert len(arr) % 2 == 0
    arr = np.array(arr).reshape((len(arr)/2,2))
    return arr.tolist()
    
def prob_row(n):
    r = [random() for i in range(n)]
    return  np.divide(r,sum(r))

def check_two_nonzeroes(vector):
    nonzero = 0
    for v in vector:
        if v > 0:
            nonzero += 1
            if nonzero > 1:
                return True
    return False

def nparams(hmm):
    """
    Returns the number of (free) parameters in an HMM. I copied this from the ghmm 
    C++ code where it exists but isn't exposed to SWIG. 
    """
    assert isinstance(hmm, ghmm.HMM)
    df = 0
    done_pi = False
    matrix = hmm.asMatrices()
    transition,means,initial = matrix
    for state in range(len(means)):
        # state transitions
        if check_two_nonzeroes(transition[state]):
            df += hmm.cmodel.cos * (len(means) - 1)
        # initial probs
        if initial[state] not in (0,1):
            df += 1
            done_pi = True
        # the mean and variance of the state
        df += 2
    if done_pi:
        df -= 1
    return df

def posteriorProbOfPath(hmm, obs, path):
    prob = 1.
    assert isinstance(obs, ghmm.EmissionSequence)
    posteriors = hmm.posterior(obs)
    for i,state in enumerate(path):
        prob *= posteriors[i][state]
    return prob

def posteriorProbOfPaths(hmm, observations, paths):
    prob = 1.
    assert isinstance(observations, ghmm.SequenceSet)
#     print "Paths:", paths
    return [posteriorProbOfPath(hmm, obs, path) for obs, path in zip(observations,paths)]

def _score(hmm, obs):
    """
    Calculates the common terms of BIC and AIC. 
    """
    df = nparams(hmm)
    is_multivariate = isinstance(hmm.asMatrices()[1][0][0],list)

    # we need to treat observations and observation sets 
    # separately
    n = 0
    if isinstance(obs, ghmm.EmissionSequence):
        n = len(obs)
        if is_multivariate:
            n = n / 2
#         print "Found N as %f" % n
    else:
        for seq in obs:
            nn = len(seq)
            if is_multivariate:
                nn = nn / 2
            n += nn
#         print "Found N as %f" % n
    llg = hmm.loglikelihood(obs)
    return llg, df, n
    
def bic(hmm, obs):
    llg, df, n = _score(hmm,obs)
    return -2 * llg + df * np.log(n)

def aic(hmm, obs):
    llg, df, n = _score(hmm,obs)
    return -2 * llg + df * 2

class PickleableSWIG(object):

    def __setstate__(self, state):
        self.__init__(*state['args'])

    def __getstate__(self):
        return {'args': self.args}
    
class PickleableMultivariateHMM(ghmm.MultivariateGaussianDistribution, PickleableSWIG):
    def __init__(self, *args):
        self.args = args
        ghmm.MultivariateGaussianDistribution.__init__(self, *args)
        
class PickleableUnivariateHMM(ghmm.GaussianDistribution, PickleableSWIG):
    def __init__(self, *args):
        self.args = args
        ghmm.GaussianDistribution.__init__(self, *args)


# In[1]:

def train_hmm_on_set_of_obs(data, nstates, range_x, range_y=None):
    from leaparticulator.data.hmm import HMM
    domain = ghmm.Float()
    multivariate = np.asarray(data[0]).ndim == 2
    data_np = []
    for traj in data:
        for f in traj:
            data_np.append(f)
    data_np = np.asarray(data_np)
    
#     print "Is multivariate?", multivariate
    import sys;sys.stdout.flush()
    variance = means = B = dist = None
    
    if multivariate:
        obs = [flatten_to_emission(d) for d in data]
        obs = ghmm.SequenceSet(domain, obs)
        rand_obs = randint(0,len(obs)-1)
        variance = np.cov(list(obs[rand_obs])[::2],list(obs[rand_obs])[1::2]).flatten()
        means = list(KMeans(n_clusters=nstates).fit(data_np).cluster_centers_)
        B = [[list(mean), variance] for mean in means]
        dist = ghmm.MultivariateGaussianDistribution(domain)
    else:
        obs = ghmm.SequenceSet(domain, [ghmm.EmissionSequence(domain, d) for d in data])
        variance = np.var([item for seq in data for item in seq])
        means = KMeans(n_clusters=nstates).fit(data_np.reshape((len(data_np),1))).cluster_centers_
        B = [[list(mean)[0], variance] for mean in means]
        dist = ghmm.GaussianDistribution(domain)
    
    hmm_g = ghmm.HMMFromMatrices(domain, 
                         dist, 
                         A=[prob_row(nstates) for i in range(nstates)],
                         B=B,#[[randint(*range_x),variance] for i in range(nstates)],
                         pi=prob_row(nstates))
    hmm_g.baumWelch(obs)
    hmm_g.obs = obs
    return HMM(hmm_g, obs, hmm_type="ghmm")


# In[4]:

# import ghmm
# import ExperimentalData
# nstates=4
# domain=ghmm.Float()
# obs = flatten_to_emission([(random()*10,random()*10) for i in range(1000)])
# variance = np.cov(list(obs)[::2],list(obs)[1::2]).flatten()
# # print variance
# import sys;sys.stdout.flush()
# hmm = ghmm.HMMFromMatrices(domain, 
#                      ghmm.MultivariateGaussianDistribution(domain), 
#                      A=[prob_row(nstates) for i in range(nstates)],
#                      B=[[[i,i],variance] for i in range(nstates)],
#                      pi=prob_row(nstates))
# hmm.baumWelch(obs)
# hmm.obs = obs
# # print type(obs)
# hmm = ExperimentalData.HMM(hmm, obs, hmm_type="ghmm")

# # import dill
# import pickle
# import jsonpickle
# import numpy as np
# import copy_reg

# copy_reg.constructor(ExperimentalData.reconstruct_hmm)
# copy_reg.pickle(type(hmm), ExperimentalData.reduce_hmm, ExperimentalData.reconstruct_hmm)
# # print hmm.multivariate
# d = pickle.dumps(hmm)
# # print d
# hmm2 = pickle.loads(d)
# from difflib import ndiff
# print "\n".join(ndiff(str(hmm).splitlines(),str(hmm2).splitlines()))


# In[5]:

# def pickleish(hmm):
#     a,b,pi = hmm.asMatrices()
#     domain = ghmm.Float()
#     dist = ghmm.MultivariateGaussianDistribution(domain)
#     return ghmm.MultivariateGaussianMixtureHMM, (domain,dist,a,b,pi)
    
# def unpickleish(matrix,):
#     print "UNPICKLING"
#     print matrix
#     matrices = matrix
# #     print matrices[1][0][0], np.asarray(matrices[1][0][0]).ndim
#     multivariate = len(list(matrices[1][0][0])) == 2
#     print "Multivariate?", multivariate
#     domain = ghmm.Float()
#     dist = ghmm.GaussianDistribution(domain)
#     if multivariate:        
#         dist = ghmm.MultivariateGaussianDistribution(domain)
#     A,B,pi = matrices
#     model = ghmm.HMMFromMatrices(emissionDomain=domain, distribution=dist, A=A,B=B,pi=pi)
#     return model
# import copy_reg, pickle
# copy_reg.constructor(unpickleish)
# copy_reg.pickle(ghmm.MultivariateGaussianMixtureHMM, pickleish, constructor_ob=unpickleish)



# def getstate(self):
    
# def setstate(self):
#     pass
# hmm.__get_state__ = 
# dumped = pickle.dumps(hmm)
# hmm2 = pickle.loads(dumped)
# import difflib
# print '\n'.join(difflib.ndiff(str(hmm).splitlines(),str(hmm2).splitlines()))
# print hmm
# pickle.dumps
# dmp = jsonpickle.encode(hmm)
# hmm2 = ghmm.MultivariateGaussianMixtureHMM?

# print hmm2.asMatrices()
#         if transition[state]
    
# range_x=(-300,300) 
# range_y=(0,550)
# # state1 = [randint(*range_x) for i in range(100)]
# # state2 = [randint(*range_y) for i in range(100)]
# # data=state1 + state2
# data = [(randint(*range_x),randint(*range_y)) for i in range(1000)]
# data = [data, [(randint(*range_x),randint(*range_y)) for i in range(1000)]]
# hmm, obs = train_hmm_on_set_of_obs(data, nstates=2, range_x=(-300,550), range_y=range_y)
# print bic(hmm,obs), aic(hmm,obs)
# path, llg = hmm.viterbi(obs[0])
# print posteriorProbOfPath(hmm=hmm, obs=obs[0], path=path)

# data = [randint(*range_x) for i in range(1000)]
# data = [data,[randint(*range_x) for i in range(1000)]]
# hmm, obs = train_hmm_on_set_of_obs(data, nstates=2, range_x=(-300,550), range_y=range_y)
# print bic(hmm,obs), aic(hmm,obs)
# print len(hmm.sampleSingle(10))
# # print hmm.viterbi(flatten_to_emission(data[0]))
# paths = [hmm.viterbi(obs[i])[0] for i,o in enumerate(data)] 
# print posteriorProbOfPaths(hmm=hmm, observations=obs, paths=paths)
# aic(hmm, obs)


# In[6]:

# #, 100,10000)
# obs = flatten_to_emission(state2)
# print obs
# path,ll = hmm.viterbi(obs)
# print path
# print len(path), len(obs)
# posteriors = hmm.posterior(obs)
# prob = 1.
# for i,state in enumerate(path):
#     prob *= posteriors[i][state]
# print prob
# hmm.getTransition?

