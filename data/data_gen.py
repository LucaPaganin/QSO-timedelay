import numpy as np

def get_time_domain(ts, delta, sampling):
    return np.linspace(0, ts*delta, num=ts*sampling)

def get_means(tmax):
    return np.random.random(20) * tmax

def get_sigmas(tmax):
    return np.random.random(20) * tmax/4

def signal(t, means, sigmas):
    f = np.zeros(t.shape)
    for mean,sigma in zip(means, sigmas):
        f += np.exp(-(t-mean)**2/(2*sigma**2))
    return f