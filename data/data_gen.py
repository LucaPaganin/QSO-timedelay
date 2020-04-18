import numpy as np

def get_irregular_time_domain(ts=None, delta=None, sampling_freq=None):
    """!
    Function to generate an irregularly sampled time domain.
    First a regularly spaced time domain is generated, then at each point
    a uniform random number in (-0.5*z, 0.5*z) is added, where z = delta/sampling_freq.
    
    @param ts: upper limit, in units of delta
    
    @param delta: characteristic time, unit is days
    
    @param sampling_freq: sampling frequency, unit is 1/delta
    """
    if any([x is None for x in [ts,delta,sampling_freq]]):
        raise Exception("one of the parameters is None")
        
    t_regular = np.linspace(0, ts*delta, num=ts*sampling_freq)
    sampling_noise = (np.random.random(len(t_regular)) - 0.49) * (delta/sampling_freq) 
    t = t_regular + sampling_noise
    return t

def get_signal_pars(tmax):
    """!
    This function computes the means and the sigmas of the gaussian
    to superimpose in order to obtain the underlying signal
    
    @param tmax: end of time domain
    """
    means = np.random.random(20) * tmax
    sigmas = np.random.random(20) * (tmax/4)
    return means,sigmas

def basic_signal(t, means, sigmas):
    """!
    This function computes the functional form over the given time domain
    
    @param t: time domain, numpy array
    
    @param means: means of the gaussian functions, numpy array
    
    @param sigmas: sigmas of the gaussian functions, numpy array
    """
    if means.shape != sigmas.shape:
        raise Exception("means and sigmas arrays must have the same shape!")
    
    f = np.zeros(t.shape)
    for mean,sigma in zip(means, sigmas):
        f += np.exp(-(t-mean)**2/(2*sigma**2))
    return f

def generate_lightcurves(ts=None, delta=None, sampling_freq=None, means=None, sigmas=None):
    """!
    Function to generate two light curves, delayed wrt each other.
    
    @param ts: end of time domain, in units of delta
    
    @param delta: characteristic time, unit is days
    
    @param sampling_freq: sampling frequency, unit is 1/delta
    
    @param means: means of the gaussian functions, numpy array
    
    @param sigmas: sigmas of the gaussian functions, numpy array
    """
    
    if any([x is None for x in [ts,delta,sampling_freq,means,sigmas]]):
        raise Exception("one of the parameters is None")
        
    tmax = ts*delta
    t = get_irregular_time_domain(ts=ts, delta=delta, sampling_freq=sampling_freq)
    
    delay = delta
    a = basic_signal(t, means, sigmas)
    b = basic_signal(t + delay, means, sigmas)
    
    return t, a, b, delay

    
    