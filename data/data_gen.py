import numpy as np

class ParameterError(Exception):
    pass

def get_irregular_time_domain(ts=None, delta=None, sampling_freq=None):
    """!
    Function to generate an irregularly sampled time domain.
    First a regularly spaced time domain is generated, then at each point
    a uniform random number in (-0.5*z, 0.5*z) is added, where z = delta/sampling_freq.
    
    @param ts: upper limit, in units of delta
    
    @param delta: characteristic time, unit is days
    
    @param sampling_freq: sampling frequency, unit is 1/delta
    """
    if any([x is None for x in locals().values()]):
        raise ParameterError("one of the parameters is None")
        
    t_regular = np.linspace(0, ts*delta, num=ts*sampling_freq)
    sampling_noise = (np.random.random(len(t_regular)) - 0.5) * (delta/sampling_freq) 
    t = t_regular + sampling_noise
    return t

def get_signal_pars(ts=None, delta=None):
    """!
    This function computes the means and the sigmas of the gaussian
    to superimpose in order to obtain the underlying signal
    
    @param tmax: end of time domain
    """
    tmax = ts * delta
    means = np.random.random(20) * tmax
    sigmas = np.random.random(20) * (tmax/4)
    return means, sigmas

def basic_signal(t, means, sigmas):
    """!
    This function computes the functional form over the given time domain
    
    @param t: time domain, numpy array
    
    @param means: means of the gaussian functions, numpy array
    
    @param sigmas: sigmas of the gaussian functions, numpy array
    """
    if means.shape != sigmas.shape:
        raise ParameterError("means and sigmas arrays must have the same shape!")
    
    f = np.zeros(t.shape)
    for mean,sigma in zip(means, sigmas):
        f += np.exp(-(t-mean)**2/(2*sigma**2))
    return f

def get_gap_mask(t_dom=None, ngaps=None, gapsize=None):
    """!
    This function computes a boolean mask based on the irregularly sampled time domain
    placing ngaps random gaps with a given size.
    
    @param t_dom: the sampled time domain, numpy array
    
    @param ngaps: the number of gaps to place, positive integer
    
    @param gapsize: the size (in indices) of the gaps, positive integer
    """
    if any([x is None for x in locals().values()]):
        raise ParameterError("one of the parameters is None")
    
    if ngaps < 0 or ngaps != int(ngaps):
        raise ParameterError(f"Invalid ngaps parameter: {ngaps}")
    if gapsize < 0 or gapsize != int(gapsize):
        raise ParameterError(f"Invalid gapsize parameter: {gapsize}")
    
    # Compute centers of random gaps
    possible_gap_centers = np.arange(gapsize-1, len(t_dom)-gapsize, step=gapsize+1, dtype=int)
    gap_centers = np.random.choice(possible_gap_centers, size=ngaps, replace=False)
    
    mask = np.ones(len(t_dom), dtype=bool)
    for c in gap_centers:
        # Compute left and right indices around gap
        a = c - int(gapsize/2) + (1 - gapsize%2)
        b = c + int(gapsize/2) + 1
        mask[a:b] = False
    
    return mask


def generate_lightcurves(t_dom=None, delay=None, means=None, sigmas=None, ngaps=None, gapsize=None, noise=None):
    """!
    Function to generate two light curves, delayed wrt each other of delay
    A masking with ngaps random gaps of size gapsize is applied.
    Noise will be computed as noise*signal
    
    @param t_dom: time domain, numpy array
    
    @param delay: value of time delay in days
        
    @param means: means of the gaussian functions, numpy array
    
    @param sigmas: sigmas of the gaussian functions, numpy array
    
    @param ngaps: the number of gaps to place, positive integer
    
    @param gapsize: the size (in indices) of the gaps, positive integer
    
    @param noise: relative value of errors wrt to signal value
    """
    
    if any([x is None for x in locals().values()]):
        raise ParameterError("one of the parameters is None")
    
    mask = get_gap_mask(t_dom=t_dom, ngaps=ngaps, gapsize=gapsize)
    
    t = t_dom[mask]
    a = basic_signal(t, means, sigmas)
    b = basic_signal(t + delay, means, sigmas)
    err_a = np.abs(noise) * a
    err_b = np.abs(noise) * b
    
    return t, a, b, err_a, err_b
    
        
    
    
    