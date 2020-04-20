import numpy as np
import pandas as pd

class ParameterError(Exception):
    pass


class DataGenerator:
    def __init__(self,
                 ts=None,
                 delta=None,
                 samples_per_delta=None,
                 image_ratio=1. / 1.44,
                 seed=0,
                 n_gaps=None,
                 gap_size=None,
                 noise_amplitude=0,
                 relative_error_amplitude=None):
        """!
        Constructor of the class

        :param ts: total observational time, in unit of delta

        :param delta: value of the time delay, unit is days

        :param samples_per_delta: value of the regular sampling frequency, in units of 1/delta

        :param image_ratio: ratio between the two image signals

        :param seed: seed of numpy random generator

        :param n_gaps: number of gaps to simulate in artificial data, positive integer

        :param gap_size: size of simulated gaps, positive integer. It represents the number of missing samples for
        each gap

        :param noise_amplitude: relative size of gaussian noise sigma wrt to signal

        :param relative_error_amplitude: relative size of error bars wrt to signal
        """
        if ts is None:
            raise ParameterError("Total observational time cannot be None")
        self.ts = ts
        if delta is None:
            raise ParameterError("Time delay cannot be None")
        self.delta = delta
        if samples_per_delta is None:
            raise ParameterError("Sampling frequency cannot be None")
        self.samples_per_delta = samples_per_delta
        if relative_error_amplitude is None:
            raise ParameterError("Relative error amplitude cannot be none")
        self.relative_error_amplitude = relative_error_amplitude
        if image_ratio is None:
            raise ParameterError("Image ratio cannot be None!")
        self.image_ratio = image_ratio

        np.random.seed(seed)
        self.tmax = self.ts * self.delta
        self.original_n_samples = self.ts * self.samples_per_delta
        self.t_regular = np.linspace(0, self.tmax, num=self.original_n_samples)
        self.t_domain = self.t_regular.copy()

        self.n_gaps = n_gaps
        self.gap_size = gap_size
        self.noise_amplitude = noise_amplitude

        if self.n_gaps is not None:
            if self.n_gaps < 0 or int(self.n_gaps) != self.n_gaps:
                raise ParameterError(f"Number of gaps parameter must be a positive integer, not {self.n_gaps}")
        if self.gap_size is not None:
            if self.gap_size < 0 or int(self.gap_size) != self.gap_size:
                raise ParameterError(f"Gap size parameter must be a positive integer, not {self.gap_size}")
        if self.noise_amplitude < 0:
            raise ParameterError("Noise amplitude cannot be negative")
        if self.image_ratio is not None and self.image_ratio < 0:
            raise ParameterError("Image ratio cannot be negative")
        self.gaussian_means = None
        self.gaussian_sigmas = None

    def generate_gaussian_signal_parameters(self):
        means = np.random.random(20) * self.tmax
        sigmas = np.random.random(20) * (self.tmax / 4)
        self.gaussian_means = means.copy()
        self.gaussian_sigmas = sigmas.copy()
        return means, sigmas

    def generate_noisy_time_domain(self):
        sampling_noise = (np.random.random(len(self.t_regular)) - 0.5) * (self.delta / self.samples_per_delta)
        t_domain = self.t_regular + sampling_noise
        self.t_domain = t_domain.copy()
        return t_domain

    def set_gap_parameters(self, n_gaps, gap_size):
        self.n_gaps = n_gaps
        self.gap_size = gap_size

    @staticmethod
    def gaussian_signal(t_domain, means, sigmas):
        if means.shape != sigmas.shape:
            raise ParameterError("means and sigmas arrays must have the same shape!")

        signal = np.zeros(t_domain.shape)
        for mean, sigma in zip(means, sigmas):
            signal += np.exp(-(t_domain - mean) ** 2 / (2 * sigma ** 2))
        return signal

    def get_supersampled_timedomain(self):
        t_supersampled = np.linspace(0, self.tmax, num=1000)
        return t_supersampled
        
    def get_underlying_signal(self, t, image=None, means=None, sigmas=None):
        means = means if means is not None else self.gaussian_means
        sigmas = sigmas if sigmas is not None else self.gaussian_sigmas
        
        if image is None:
            raise ParameterError("image string cannot be None")
        if image == "A":
            underlying_function = self.gaussian_signal(t, means, sigmas)
        elif image == "B":
            underlying_function = self.image_ratio * self.gaussian_signal(t + self.delta, 
                                                                          means, 
                                                                          sigmas)
        else:
            raise ParameterError(f"Not recognized image option {image}")
        
        return underlying_function
        
    def generate_gap_mask(self):
        possible_gap_centers = np.arange(self.gap_size - 1,
                                         len(self.t_domain) - self.gap_size,
                                         step=self.gap_size + 1,
                                         dtype=int)
        gap_centers = np.random.choice(possible_gap_centers, size=self.n_gaps, replace=False)

        mask = np.ones(len(self.t_domain), dtype=bool)
        for c in gap_centers:
            # Compute left and right indices around gap
            a = c - int(self.gap_size / 2) + (1 - self.gap_size % 2)
            b = c + int(self.gap_size / 2) + 1
            mask[a:b] = False

        return mask

    def generate_light_curves(self, mask=None, means=None, sigmas=None, noise_amplitude=None):
        """
        :param mask, numpy boolean array for masking

        :param means, numpy array containing means of gaussians to be superimposed

        :param sigmas, numpy array containing sigmas of gaussians to be superimposed

        :param noise_amplitude, float representing the fraction of the flux used to compute the sigma of the gaussian
        noise

        :rtype: tuple containing time_domain, signal a, signal b, errors on a, errors on b
        """
        time_domain = self.t_domain
        noise_amplitude = noise_amplitude if noise_amplitude is not None else 0
        if mask is not None:
            time_domain = self.t_domain[mask]
        if means is None or sigmas is None:
            raise ParameterError("Gaussian means or sigmas cannot be None")

        a = self.get_underlying_signal(time_domain, image="A", means=means, sigmas=sigmas)
        b = self.get_underlying_signal(time_domain, image="B", means=means, sigmas=sigmas)
        sigma_a = np.abs(self.relative_error_amplitude * a)
        sigma_b = np.abs(self.relative_error_amplitude * b)
        if noise_amplitude > 0:
            noise_a = np.random.normal(0, noise_amplitude * a)
            noise_b = np.random.normal(0, noise_amplitude * b)
            a += noise_a
            b += noise_b

        return time_domain, a, b, sigma_a, sigma_b

    def generate_dataset(self, mask=None, means=None, sigmas=None, noise_amplitude=None):
        *data, = self.generate_light_curves(mask=mask, means=means, sigmas=sigmas, noise_amplitude=noise_amplitude)
        df = pd.DataFrame(data=np.array(data).T, columns=["time", "A", "B", "SigmaA", "SigmaB"])
        return df