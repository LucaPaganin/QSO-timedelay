import numpy as np
import pandas as pd
from pathlib import Path
import os


class ParameterError(Exception):
    pass


class DataGenerator:
    """This is a class which generates datasets of artificial light curves data

    :param ts: total observation time, in unit of delta
    :type ts: float
    :param delta: value of the time delay, unit is days
    :type delta: float
    :param tmax: total observation time, obtained from the product between ``ts`` and ``delta``
    :type tmax: float
    :param samples_per_delta: value of the regular sampling frequency, in units of 1/delta
    :type samples_per_delta: float
    :param image_ratio: ratio between the two image signals
    :type image_ratio: float
    :param seed: seed of numpy random generator
    :type seed: float
    :param n_gaps: number of gaps to simulate in artificial data
    :type n_gaps: int
    :param gap_size: size of simulated gaps, positive integer. It represents the number of missing samples for
                     each gap
    :type gap_size: int
    :param noise_level: relative size of error bars wrt to signal
    :type noise_level: float
    :param image_ratio: multiplicative factor between the two light curves
    :type image_ratio: float
    """
    def __init__(self,
                 ts=None,
                 delta=None,
                 samples_per_delta=None,
                 image_ratio=1. / 1.44,
                 seed=0,
                 n_gaps=None,
                 gap_size=None,
                 noise_level=None):

        if ts is None:
            raise ParameterError("Total observational time cannot be None")
        self.ts = ts
        if delta is None:
            raise ParameterError("Time delay cannot be None")
        self.delta = delta
        if samples_per_delta is None:
            raise ParameterError("Sampling frequency cannot be None")
        self.samples_per_delta = samples_per_delta
        if noise_level is None or noise_level < 0:
            raise ParameterError("Noise level cannot be none or negative")
        self.noise_level = noise_level
        if image_ratio is None or image_ratio < 0:
            raise ParameterError("Image ratio cannot be None or negative!")
        self.image_ratio = image_ratio

        np.random.seed(seed)
        self.tmax = self.ts * self.delta
        self.t_regular = np.linspace(0, self.tmax, num=int(self.ts * self.samples_per_delta))
        self.t_domain = self.t_regular.copy()
        self.n_gaps = n_gaps
        self.gap_size = gap_size

        if self.n_gaps is not None:
            if self.n_gaps < 0 or int(self.n_gaps) != self.n_gaps:
                raise ParameterError(f"Number of gaps parameter must be a positive integer, not {self.n_gaps}")
        if self.gap_size is not None:
            if self.gap_size < 0 or int(self.gap_size) != self.gap_size:
                raise ParameterError(f"Gap size parameter must be a positive integer, not {self.gap_size}")
        self.gaussian_means = None
        self.gaussian_sigmas = None
        self.gaussian_peak_heights = None

    def generate_gaussian_signal_parameters(self):
        means = np.random.random(20) * self.tmax
        sigmas = np.random.random(20) * (self.tmax / 4)
        # heights = 1 + 5 * np.random.random(20)
        heights = np.ones(means.shape)
        self.gaussian_means = means.copy()
        self.gaussian_sigmas = sigmas.copy()
        self.gaussian_peak_heights = heights.copy()
        return means, sigmas, heights

    def generate_noisy_time_domain(self):
        sampling_noise = (np.random.random(len(self.t_regular)) - 0.5) * (self.delta / self.samples_per_delta)
        t_domain = self.t_regular + sampling_noise
        self.t_domain = t_domain.copy()
        return t_domain

    @staticmethod
    def gaussian_signal(t_domain, means, sigmas, heights):
        if not (means.shape == sigmas.shape and means.shape == heights.shape):
            raise ParameterError("means, sigmas and heights arrays must have the same shape!")

        signal = np.zeros(t_domain.shape)
        for mean, sigma, height in zip(means, sigmas, heights):
            signal += height * np.exp(-(t_domain - mean) ** 2 / (2 * sigma ** 2))
        return signal

    @staticmethod
    def get_dataset_filepath(fname=None, noise_level=None, gap_size=None, realization_id=None, outdir=None):
        dst_dir = outdir / f"{fname}" / f"gap_{gap_size}" / f"noise_{noise_level:.2f}"
        filename = f"DS-500_{fname}_noise{noise_level}_gapsize_{gap_size}_{realization_id}.csv"
        return dst_dir / filename

    def get_supersampled_timedomain(self):
        t_supersampled = np.linspace(0, self.tmax, num=1000)
        return t_supersampled

    def get_underlying_signal(self, t, image=None, means=None, sigmas=None, heights=None):
        means = means if means is not None else self.gaussian_means
        sigmas = sigmas if sigmas is not None else self.gaussian_sigmas
        heights = heights if heights is not None else self.gaussian_peak_heights

        if image is None:
            raise ParameterError("image string cannot be None")
        if image == "A":
            underlying_function = self.gaussian_signal(t, means, sigmas, heights)
        elif image == "B":
            underlying_function = self.image_ratio * self.gaussian_signal(t + self.delta,
                                                                          means,
                                                                          sigmas,
                                                                          heights)
        else:
            raise ParameterError(f"Not recognized image option {image}")

        return underlying_function

    def generate_gap_mask(self, gap_size=None):
        if gap_size is None or gap_size < 0:
            raise ParameterError(f"Invalid gap_size value {gap_size}")

        possible_gap_centers = np.arange(gap_size - 1,
                                         len(self.t_domain) - gap_size,
                                         step=gap_size + 1,
                                         dtype=int)
        gap_centers = np.random.choice(possible_gap_centers, size=self.n_gaps, replace=False)

        mask = np.ones(len(self.t_domain), dtype=bool)
        for c in gap_centers:
            # Compute left and right indices around gap
            a = c - int(gap_size / 2) + (1 - gap_size % 2)
            b = c + int(gap_size / 2) + 1
            mask[a:b] = False

        return mask

    def generate_light_curves(self, mask=None, means=None, sigmas=None, heights=None, noise_level=None):
        """Method to generate a masked time domain and two light curves sampled over it

        :param mask: numpy boolean array for masking
        :type mask: numpy.ndarray
        :param means: numpy array containing means of gaussians to be superimposed
        :type means: numpy.ndarray
        :param sigmas: numpy array containing sigmas of gaussians to be superimposed
        :type sigmas: numpy.ndarray
        :param heights: numpy array containing peak heights of gaussians to be superimposed
        :type sigmas: numpy.ndarray
        :param noise_level: float representing the fraction of the flux used to compute the sigma of the gaussian
                            noise
        :type noise_level: float
        :return: a tuple containing time_domain, signal a, signal b, errors on a, errors on b.
                 The time domain is masked with the boolean mask parameter if it is provided.
        :rtype: tuple of numpy.ndarray
        """
        time_domain = self.t_domain
        if mask is not None:
            time_domain = self.t_domain[mask]
        if means is None or sigmas is None:
            raise ParameterError("Gaussian means or sigmas cannot be None")
        if noise_level is None or noise_level < 0:
            raise ParameterError("Noise level cannot be None or negative")

        a = self.get_underlying_signal(time_domain, image="A", means=means, sigmas=sigmas, heights=heights)
        b = self.get_underlying_signal(time_domain, image="B", means=means, sigmas=sigmas, heights=heights)
        sigma_a = np.abs(noise_level * a)
        sigma_b = np.abs(noise_level * b)
        noise_a = np.random.normal(0, sigma_a)
        noise_b = np.random.normal(0, sigma_b)
        a += noise_a
        b += noise_b

        return time_domain, a, b, sigma_a, sigma_b

    @staticmethod
    def get_num_realizations(gap_size=None, noise_level=None):
        if noise_level == 0:
            if gap_size == 0:
                n = 1
            else:
                n = 10
        else:
            if gap_size == 0:
                n = 50
            else:
                n = 500
        return n

    def generate_single_realization_dataset(self, means=None, sigmas=None, heights=None, noise_level=None, gap_size=None):
        mask = self.generate_gap_mask(gap_size=gap_size)
        *data, = self.generate_light_curves(mask=mask, means=means, sigmas=sigmas, heights=heights, noise_level=noise_level)
        columns = ["time", "A", "B", "sigmaA", "sigmaB"]
        df = pd.DataFrame(data=np.array(data).T, columns=columns)
        return df

    def generate_single_waveform_dataset(self, means=None, sigmas=None, heights=None, fname=None, outdir=None):
        if any([x is None for x in locals().values()]):
            raise ParameterError("One of the parameters is None")

        if not isinstance(outdir, Path):
            try:
                outdir = Path(outdir)
            except:
                raise Exception(f"Cannot construct pathlib.Path from parameter {outdir}")

        noise_levels = [0.00, 0.01, 0.02, 0.03]
        gap_sizes = [0, 1, 2, 3, 4, 5]

        for gap_size in gap_sizes:
            for noise_level in noise_levels:
                n = self.get_num_realizations(gap_size=gap_size, noise_level=noise_level)
                for k in range(n):
                    df = self.generate_single_realization_dataset(means=means,
                                                                  sigmas=sigmas,
                                                                  heights=heights,
                                                                  noise_level=noise_level,
                                                                  gap_size=gap_size)
                    file_path = self.get_dataset_filepath(outdir=outdir,
                                                          fname=fname,
                                                          gap_size=gap_size,
                                                          noise_level=noise_level,
                                                          realization_id=k+1)
                    dir_path = file_path.parent
                    dir_path.mkdir(parents=True, exist_ok=True)
                    df.to_csv(file_path)
                    del df
