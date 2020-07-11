import numpy as np
import pandas as pd
import itertools
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from scipy import integrate
from scipy import interpolate
from scipy import signal
from scipy.optimize import curve_fit
import time
from pathlib import Path
import os
from typing import Dict, Union, Callable, Tuple, List


def power_law_sf(tau: Union[float, np.ndarray], pars) -> Union[float, np.ndarray]:
    return 10 ** pars['intercept'] * tau ** pars['slope']


def exponential_sf(tau: Union[float, np.ndarray], pars) -> Union[float, np.ndarray]:
    return pars['V0'] * (1 - np.exp(- tau / pars['dtau0']))


# noinspection PyTupleAssignmentBalance
class LightCurvePRHGenerator:
    t: np.ndarray
    qso_lc_vals: np.ndarray
    qso_lc_errs: np.ndarray
    qso_df: pd.DataFrame
    image: str
    lags_matrix: np.ndarray
    sf_lags: np.ndarray
    sf_vals: np.ndarray
    sf_fit_pars: Dict[str, float]
    sf_fit_func: Callable

    def __init__(self, image: str = 'A', sf_model: str = 'power_law', seed: float = 1234):
        self.t = None
        self.N_samples = None
        self.qso_lc_vals = None
        self.qso_lc_errs = None
        self.qso_df = None
        self.qso_id = None
        self.image = image
        self.sf_model = sf_model
        self.sf_lags = None
        self.sf_vals = None
        self.sf_fit_pars = None
        self.sf_fit_func = None
        self.seed = seed
        self.h5_file = None

        if self.sf_model not in {'power_law', 'exponential'}:
            raise ValueError(f'Invalid structure function model {self.sf_model}, available '
                             f'options are: "power_law" or "exponential"')
        np.random.seed(self.seed)

    def loadQSODataFromFile(self, input_file: Union[str, Path]) -> None:
        qso_df = pd.read_table(input_file)
        self.t = qso_df['mhjd'].values
        self.N_samples = len(self.t)
        self.qso_df = qso_df
        self.qso_id = Path(input_file).name.split('_')[0]
        self.qso_lc_vals = self.qso_df[f'mag_{self.image}'].values
        self.qso_lc_errs = self.qso_df[f'magerr_{self.image}'].values

    def evaluateStructureFunction(self) -> None:
        tau_matrix = self.computeLagsMatrix(self.t)
        sf_matrix = self.computeSFPointEstimatesMatrix(self.qso_lc_vals, self.qso_lc_errs)
        tau_v_sorted = [[my_tau, my_sf_val] for my_tau, my_sf_val in sorted(zip(tau_matrix.ravel(),
                                                                                sf_matrix.ravel()))]
        tau_v_sorted = np.array(tau_v_sorted)
        tau_vals = tau_v_sorted[:, 0]
        sf_vals = tau_v_sorted[:, 1]
        tau_bin_means = stats.binned_statistic(tau_vals, tau_vals, bins=100)[0]
        sf_bin_means = stats.binned_statistic(tau_vals, sf_vals, bins=100)[0]
        self.sf_lags = tau_bin_means
        self.sf_vals = sf_bin_means

    def fitStructureFunctionModel(self) -> None:
        lag_cut_off = 0.6 * (self.t[-1] - self.t[0])
        lags_mask = self.sf_lags <= lag_cut_off
        lags_vals_to_fit = self.sf_lags[lags_mask]
        sf_vals_to_fit = self.sf_vals[lags_mask]
        self.sf_fit_pars = {}
        if self.sf_model == 'power_law':
            pars = stats.linregress(np.log10(lags_vals_to_fit), np.log10(sf_vals_to_fit))
            self.sf_fit_pars['slope'] = pars[0]
            self.sf_fit_pars['intercept'] = pars[1]
            self.sf_fit_func = power_law_sf
        elif self.sf_model == 'exponential':
            V0, dtau0 = curve_fit(exponential_sf, lags_vals_to_fit, sf_vals_to_fit)
            self.sf_fit_pars['V0'] = V0
            self.sf_fit_pars['dtau0'] = dtau0
            self.sf_fit_func = exponential_sf
        else:
            raise ValueError(f'Invalid structure function model {self.sf_model}, available '
                             f'options are: "power_law" or "exponential"')

    def estimateCovarianceMatrix(self, delay: float, regularize: bool = True) -> np.ndarray:
        t_doubled = np.concatenate([self.t, self.t - delay])
        tau_matrix = self.computeLagsMatrix(t_doubled)

        signal_square_mean = integrate.simps(self.qso_lc_vals**2, self.t) / (self.t[-1] - self.t[0])

        C = signal_square_mean - self.sf_fit_func(tau_matrix, self.sf_fit_pars)
        if regularize:
            C += 1e-10 * np.eye(2*self.N_samples)
        return C

    def generatePRHLightCurves(self, delay: float, regularize_covariance: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        C = self.estimateCovarianceMatrix(delay, regularize=regularize_covariance)
        L = np.linalg.cholesky(C)
        err_y_doubled = np.concatenate([self.qso_lc_errs, self.qso_lc_errs])
        mc_light_curves_vals = L @ np.random.normal(0, 1, size=L.shape[0]) \
                               + err_y_doubled**2 * np.random.normal(0, 1, size=L.shape[0])
        return mc_light_curves_vals[:self.N_samples], mc_light_curves_vals[self.N_samples:]

    def generatePRHDatasetForDelay(self, true_delay: float, N_MC_samples: int, outfile: Union[str, Path]) -> None:
        dataset = LightCurvePRHDataset()
        dataset.createNewFile(outfile)
        dataset.writeOriginalQSOData(self.t, self.qso_lc_vals, self.qso_lc_errs)
        dataset.createMonteCarloDataGroup(true_delay=true_delay)
        for i in range(N_MC_samples):
            yA, yB = self.generatePRHLightCurves(delay=true_delay)
            dataset.writeMonteCarloLightCurvesRealization(yA=yA, yB=yB,
                                                          errA=self.qso_lc_errs,
                                                          errB=self.qso_lc_errs,
                                                          realization_id=i+1)
        dataset.close()

    def generatePRHDatasetWithRandomDelays(self, min_delay: float, max_delay: float, N_MC_samples: int,
                                           outfile: Union[str, Path]) -> None:
        dataset = LightCurvePRHDataset()
        dataset.createNewFile(outfile)
        dataset.createMonteCarloDataGroup()
        true_delays = (max_delay - min_delay) * np.random.random(size=N_MC_samples) + min_delay
        for i, true_delay in enumerate(true_delays):
            yA, yB = self.generatePRHLightCurves(delay=true_delay)
            dataset.writeMonteCarloLightCurvesRealization(yA=yA, yB=yB,
                                                          errA=self.qso_lc_errs,
                                                          errB=self.qso_lc_errs,
                                                          realization_id=i + 1,
                                                          true_delay=true_delay)
        dataset.close()

    @staticmethod
    def computeLagsMatrix(t: np.ndarray) -> np.ndarray:
        N_samples = len(t)
        t_row_repeated = np.repeat(t[:, np.newaxis], N_samples, axis=1)
        t_col_repeated = np.repeat(t[np.newaxis, :], N_samples, axis=0)
        tau = np.abs(t_row_repeated - t_col_repeated)
        return tau

    @staticmethod
    def computeSFPointEstimatesMatrix(y: np.ndarray, err_y: np.ndarray) -> np.ndarray:
        N_samples = len(y)
        if len(err_y) != N_samples:
            raise ValueError(f'y vals are {N_samples}, errors are {len(err_y)}')
        y_row_repeated = np.repeat(y[np.newaxis, :], N_samples, axis=0)
        y_col_repeated = np.repeat(y[:, np.newaxis], N_samples, axis=1)
        y_err_row_repeated = np.repeat(err_y[np.newaxis, :], N_samples, axis=0)
        y_err_col_repeated = np.repeat(err_y[:, np.newaxis], N_samples, axis=1)
        sf_matrix = (y_row_repeated - y_col_repeated)**2
        sf_matrix -= (y_err_row_repeated**2 + y_err_col_repeated**2)
        return sf_matrix


class LightCurvePRHDataset:
    _h5_file: h5py.File
    mc_group: h5py.Group
    mc_realizations_group: h5py.Group
    qso_data_group: h5py.Group
    true_delay: float
    N_MC_realizations: int

    def __init__(self, input_file: Union[str, Path] = None):
        self._h5_file = None
        self.qso_data_group = None
        self.mc_group = None
        self.mc_realizations_group = None
        self.true_delay = None
        self.N_MC_realizations = None

        if input_file is not None:
            self.loadFromFile(input_file)

    @property
    def keys(self):
        return self._h5_file.keys()

    @property
    def file(self) -> h5py.File:
        return self._h5_file

    def __getitem__(self, item: str):
        return self._h5_file[item]

    def createNewFile(self, file: Union[str, Path]) -> None:
        if not isinstance(file, (str, Path)):
            raise TypeError(f'Invalid type {type(file)} for new file path')
        if Path(file).exists():
            raise FileExistsError(file)

        self._h5_file = h5py.File(file, 'w')

    def writeOriginalQSOData(self, t_domain: np.ndarray,
                             qso_light_curve_vals: np.ndarray,
                             qso_light_curve_errs: np.ndarray):
        orig_data_group = self._h5_file.create_group('original_qso_data')
        orig_data_group.create_dataset('time_domain', data=t_domain,
                                       compression='gzip', compression_opts=9)
        orig_data_group.create_dataset('qso_light_curve_values', data=qso_light_curve_vals,
                                       compression='gzip', compression_opts=9)
        orig_data_group.create_dataset('qso_light_curve_errors', data=qso_light_curve_errs,
                                       compression='gzip', compression_opts=9)

    def createMonteCarloDataGroup(self, true_delay: float = None) -> None:
        self.mc_group = self._h5_file.create_group('MonteCarlo_Data')
        if isinstance(true_delay, float):
            self.mc_group.create_dataset('true_delay', data=true_delay)
        self.mc_realizations_group = self.mc_group.create_group('Realizations')

    def writeMonteCarloLightCurvesRealization(self, yA: np.ndarray, yB: np.ndarray, errA: np.ndarray, errB: np.ndarray,
                                              realization_id: int, true_delay: float = None) -> None:
        r_group = self.mc_realizations_group.create_group(f'realization_{realization_id}')
        r_group.create_dataset('yA', data=yA, compression='gzip', compression_opts=9)
        r_group.create_dataset('yB', data=yB, compression='gzip', compression_opts=9)
        r_group.create_dataset('errA', data=errA, compression='gzip', compression_opts=9)
        r_group.create_dataset('errB', data=errB, compression='gzip', compression_opts=9)
        if isinstance(true_delay, float):
            r_group.create_dataset('true_delay', data=true_delay)

    def loadFromFile(self, file: Union[str, Path]) -> None:
        self._h5_file = h5py.File(file, 'r')
        self.qso_data_group = self._h5_file['original_qso_data']
        self.mc_group = self._h5_file['MonteCarlo_Data']
        if 'true_delay' in self.mc_group.keys():
            self.true_delay = self.mc_group.get('true_delay')
        self.mc_realizations_group = self.mc_group['Realizations']
        self.N_MC_realizations = len(self.mc_realizations_group.keys())

    def getOriginalQSOData(self) -> h5py.Group:
        return self.qso_data_group

    def getMCRealization(self, realization_id: int) -> h5py.Group:
        try:
            realization_key = f'realization_{realization_id}'
            mc_realization_group = self.mc_realizations_group[realization_key]
        except KeyError:
            raise KeyError(f'Invalid realization id {realization_id}, available IDs are '
                           f'from 1 to {self.N_MC_realizations}')
        return mc_realization_group

    def close(self):
        self._h5_file.close()


