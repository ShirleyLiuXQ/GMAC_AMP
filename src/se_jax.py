import time
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import logging
from typing import Tuple
from tqdm import tqdm
from ldpc_jossy.py.ldpc_awgn import bpsk
from src.helper_fns import initialise_logger, timestamp
from numba import jit

from src.init_model import SignalMatrix, create_codebook, noise_mat_W
from src.denoiser_jax import Denoiser_jax
from src.terminate_algo import terminate

# Functions for running SE. Compatible with any denoising function eta.
# "mc" in function names represents "Monte Carlo" (MC).
#
# The strategy is to use MC samples of G to take expectation over G and
# analytically evaluates the expectation over X0. This produces the same
# result as when one uses MC samples of both G and X0.
logger = logging.getLogger(__name__)
MSE_RTOL = 1e-7
MSE_RTOL_SC = 1e-6

class SE_jax:
    def __init__(self, δ: float, σ2_W: float, d: int, denoiser: Denoiser_jax, \
                  log_file_name: str, iter_max: int, \
                    num_G_samples: int=10 ** 4, num_X0_samples: int=1,
                    use_allzero_X0: bool=False) -> None:
        """
        iter_max should be the same as that of AMP.
        δ is the undersampling ratio δ=n/L
        sigma2_W is the noise variance of the additive noise matrix W.

        NOTE: use_allzero_X0=True should only be used for LDPC
        codes with non-marginal denoiser.
        The X0 samples are 0 and 1 for mmse-marginal denoiser.
        Entire codebook is used as XO samples for simple codes.
        """
        self.log_file_name = log_file_name
        initialise_logger(logger, log_file_name)
        # logger.debug("Initialising SE jax")
        assert δ > 0
        self.δ = δ

        # Actual denoiser to apply on matrix effective observation when we estimate
        # pErrors etc via Monte Carlo:
        self.denoiser = denoiser
        self.actual_d = d

        # Define a different denoiser to run SE:
        if denoiser.type == "mmse" or \
            denoiser.type == "ldpc-bp" or \
            denoiser.type == "thres-chi_squared" or denoiser.type == "thres-gaussian" or \
            denoiser.type == "mmse-marginal-thesis":
            # mmse-marginal-thesis needs to store different entries
            # along the diagonal of the covariance matrix, whereas
            # mmse-margin only stores the avg variance as a scalar.
            self.d = d
        elif denoiser.type == "mmse-marginal":
            self.d = 1
        else:
            raise NotImplementedError

        self.signal_power = np.eye(self.d) * (1-denoiser.α)
        assert σ2_W >= 0
        self.Σ_W = σ2_W * np.eye(self.d) # covariance matrix of W
        self.iter_max = iter_max
        self.num_G_samples, self.num_X0_samples = num_G_samples, num_X0_samples
        self.use_allzero_X0 = use_allzero_X0

    def run(self) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        logger.debug("Running SE jax")
        start_time = time.time()
        # Initialize arrays to store results:
        mse_t_arr = np.zeros(self.iter_max) # scalars

        MSE_t = self.signal_power # dxd MSE matrix of last iteration
        mse_t_arr[0] = np.mean(np.diag(MSE_t))
        Tt = self.Σ_W + 1/self.δ * MSE_t # dxd effective noise cov matrix of last iteration
        Tt_arr = np.zeros((self.iter_max, self.d, self.d)) # across iterations
        Tt_arr[0] = Tt
        for t in tqdm(range(1, self.iter_max)):
            MSE_t = MSE_mc(Tt, self.denoiser, self.log_file_name,
                self.num_G_samples, self.num_X0_samples, self.use_allzero_X0)

            mse_t_arr[t] = np.mean(np.diag(MSE_t))
            Tt = self.Σ_W + 1/self.δ * MSE_t
            Tt_arr[t] = Tt
            # Monte Carlo is extremely computationally expensive, terminate early
            # according to scalar mse:
            if terminate(mse_t_arr, t, self.iter_max, MSE_RTOL):
                # Fill in the remaining entries with constant values:
                mse_t_arr[t+1 :] = -1 # mse_t_arr[t]
                break
        logger.debug(f"SE terminated after {t} iterations")
        logger.debug('Finished SE in %s hours', (time.time() - start_time)/3600)
        return MSE_t, mse_t_arr, t, Tt, Tt_arr

    def last_iter_mc(self, Tt_conv: np.array, last_iter_num_G_samples: int=None) \
        -> Tuple[np.array, np.array, np.array]:
        """
        Reproduces effective observation for the last iteration of AMP, and applies
        denoiser to reproduce Xt upon convergence of AMP.
        Also returns MAP estimate of X.

        Tt_conv is the noise covariance of the noise in the last iteration of SE.

        last_iter_num_G_samples: X0 samples in the last iteration has shape
        last_iter_num_G_samples x d. Make sure last_iter_num_G_samples is sufficiently
        large to capture small error probabilities.
        last_iter_num_G_samples defaults to self.num_G_samples.
        """
        if last_iter_num_G_samples is None:
            last_iter_num_G_samples = self.num_G_samples
        # if self.denoiser.type == "mmse-marginal":
        #     Tt_conv = np.eye(self.actual_d) * Tt_conv # convert into diagonal matrix
            # so the sampling step below samples the correct number of columns in X
            # Tt_arr = np.eye(self.actual_d)[np.newaxis, :] * Tt_arr
        # TODO: make X0 deterministic and sample G_arr randomly like in MSE_mc:
        X0 = SignalMatrix(last_iter_num_G_samples, self.denoiser.α,
                self.denoiser.codebook, self.denoiser.ldpc_code, self.actual_d).sample()  # num_G_samples x actual_d
        G_arr = noise_mat_W(last_iter_num_G_samples, self.actual_d, Tt_conv)  # num_G_samples x actual_d
        S = X0 + G_arr  # num_G_samples x actual_d
        Xt, _, X_MAP = self.denoiser.η_sum_dη_XMAP(
            S, Tt_conv, calc_sum_dη=False, calcMAP_estimate=True)
        return X0, Xt, X_MAP


class SE_SC_jax:
    def __init__(self, W: np.array, δ_in: float, σ2_W: float, d: int, \
                 denoiser: Denoiser_jax, log_file_name: str, iter_max: int, \
                    num_G_samples: int=10 ** 3, num_X0_samples: int=1,
                    use_allzero_X0: bool=False) -> None:
        """
        By default, use fewer MC samples of G than iid case for speed.

        NOTE: use_allzero_X0=True should only be used for LDPC
        codes with non-marginal denoiser.
        The X0 samples are 0 and 1 for mmse-marginal denoiser.
        Entire codebook is used as XO samples for simple codes.
        """
        self.log_file_name = log_file_name
        initialise_logger(logger, log_file_name)
        # logger.debug("Initialising SE-SC jax")

        # Base matrix W:
        self.W = W
        self.R, self.C = self.W.shape
        assert δ_in > 0
        self.δ_in = δ_in

        # Actual denoiser to apply on matrix effective observation in Monte Carlo estimation:
        self.denoiser = denoiser
        self.actual_d = d

        # Define a different denoiser to run SE:
        if denoiser.type == "mmse" or \
            denoiser.type == "ldpc-bp" or \
            denoiser.type == "thres-chi_squared" or denoiser.type == "thres-gaussian" or \
            denoiser.type == "mmse-marginal-thesis":
            self.d = d
        elif denoiser.type == "mmse-marginal":
            self.d = 1
        else:
            raise NotImplementedError

        self.signal_power = np.eye(self.d) * (1-denoiser.α)
        assert σ2_W >= 0
        self.Σ_W = σ2_W * np.eye(self.d) # covariance matrix of W
        self.iter_max = iter_max
        self.num_G_samples, self.num_X0_samples = num_G_samples, num_X0_samples
        self.use_allzero_X0 = use_allzero_X0
        # logger.debug("Finished initialising SE-SC jax")

    def run(self):
        """Run Spatially Coupled State Evolution for the specified setting"""
        logger.debug("Running SE-SC")
        start_time = time.time()

        mse_t_arr = np.zeros((self.iter_max, self.C)) # for each column block
        avg_var_c = np.zeros((self.iter_max, self.C)) # for each column block
        Phi_t_arr = np.zeros((self.iter_max, self.R, self.d, self.d)) # across iterations
        # For current iteration t, define the following variables:
        # Psi (Cxdxd), mse_c (C), Phi (Rxdxd), Phi_inv (Rxdxd)
        # logger.debug("Initialising Psi, mse_c")
        Psi_t, mse_c = self._init_Psi_mse_c(self.signal_power, self.W)
        # logger.debug("Initialising Phi, Phi_inv")
        Phi_t, Phi_t_inv = self._update_Phi_Phi_inv(self.W, self.Σ_W, self.δ_in, Psi_t)
        # logger.debug("Initialising mse")
        mse_t_arr[0] = mse_c # average across all columns of W
        Phi_t_arr[0] = Phi_t

        for t in tqdm(range(1, self.iter_max)):
            # time_start = time.time()
            # logger.debug("Updating Pt")
            Pt = self._update_Pt(self.W, Phi_t_inv) # Cxdxd
            avg_var_c[t] = np.diagonal(Pt, axis1=1, axis2=2).mean(axis=1)
            # logger.debug("Updating Psi")
            for c in range(self.C):
                Psi_t[c] = MSE_mc(Pt[c], self.denoiser, self.log_file_name, \
                    self.num_G_samples, self.num_X0_samples, self.use_allzero_X0)
            # logger.debug("Updating mse_c")
            mse_c = self._update_mse_c(self.C, Psi_t)
            # logger.debug("Updating Phi, Phi_inv")
            # if (np.any(np.isnan(Psi_t))):
            #     Psi_t_contains_nan = True
            #     logger.debug("Psi_t contains nan") # this shouldnt happen,
            #     # but when it does, it's usually the number of MC samples is too low.
            #     t = t-1 # avoid storing current iteration's results
            # else:
            #     Psi_t_contains_nan = False
            Phi_t, Phi_t_inv = self._update_Phi_Phi_inv(self.W, self.Σ_W, self.δ_in, Psi_t)
            # logger.debug("Updating mse")
            mse_t_arr[t] = mse_c
            Phi_t_arr[t] = Phi_t
            # logger.debug(f"Finished iteration {t} in {time.time() - time_start} seconds")
            avg_mse_t_arr = np.mean(mse_t_arr, axis=1) # avg across column blocks
            if terminate(avg_mse_t_arr, t, self.iter_max, MSE_RTOL_SC): # or Psi_t_contains_nan:
                # mse_t_arr = mse_t_arr[:t+1] # Pablo truncates the array
                mse_t_arr[t+1 :, :] = -1 # mse_t_arr[t] # I fill in remaining entries with constant values
                break
        if False:
            plt.figure()
            n_lines = 10
            cmap = mpl.colormaps['plasma']
            # Take colors at regular intervals spanning the colormap.
            colors = cmap(np.linspace(0, 1, n_lines))
            for i_t, t in enumerate(np.linspace(1, self.iter_max-1, num=n_lines, dtype=int)):
                # idx=0 is empty (i.e. all zeros)
                plt.plot(avg_var_c[t], label=f't={t}', color=colors[i_t])
            plt.legend()
            plt.xlabel('Column block index')
            plt.ylabel('Variance of effective noise')
            plt.title(f'SC-SE, C={self.C}, R={self.R}, d={self.d}, denoiser={self.denoiser.type}')
            my_time = timestamp()
            plt.savefig('./results/sc_se_noise_var_c_' + my_time + '.pdf')
        plt.figure()
        n_lines = np.min([10, t])
        cmap = mpl.colormaps['plasma']
        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, n_lines))
        for i_t0, t0 in enumerate(np.linspace(0, t, num=n_lines, dtype=int)):
            # idx=0 is empty (i.e. all zeros)
            plt.plot(mse_t_arr[t0], label=f't={t0}', color=colors[i_t0])
        plt.legend()
        plt.xlabel('Column block index')
        plt.ylabel('mse per block')
        plt.title(f'SC-SE, C={self.C},R={self.R},d={self.d},δ_in={self.δ_in},\n' + \
                  f'denoiser={self.denoiser.type},{self.num_G_samples}G')
        my_time = timestamp()
        plt.savefig('./results/sc_se_mse_c_' + my_time + '.pdf')
        logger.debug(f"SE-SC terminated after {t} iterations")
        logger.debug('Finished SE-SC in %s hours', (time.time() - start_time)/3600)
        # TODO: ask Pablo which variables are needed to run AMP. Return those.
        return Psi_t, Phi_t, mse_t_arr, t, Pt, Phi_t_arr

    @staticmethod
    @jit(nopython=True)
    def _init_Psi_mse_c(signal_power, W):
        R, C = W.shape
        d = signal_power.shape[0]
        # For current iteration t, define the following variables:
        # Psi is Cxdxd. np.tile not supported by numba
        # Psi = np.tile(signal_power, (C, 1, 1)) # signal_power is dxd
        Psi = np.zeros((C, d, d))
        for c in range(C):
            Psi[c] = signal_power
        mse_c = np.mean(np.diag(signal_power)) * np.ones(C) # length-C vector
        return Psi, mse_c

    @staticmethod
    @jit(nopython=True)
    def _update_Phi_Phi_inv(W, Σ_W, δ_in, Psi):
        R, C = W.shape
        d = Σ_W.shape[0]
        Phi = np.zeros((R,d,d))
        Phi_inv = np.zeros((R,d,d))
        for r in range(R):
            # Phi[r] = Σ_W + (1/δ_in)*np.einsum('i,ijk->jk', W[r,:], Psi)
            Phi[r] = Σ_W
            for c in range(C):
                Phi[r] += (1/δ_in)*W[r,c]*Psi[c]
            Phi_inv[r] = np.linalg.inv(Phi[r])
        return Phi, Phi_inv

    @staticmethod
    @jit(nopython=True)
    def _update_Pt(W, Phi_inv):
        """Phi_inv is Rxdxd. Pt is the covariance of Gc, effective noise of block c."""
        R, C = W.shape
        d = Phi_inv.shape[-1]
        Pt = np.zeros((C,d,d))
        for c in range(C):
            # sum_W_Phi = np.einsum('i,ijk->jk', W[:,c], Phi_inv)
            sum_W_Phi = np.zeros((d,d))
            for r in range(R):
                sum_W_Phi += W[r,c]*Phi_inv[r]
            Pt[c] = np.linalg.inv(sum_W_Phi)
        return Pt

    @staticmethod
    @jit(nopython=True)
    def _update_mse_c(C, Psi):
        mse_c = np.zeros(C)
        for c in range(C):
            mse_c[c] = np.mean(np.diag(Psi[c]))
        return mse_c


    def last_iter_mc(self, Pt_conv: np.array, last_iter_num_G_samples: int=None) \
        -> Tuple[np.array, np.array, np.array]:
        """
        Reproduces effective observation for the last iteration of AMP, and applies
        denoiser to reproduce Xt upon convergence of AMP.
        Also returns MAP estimate of X.

        Pt_conv: Cxdxd is the noise covariance of the noise in the last iteration of SE.
        When denoiser is mmse-marginal, Pt_conv is Cx1x1.

        last_iter_num_G_samples: X0 samples in the last iteration has shape
        last_iter_num_G_samples*C x d. Make sure last_iter_num_G_samples is sufficiently
        large to capture small error probabilities.
        last_iter_num_G_samples defaults to self.num_G_samples.
        """
        # if self.denoiser.type == "mmse-marginal":
        #     # Pt_conv is Cxdxd where d=1
        #     Pt_conv = Pt_conv.reshape(len(Pt_conv), 1, 1) * \
        #         np.eye(self.actual_d)[np.newaxis, :] # convert into C diagonal matrices
        #     assert Pt_conv.shape == (self.C, self.actual_d, self.actual_d)
        #     # so the sampling step below samples the correct number of columns in X
        if last_iter_num_G_samples is None:
            last_iter_num_G_samples = self.num_G_samples

        N = last_iter_num_G_samples
        X0 = SignalMatrix(int(N * self.C), self.denoiser.α,
                          self.denoiser.codebook, self.denoiser.ldpc_code,
                          self.actual_d).sample()
        G_arr = np.ones((self.C, N, self.actual_d))
        S = np.ones(X0.shape)
        Xt = np.ones(X0.shape)
        X_MAP = np.ones(X0.shape)
        # Estimate one column block at a time:
        for c in range(self.C):
            G_arr[c] = noise_mat_W(N, self.actual_d, Pt_conv[c])  # num_G_samples x d
            S[N*c : (N*c + N)] = X0[N*c : (N*c + N)] + G_arr[c]  # num_G_samples x d
            Xt[N*c : (N*c + N)], _, X_MAP[N*c : (N*c + N)] = \
                self.denoiser.η_sum_dη_XMAP(
                S[N*c : (N*c + N)], Pt_conv[c],
                calc_sum_dη=False, calcMAP_estimate=True)  # num_G_samples x d
        return X0, Xt, X_MAP

def MSE_mc(Tt_or_τt, denoiser: Denoiser_jax, log_file_name, \
           num_G_samples: int, num_X0_samples: int=1,
           use_allzero_X0: bool=False) -> np.array:
    """
    denoiser is the denoiser to use in SE.
    For mmse-marginal denoisers, Σ_or_σ2 is a scalar and d=1.

    This function estimates MSE = E[(X0 - eta(S)) * (X0 - eta(S))^T]
                                = E[(X0 - eta(X0+G)) * (X0 - eta(X0+G))^T]]

    It draws num_G_samples Monte Carlo samples of G from N_d(0, Tt) to estimate the
    expectation over G, and analytically evaluates the expectation over X0 wherever
    possible.

    For linear codes with additive Gaussian noise, we can simply analyse the
    decoder's performance on the all-zero (bpsk all-one) codeword X0 due to
    symmetry in the codewords and the Gaussian noise.

    This function cannot be jitted because ldpc.decode() cannot be jitted, and
    operations such as einsum arent supported by numba. We've jitted most of its
    sub-functions and left the unjittable ones as they are.

    NOTE: This function creates samples at once so is prone to memory overflow.
    But with num_G_samples=1000, num_X0_samples=1000, d=1000, it uses 8GB so
    need hi-mem setting. (each CPU on HPC has 6GB memory, hi-memory CPU each has 12GB)

    NOTE:
    When use_allzero_X0 is True, the function only uses the all-zero codeword
    as its X0 sample. This means num_X0_samples=1 and the input num_X0_samples is ignored.
    When denoiser_type is 'mmse-marginal', sample 0 and 1, which means num_X0_samples=2
    and input num_X0_samples is ignored.
    When denoiser is mmse, use entire codebook as samples of X0, so num_X0_samples is ignored.
    """
    # assert is_valid_covariance_matrix(Tt)
    # TODO: could remove the check above because Tt may have small negative
    # eigenvalues due to numerical errors.
    initialise_logger(logger, log_file_name)

    # logger.debug("Entered MSE_mc")
    Tt_or_τt = np.array([[Tt_or_τt]]) if np.ndim(Tt_or_τt) == 0 else Tt_or_τt
    d = Tt_or_τt.shape[0] # d for SE
    assert Tt_or_τt.shape == (d, d)
    Tt_or_τt = Tt_or_τt + 1e-12 * np.eye(d) # add small positive constant to diagonal

    # logger.debug("Sampling G")
    # 2) Samples of noise only (all-zero signal):
    def MSE_mc_G_batch(num_G_samples, num_X0_samples):
        G_arr = noise_mat_W(num_G_samples, d, Tt_or_τt)
        if denoiser.α == 1: # all users silent, no need to sample X0:
            η_zero_X_arr = denoiser.η_sum_dη_XMAP(G_arr, Tt_or_τt)[0] # num_G_samples x d
            mse_zero_X = denoiser.α * η_zero_X_arr.T @ η_zero_X_arr / num_G_samples
            assert mse_zero_X.shape == (d, d)
            MSE = mse_zero_X
        else: # not all users are silent, sample X0:
            # For mmse-marginal or ldpc-bp denoisers, we only use all-zero X0.
            # For mmse denoiser, we sample codewords (mmse is never used for LDPC codes):
            # logger.debug("Sampling X0")
            if denoiser.type == 'mmse-marginal': # only all-zero codeword also works fine
                codewords = create_codebook(d=1)
                assert codewords.shape == (2, 1)
                num_X0_samples = codewords.shape[0]
            elif denoiser.type == 'mmse-marginal-thesis' or \
                'thres' in denoiser.type:
                # Hack:
                # codewords = create_codebook(d)
                # num_X0_samples = codewords.shape[0]
                # Randomly sample a nonzero uncoded sequence, like for ldpc-bp:
                codewords = np.random.randint(0, 2, (num_X0_samples, d))
                codewords = bpsk(codewords) # convert to BPSK
            elif denoiser.type == 'ldpc-bp':
                if use_allzero_X0:
                    num_X0_samples = 1
                    # all-zero codeword (zeros are mapped into BPSK +1; ones into BPSK -1):
                    codewords = np.ones((num_X0_samples, d))
                else:
                    # LDPC codes have too many codewords, cannot use all.
                    # Sample codewords at random, or rather, sample user bits
                    # (which are independent) at random:
                    codewords = SignalMatrix(num_X0_samples, denoiser.α,\
                                            None, denoiser.ldpc_code).sample()
            else: # mmse denoiser for simple codes or for RA without outer code.
                # Use entire codebook as samples:
                assert denoiser.codebook is not None, \
                    "this should be simple codes with mmse denoiser or GMAC with random user activity " + \
                    "(uncoded case is handled by mmse-marginal, LDPC codes by ldpc-bp)"
                codewords = denoiser.codebook
                assert codewords.shape[1] == d
                num_X0_samples = codewords.shape[0] # all codewords used as samples
                # Below doesnt work because additive noise in effective observation is
                # not always uncorrelated across entries.
                # num_X0_samples = 2 # arbitrary
                # codewords = codewords[np.random.choice(codewords.shape[0], num_X0_samples, replace=True), :]
            X0 = codewords[..., np.newaxis]
            # logger.debug("Finished sampling X0")

            if denoiser.α == 0: # all users active
                mse_zero_X = np.zeros((d, d))
            else:
                # logger.debug("Calculating eta_zero_X_arr")
                η_zero_X_arr = denoiser.η_sum_dη_XMAP(G_arr, Tt_or_τt)[0] # num_G_samples x d
                # logger.debug("Calculating mse_zero_X")
                mse_zero_X = denoiser.α * η_zero_X_arr.T @ η_zero_X_arr / num_G_samples
                assert mse_zero_X.shape == (d, d)

            # 1) samples of signal + noise:
            # logger.debug("Creating S")
            S = X0 + G_arr.T[np.newaxis, ...]
            assert S.shape == (num_X0_samples, d, num_G_samples)
            num_rows = num_X0_samples * num_G_samples
            # logger.debug('Flattening S')
            S_flattened = S.transpose(0, 2, 1).reshape(-1, d) # reshape not supported by
            # numba because array in non-contiguous
            assert S_flattened.shape == (num_rows, d)
            # logger.debug("Calculating eta_nonzero_X")
            η_nonzero_X = denoiser.η_sum_dη_XMAP(S_flattened, Tt_or_τt)[0]  # num_rows x d
            # X0 is 3D: num_X0_samples x d x 1
            # logger.debug("Calculating mse_nonzero_X")
            diff =(X0.transpose(0, 2, 1) - η_nonzero_X.reshape(
                num_X0_samples, num_G_samples, d)).reshape(-1, d)
            assert diff.shape == (num_rows, d)
            mse_nonzero_X = (1 - denoiser.α) * diff.T @ diff / num_rows
            assert mse_nonzero_X.shape == (d, d)

            MSE = mse_nonzero_X + mse_zero_X
            return MSE

    # NOTE: num_G_samples too large causes memory overflow.
    # Ensure d x num_Gs_per_batch < 200k (2himem), 800k (8himem)
    num_Gs_per_batch = np.min([int(8e5 / d), num_G_samples])
    num_batches = int(np.ceil(num_G_samples / num_Gs_per_batch))
    MSE = np.zeros((d, d))
    for _ in range(num_batches):
        MSE += MSE_mc_G_batch(num_Gs_per_batch, num_X0_samples)
    MSE = MSE / num_batches
    # logger.debug("Exiting MSE_mc")
    # assert is_valid_covariance_matrix(MSE)
    # NOTE: dont check this because eigvals may be small negative due to numerical error
    assert MSE.shape == (d, d)
    return MSE
