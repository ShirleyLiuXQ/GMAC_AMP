import time
from matplotlib import pyplot as plt
import matplotlib as mpl
from numba import jit
import numpy as np
import logging
from ldpc_jossy.py.ldpc import decode
from ldpc_jossy.py.ldpc_awgn import bpsk, ch2llr
from src.helper_fns import initialise_logger, timestamp
from tqdm import tqdm

from src.init_model import SignalMatrix, add_iid_noise, create_codebook, design_mat_A, design_mat_A_sc
from src.terminate_algo import calc_PUPE, count_errors, terminate
from src.sub_dct import sub_dct_iid_mat
from src.denoiser_jax import Denoiser_jax
from src.constants import JOSSY_MAX_ITCOUNT

logger = logging.getLogger(__name__)
MSE_RTOL = 1e-7
MSE_RTOL_SC = 1e-6
# parallel=True slows code down

class AMP_jax:
    def __init__(self, n: int, L: int, d: int, σ2_W, denoiser: Denoiser_jax, \
                 log_file_name: str, iter_max: int, num_trials: int, \
                    A_type: str = 'dct', estimate_Tt: bool = True) -> None:
        """
        iter_max should be the same as that of SE.
        A_type: type of sensing matrix: 'gaussian' or 'dct'
        estimate_Tt: - if True estimate Tt on the fly using AMP (i.e. by the
                       empirical covariance of the rows in Zt), this sometimes gives
                       diverging mse (i.e. mse first decreases then increases away from 0);
                     - if False use Tt from SE which user needs to provide to run().
        """
        self.log_file_name = log_file_name
        initialise_logger(logger, log_file_name)
        # logger.debug('Initialising AMP')
        assert n > 0 and L > 0 and d > 0 \
            and np.isscalar(n) and np.isscalar(L) and np.isscalar(d)
        self.n = n
        self.L = L
        self.d = d
        self.denoiser = denoiser
        # denoiser is storing the signal prior:
        self.signal_matrix = SignalMatrix(self.L, denoiser.α, \
                            denoiser.codebook, denoiser.ldpc_code, self.d)
        self.signal_power = (1-denoiser.α) * np.eye(self.d)

        assert σ2_W >= 0 and np.isscalar(σ2_W)
        self.σ2_W = σ2_W
        assert iter_max >= 1 and num_trials >= 1
        self.iter_max, self.num_trials = iter_max, num_trials

        self.A_type = A_type
        self.estimate_Tt = estimate_Tt
        # logger.debug('Finished initialising AMP')


    def _one_iter(self, Y, Xt, Zt, Tt: np.array = None, is_last_iter: bool = False):
        """
        AMP recursion to update the estimate Xt and the residual Zt.

        Y, Xt, Zt must be 2D arrays.
        Xt in Lxr, Y and Zt are nxr.
        Tt is the effective noise covariance.

        If is_last_iter==True, then return the MAP estimate of X.
        """
        # logger.debug('Starting one iteration')
        assert Y.shape == (self.n, self.d) == Zt.shape
        assert Xt.shape == (self.L, self.d)
        if Tt is None: # estimate Tt on the fly
            if self.denoiser.type == 'mmse' or \
                self.denoiser.type == 'ldpc-bp' or \
                self.denoiser.type == 'thres' or \
                self.denoiser.type == 'mmse-marginal-thesis':
                Tt = np.cov(Zt, rowvar=False) # scalar when d=1
                assert Tt.shape == (self.d, self.d) if self.d>1 else True
            elif self.denoiser.type == 'mmse-marginal':
                Tt = np.var(Zt.flatten()) # scalar
                # Zt_flattened = Zt.reshape(-1, 1)
                # Tt = np.cov(Zt_flattened, rowvar=False) # scalar
            else:
                raise NotImplementedError
        # logger.debug('Finished estimating Tt')

        AZ_term = self.A.T @ Zt if self.A_type == 'gaussian' else self.AY(Zt)
        # logger.debug('Finished calculating AZ_term')
        St = Xt + AZ_term
        assert not np.any(np.isnan(St))
        Xt, sum_dη, X_MAP = self.denoiser.η_sum_dη_XMAP(
            St, Tt, calc_sum_dη=True, calcMAP_estimate=is_last_iter)
        assert not np.any(np.isnan(Xt))
        assert not np.any(np.isnan(sum_dη))
        # logger.debug('Finished calculating Xt, sum_dη, X_MAP')
        AX_term = self.A @ Xt if self.A_type == 'gaussian' else self.AX(Xt)
        # logger.debug('Finished calculating AX_term')
        Zt = Y - AX_term + 1 / self.n * Zt @ sum_dη.T
        # logger.debug('Finished calculating Zt')
        assert Zt.shape == (self.n, self.d)
        # logger.debug('Finished one iteration')
        return St, Xt, Zt, Tt, X_MAP


    def _one_trial(self, Tt_arr: np.array = None):
        """
        For a fixed sensing matrix, run AMP for iter_max iterations.

        Tt_arr is the array of Tt's across iter_max iterations.
        """
        assert Tt_arr.shape == (self.iter_max, self.d, self.d) \
            if Tt_arr is not None else True
        mse_t_arr = np.zeros(self.iter_max)

        # Initialise:
        logger.debug('Initialising one_trial')
        # AMP theory is based on randomness in A, so A needs to be sampled per trial:
        # logger.debug(f"Creating A with shape n={self.n} x L={self.L}")
        # Initialize and fix the sensing matrix for all trials:
        if self.A_type == 'gaussian':
            self.A = design_mat_A(self.n, self.L)
            self.AX, self.AY = None, None
        elif self.A_type == 'dct':
            self.AX, self.AY = sub_dct_iid_mat(self.n, self.L, self.d)
            self.A = None
        else:
            raise ValueError('A_type must be gaussian or dct.')

        # if self.denoiser.ldpc_code is None:
        X0 = self.signal_matrix.sample()
        # hack:
        # cb = create_codebook(self.d) # 2^d x d
        # else:
            # X0 = np.ones((self.L, self.d)) # all-zero codewords only
        nl_Y = self.A @ X0 if self.A_type == 'gaussian' else self.AX(X0)  # nxr
        Y = add_iid_noise(nl_Y, self.σ2_W)
        Xt = np.zeros_like(X0, dtype=np.float64)  # Initial signal estimate
        Zt = Y  # Initial residual
        mse_t_arr[0] = np.mean((Xt - X0) ** 2)
        # logger.debug('Finished initialising one_trial')
        for t in tqdm(range(1, self.iter_max)):
            Tt = Tt_arr[t-1] if Tt_arr is not None else None
            Xt, Zt, Tt = self._one_iter(Y, Xt, Zt, Tt)[1:-1]
            mse_t_arr[t] = np.mean((Xt - X0) ** 2)
            if terminate(mse_t_arr, t, self.iter_max, MSE_RTOL):
                mse_t_arr[t+1 :] = -1 # mse_t_arr[t]
                break
        logger.debug(f'Terminated/ converged after {t} iterations.' + \
                     f' mse upon convergence: {mse_t_arr[t]}')
        # Calc MAP estimate after AMP has converged:
        St, _, _, Tt, X_MAP = self._one_iter(Y, Xt, Zt, Tt, is_last_iter=True)
        logger.debug('Finished one_trial. mse btw X_MAP and X0: ' + \
                     f'{np.mean((X_MAP - X0)**2)}')
        return X0, X_MAP, St, Tt, mse_t_arr, t

    def run(self, Tt_arr: np.array = None, run_bp_post_amp: bool =False, \
            calc_pErrors: bool=False, \
            calcPUPE: bool=False, calcBER: bool=False):
        """
        Run AMP for num_trials times, each with iter_max iterations.

        Calculate scalar MSE by default for each trial, and also calculate
        pMD, pFA, pAUE per trial if calc_pError==True using X_MAP and ground truth X0.

        When all users are active: calculate PUPE or BER.
        Calculate PUPE per trial if calcPUPE==True.
        """
        assert self.estimate_Tt == (Tt_arr is None), \
            "Tt_arr output from SE must be provided if estimate_Tt==False, and vice versa"
        logger.debug(f"Running AMP with {self.num_trials} trials and {self.iter_max} iterations")

        mse_t_arr = np.zeros((self.num_trials, self.iter_max))
        # number of iterations needed for AMP to converge:
        iter_idx_converge_arr = np.zeros(self.num_trials)

        # By default calculate metrics using the MAP estimate:
        if calc_pErrors:
            # by default use MAP solution
            pMD_arr = np.zeros(self.num_trials)
            pFA_arr = np.zeros(self.num_trials)
            pAUE_arr = np.zeros(self.num_trials)
        else:
            pMD_arr, pFA_arr, pAUE_arr = None, None, None
        PUPE_arr = np.zeros(self.num_trials) if calcPUPE else None
        BER_arr = np.zeros(self.num_trials) if calcBER else None
        for i_trial in range(self.num_trials):
            logger.debug(f"Running AMP iid trial {i_trial + 1}/{self.num_trials}")
            (X0, X_MAP, St, Tt, mse_t_arr[i_trial, :], iter_idx_converge_arr[i_trial]
                ) = self._one_trial(Tt_arr)
            ############# Post-processing ##############
            logger.debug(f"postprocessing={run_bp_post_amp}")
            if run_bp_post_amp:
                logger.debug(f'mse before BP post AMP: {np.mean((X_MAP - X0)**2)}')
                logger.debug('Starting to run BP post AMP')
                # Tt is not None, guaranteed by _one_iter.
                # CASE 1: For AMP then BP, the AMP denoiser is marginal-mmse
                # (i.e. mmse for d=1). Tt is a 1x1 matrix.
                # CASE 2: For AMP with BP denoiser, Tt is an rxr matrix.
                if self.denoiser.type == 'mmse-marginal':
                    σ2_or_σ2_arr = np.array(Tt).item() # Same σ2 for every entry in the length-d vector St_l:
                    St = np.array(St) # post_process_row expects St to be a np.array
                    # while St output from mmse-marginal denoiser is a traced Jax object.
                elif self.denoiser.type == 'ldpc-bp':
                    σ2_or_σ2_arr = np.diag(Tt) # length-d vector
                else:
                    raise ValueError("denoiser.type must be mmse-marginal or ldpc-bp," + \
                                     f" not {self.denoiser.type}")
                X_MAP_post_BP = np.zeros_like(X0, dtype=np.float64)
                for l in range(self.L):
                    X_MAP_post_BP[l, :] = post_process_row(
                        St[l,:], σ2_or_σ2_arr, self.denoiser.ldpc_code)
                X_MAP = X_MAP_post_BP # replace
                logger.debug('Finished running BP post AMP')
                logger.debug(f"mse after BP post AMP: {np.mean((X_MAP - X0)**2)}")
            ############################################
            if calc_pErrors:
                (_, _, _, _, pMD_arr[i_trial], pFA_arr[i_trial],
                    pAUE_arr[i_trial]) = count_errors(X0, np.array(X_MAP))
            if calcPUPE:
                PUPE_arr[i_trial] = calc_PUPE(X0, X_MAP)
                logger.debug(f"PUPE after convergence: {PUPE_arr[i_trial]}")
            if calcBER:
                BER_arr[i_trial] = np.mean(X_MAP != X0)
                logger.debug(f"BER after convergence: {BER_arr[i_trial]}")
        logger.debug(f"Finished AMP iid")
        return mse_t_arr, iter_idx_converge_arr, pMD_arr, pFA_arr, pAUE_arr, \
            PUPE_arr, BER_arr



class AMP_SC_jax:
    def __init__(self, W: np.array, n: int, L: int, d: int, sigma2_W, denoiser: Denoiser_jax, \
                 log_file_name: str, iter_max: int, num_trials: int, \
                    A_type: str = 'dct', estimate_Phi: bool = True) -> None:
        """
        iter_max should be the same as that of SE.
        A_type: type of sensing matrix: 'gaussian' or 'dct'
        estimate_Phi: - if True estimate Phi on the fly using AMP (i.e. by the
                       empirical covariance of the rows in Zt), this sometimes gives
                       diverging mse (i.e. mse first decreases then increases away from 0);
                     - if False use Tt from SE which user needs to provide to run().
        """
        self.log_file_name = log_file_name
        initialise_logger(logger, log_file_name)
        logger.debug('Initialising AMP-SC jax')
        self.W = W
        self.R, self.C = W.shape
        self.n = n
        self.L = L
        self.d = d
        assert self.L % self.C == 0, "L/C should be integer"
        assert self.n % self.R == 0, "n/R should be integer"
        self.M = int(n/self.R)
        self.N = int(L/self.C)
        self.δ_in = self.M/self.N
        self.denoiser = denoiser
        # denoiser is storing the signal prior:
        self.signal_matrix = SignalMatrix(self.L, denoiser.α, \
                            denoiser.codebook, denoiser.ldpc_code, self.d)
        self.signal_power = (1-denoiser.α) * np.eye(self.d)

        assert sigma2_W >= 0 and np.isscalar(sigma2_W)
        self.sigma2_W = sigma2_W
        assert iter_max >= 1 and num_trials >= 1
        self.iter_max, self.num_trials = iter_max, num_trials

        self.A_type = A_type
        self.estimate_Phi = estimate_Phi
        if self.denoiser.type == 'mmse-marginal':
            self._calc_B = self._calc_B_scalar_se # checked
            self._calc_Phi_inv_or_Phi = self._calc_Phi_scalar_se # checked
            self._calc_G_cov_Q = self._calc_G_cov_Q_scalar_se
            self._calc_Vt = self._calc_Vt_scalar_se # checked
        else:
            self._calc_B = self._calc_B_vector_se
            self._calc_Phi_inv_or_Phi = self._calc_Phi_inv_vector_se
            self._calc_G_cov_Q = self._calc_G_cov_Q_vector_se
            self._calc_Vt = self._calc_Vt_vector_se_pablo_my # self._calc_Vt_vector_se_pablo
        logger.debug('Finished initialising AMP-SC jax')


    def _one_iter(self, Y, Xt, Zt, Q, eta_prime_av_T, is_last_iter: bool = False):
        """
        Loops over C, R are short. Those over n, L, or M, N are long.
        Memory usage shouldnt be much worse than iid-AMP because the intermediate
        Rxdxd or Cxdxd matrices are quite small, comapared with nxL, nxd, Lxd or Lxdxd
        which are needed for iid-AMP as well.

        ====
        For mmse-marginal denoiser where SE parameters which define the denoiser
        are scalars, not matrices.

        Input and output arguments with dimensions different from the matrix case:
        Q is RxC instead of RxCxdxd;
        G_cov is length-C instead of Cxdxd.
        """
        logger.debug('Starting to create B')
        # time_start = time.time()
        B = self._calc_B(self.δ_in, self.W, Q, eta_prime_av_T)
        # logger.debug(f"Finished creating B in {time.time() - time_start} seconds")

        logger.debug('Starting to create Z_tilde')
        # time_start = time.time()
        Z_tilde = self._calc_Z_tilde(Zt, B)
        # logger.debug(f"Finished creating Z_tilde in {time.time() - time_start} seconds")
        Zt = Y - (self.A @ Xt) + Z_tilde # nxd

        # ==================== update parameters of denoising function η ====================
        # Calculate Φ from Z - approximation of SE
        logger.debug('Starting to create Phi_inv (vector SE) or Phi (scalar SE)')
        # time_start = time.time()
        # Phi and Phi_inv are both Rxdxd
        if not self.estimate_Phi and self.denoiser.type != 'mmse-marginal':
            Phi_inv_or_Phi = np.linalg.inv(self.Phi_t_arr[self.t]) # Phi_t_arr is iterxRxdxd, inverse is Rxdxd
            # TODO: above is Phi_inv. Is Phi needed for mmse-marginal?
        else:
            Phi_inv_or_Phi = self._calc_Phi_inv_or_Phi(self.R, Zt)
        # logger.debug(f"Finished creating Phi_inv of Phi in {time.time() - time_start} seconds")

        # Update Q from Φ:
        logger.debug('Starting to update G_cov, Q')
        # time_start = time.time()
        G_cov, Q = self._calc_G_cov_Q(self.W, Phi_inv_or_Phi)
        # logger.debug(f"Finished updating Q in {time.time() - time_start} seconds")

        logger.debug('Starting to update Vt (this step is the most time-consuming))')
        # time_start = time.time()
        Vt = self._calc_Vt(self.A, Zt, Q)
        # logger.debug(f"Finished updating Vt in {time.time() - time_start} seconds")

        logger.debug('Starting to update St')
        St = Xt + Vt # effective observation, Lxd
        assert not np.any(np.isnan(St))
        logger.debug('Starting to update etaJac, X_MAP, Xt, eta_prime_av_T')
        # eta_prime_av_T has shape Cxdxd
        X_MAP = np.zeros_like(Xt, dtype=np.float64)
        for c in range(self.C): # ηt is applied one column block at a time,
            # which corresponds to each set of rows in St at a time.
            # Each set of rows contain N rows.
            # η_sum_dη_XMAP returns η, sum_dη, X_MAP
            Xt[self.N*c:(self.N*c + self.N)], \
            sum_dη, X_MAP[self.N*c:(self.N*c + self.N)] = \
                self.denoiser.η_sum_dη_XMAP(St[self.N*c:(self.N*c + self.N)],
                    G_cov[c], calc_sum_dη=True, calcMAP_estimate=is_last_iter)
            # Above fills in N rows of Xt, N dxd matrices in etaJac.
            assert not np.any(np.isnan(sum_dη))
            eta_prime_av_T[c] = sum_dη.T / self.N # Average over N rows.
            # Above averages over each column block containing N elements.
        # Return Q, eta_prime_av_T for next iteration;
        # return G_cov for potential post-processing.
        logger.debug('Finished one iteration')
        return St, Xt, Zt, Q, eta_prime_av_T, X_MAP, G_cov

    def _one_trial(self):
        """
        For a fixed sensing matrix, run AMP for iter_max iterations.

        TODO: current version doesnt take in SE parameters Q or
        eta_prime_av_T over time. Compute these on the fly.
        """
        mse_t_arr = np.zeros((self.iter_max, self.C))

        logger.debug('Initialising one_trial')
        # Initialize and fix the sensing matrix for all trials:
        assert self.A_type.lower() == 'gaussian', \
            "Only gaussian A is supported for AMP-SC."
        self.A = design_mat_A_sc(self.n, self.L, self.W)
        self.AX, self.AY = None, None
        # if self.denoiser.ldpc_code is None:
        X0 = self.signal_matrix.sample()
        # else:
            # X0 = np.ones((self.L, self.d)) # all-zero codewords only
        nl_Y = self.A @ X0 if self.A_type == 'gaussian' else self.AX(X0)  # nxr
        Y = add_iid_noise(nl_Y, self.sigma2_W)
        Xt = np.zeros_like(X0, dtype=np.float64)  # Initial signal estimate
        # X0 can be integer, force Xt to allow floats to be saved.
        Zt = Y  # Initial residual

        # Initialise to all zeros:
        if self.denoiser.type == 'mmse-marginal':
            Q = np.zeros((self.R, self.C))
        else:
            Q = np.zeros((self.R, self.C, self.d, self.d))
        eta_prime_av_T = np.zeros((self.C, self.d, self.d)) # Average of eta
        # derivative, transposed
        # mse_t_arr[0] = np.mean((Xt - X0)**2)
        for c in range(self.C):
            mse_t_arr[0, c] = np.mean((Xt[self.N*c:(self.N*c + self.N)] - \
                                    X0[self.N*c:(self.N*c + self.N)]) ** 2)
        logger.debug(f'Initial mse at iteration 0: {mse_t_arr[0]}')
        # self.t = 0
        avg_var_c = np.zeros((self.iter_max, self.C))
        for t in tqdm(range(1, self.iter_max)):
            self.t = t
            _, Xt, Zt, Q, eta_prime_av_T, _, G_cov = \
                self._one_iter(Y, Xt, Zt, Q, eta_prime_av_T) #[1:-2]
            # self.t += 1
            # G_cov is Cxdxd, each dxd matrix is the covariance of
            # effective noise in column block c
            avg_var_c[t] = np.diagonal(G_cov, axis1=1, axis2=2).mean(axis=1)
            # mse_t_arr[t] = np.mean((Xt - X0)**2)
            for c in range(self.C):
                mse_t_arr[t, c] = np.mean((Xt[self.N*c:(self.N*c + self.N)] - \
                                        X0[self.N*c:(self.N*c + self.N)]) ** 2)
            logger.debug(f"mse at iteration {t}: {mse_t_arr[t].mean()}")
            if terminate(np.mean(mse_t_arr, axis=1), t, self.iter_max, MSE_RTOL_SC):
                mse_t_arr[t+1 :, :] = -1 # mse_t_arr[t]
                break
            logger.debug(f'Terminated/ converged after {t} iterations.' + \
                    f' mse upon convergence: {mse_t_arr[t].mean()}')
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
            plt.title(f'SC-AMP, C={self.C}, R={self.R}, L={self.L}, d={self.d}, denoiser={self.denoiser.type}')
            my_time = timestamp()
            plt.savefig('./results/sc_amp_noise_var_c_' + my_time + '.pdf')

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
        plt.title(f'SC-AMP, C={self.C},R={self.R},L={self.L},δ_in={self.δ_in},d={self.d},\n' + \
                  f'{self.denoiser.type}, est_Phi={self.estimate_Phi}')
        my_time = timestamp()
        plt.savefig('./results/sc_amp_mse_c_' + my_time + '.pdf')

        # Calc MAP estimate after AMP has converged:
        St, _, _, _, _, X_MAP, G_cov = \
            self._one_iter(Y, Xt, Zt, Q, eta_prime_av_T, is_last_iter=True)
        logger.debug('Finished one_trial')
        return X0, X_MAP, St, G_cov, mse_t_arr, t

    def run(self, Phi_t_arr: np.array = None, run_bp_post_amp: bool =False, \
            calc_pErrors: bool=False, \
            calcPUPE: bool=False, calcBER: bool=False):
        """Phi_t_arr is iter_max xRxdxd"""
        logger.debug(f"Running AMP-SC with {self.num_trials} trials and {self.iter_max} iterations")

        assert self.estimate_Phi == (Phi_t_arr is None), \
            "Phi_t_arr output from SE must be provided if estimate_Phi==False," + \
            " and vice versa"
        self.Phi_t_arr = Phi_t_arr
        mse_t_arr = np.zeros((self.num_trials, self.iter_max, self.C))
        # number of iterations needed for AMP to converge:
        iter_idx_converge_arr = np.zeros(self.num_trials)

        # By default calculate metrics using the MAP estimate:
        if calc_pErrors:
            # by default use MAP solution
            pMD_arr = np.zeros(self.num_trials)
            pFA_arr = np.zeros(self.num_trials)
            pAUE_arr = np.zeros(self.num_trials)
        else:
            pMD_arr, pFA_arr, pAUE_arr = None, None, None
        PUPE_arr = np.zeros(self.num_trials) if calcPUPE else None
        BER_arr = np.zeros(self.num_trials) if calcBER else None
        for i_trial in range(self.num_trials):
            logger.debug(f"Running AMP-SC trial {i_trial + 1}/{self.num_trials}")
            (X0, X_MAP, St, G_cov,
             mse_t_arr[i_trial, :], iter_idx_converge_arr[i_trial]
                ) = self._one_trial()
            ############# Post-processing ##############
            if run_bp_post_amp:
                logger.debug(f'mse before BP post AMP: {np.mean((X_MAP - X0)**2)}')
                logger.debug('Starting to run BP post AMP')
                if self.denoiser.type == 'mmse-marginal':
                    # G_cov is a length-C vector. For each column block c, we need
                    # a dxd covariance matrix with G_cov[c] along the diagonal.
                    # Same sigma2 for every entry in the length-d vector St_l:
                    σ2_or_σ2_arr_c = G_cov # length-C
                    St = np.array(St) # post_process_row expects St to be a np.array
                    # while St output from mmse-marginal denoiser is a traced Jax object.
                elif self.denoiser.type == 'ldpc-bp':
                    # G_cov is Cxdxd
                    σ2_or_σ2_arr_c = np.zeros((self.C, self.d))
                    for c in range(self.C):
                        σ2_or_σ2_arr_c[c] = np.diag(G_cov[c])
                else:
                    raise ValueError(f"denoiser.type must be mmse-marginal or ldpc-bp, not {self.denoiser.type}")
                X_MAP_post_BP = np.zeros_like(X0, dtype=np.float64)
                # X_MAP should be integers anyways but to be safe allow floats.
                # G_cov is Cxdxd when denoiser is not mmse-marginal, and is length-C
                # when it is. Each column block corresponds to several rows in St.
                for l in range(self.L):
                    X_MAP_post_BP[l, :] = post_process_row(
                        St[l,:], σ2_or_σ2_arr_c[l//self.N], self.denoiser.ldpc_code)
                X_MAP = X_MAP_post_BP # replace
                logger.debug('Finished running BP post AMP')
                logger.debug(f"mse after BP post AMP: {np.mean((X_MAP - X0)**2)}")
            ############################################
            if calc_pErrors:
                (_, _, _, _, pMD_arr[i_trial], pFA_arr[i_trial],
                    pAUE_arr[i_trial]) = count_errors(X0, np.array(X_MAP))
            if calcPUPE:
                PUPE_arr[i_trial] = calc_PUPE(X0, X_MAP)
                logger.debug(f"PUPE after convergence: {PUPE_arr[i_trial]}")
            if calcBER:
                BER_arr[i_trial] = np.mean(X_MAP != X0)
                logger.debug(f"BER after convergence: {BER_arr[i_trial]}")
        logger.debug(f"Finished AMP-SC")
        return mse_t_arr, iter_idx_converge_arr, pMD_arr, pFA_arr, pAUE_arr, \
            PUPE_arr, BER_arr


    @staticmethod
    @jit(nopython=True)
    def _calc_B_vector_se(δ_in, W, Q, eta_prime_av_T):
        """
        2x faster than the non-jitted, einsum version.
        B is Rxdxd, W is RxC, Q is RxCxdxd, eta_prime_av_T is Cxdxd.
        """
        R, C = W.shape
        d = eta_prime_av_T.shape[-1]
        # assert not np.any(np.isnan(eta_prime_av_T))
        B = np.zeros((R, d, d))
        for r in range(R): # loop through r, which is equivalent to looping through i
            # because multiple i's map to the same r.
            # Below is the term in Z̃t which is multiplied with Zt to give Z̃t:
            for c in range(C):
                # W is RxC, Q is RxCxdxd, eta_prime_av_T is Cxdxd, B[r] is dxd
                # tmp = (1/δ_in) * W[r,c] * Q[r,c] @ eta_prime_av_T[c]
                # assert not np.any(np.isnan(tmp))
                B[r] += (1/δ_in) * W[r,c] * Q[r,c] @ eta_prime_av_T[c]
            # B[r]= (1/δ_in)*np.einsum('i,ijk->jk', W[r,:],
            #         np.einsum('ijk,ikl->ijl', Q[r,:], eta_prime_av_T)) # B[r] is dxd
        # Z̃t_i = Zt_i @ B[r] is a len-d row vector multiplied with a dxd matrix
        # assert not np.any(np.isnan(B))
        return B

    @staticmethod
    @jit(nopython=True)
    def _calc_B_scalar_se(δ_in, W, Q, eta_prime_av_T):
        """B is Rxdxd, W is RxC, Q is RxC, eta_prime_av_T is Cxdxd."""
        R, C = W.shape
        d = eta_prime_av_T.shape[-1]
        B = np.zeros((R, d, d))
        for r in range(R): # loop through r, which is equivalent to looping through i
            # because multiple i's map to the same r.
            # Below is the term in Z̃t which is multiplied with Zt to give Z̃t:
            for c in range(C):
                # W is RxC, Q is RxC, eta_prime_av_T is Cxdxd, B[r] is dxd
                B[r] += (1/δ_in) * W[r,c] * Q[r,c] * eta_prime_av_T[c]
            # B[r]= (1/δ_in)*np.einsum('i,ijk->jk', W[r,:],
            #         np.einsum('ijk,ikl->ijl', Q[r,:], eta_prime_av_T)) # B[r] is dxd
        # Z̃t_i = Zt_i @ B[r] is a len-d row vector multiplied with a dxd matrix
        return B

    @staticmethod
    @jit(nopython=True)
    def _calc_Z_tilde(Zt, B):
        """
        10x faster than the non-jitted version.
        B is Rxdxd, Zt is nxd, Z_tilde is nxd.
        """
        n, d = Zt.shape
        R = B.shape[0]
        M = int(n/R)
        Z_tilde = np.zeros((n, d))
        for r in range(R): # R blocks, each block shares the same B[r],
            # so fill in one block at a time.
            Z_tilde[r*M:(r+1)*M] = Zt[r*M:(r+1)*M] @ B[r]
        return Z_tilde

    @staticmethod
    @jit(nopython=True)
    def _calc_Phi_inv_vector_se(R, Zt):
        """30x faster than the non-jitted version."""
        n, d = Zt.shape
        M = int(n/R)
        Phi_inv = np.zeros((R, d, d))
        # Phi_diag = np.zeros(self.R)
        for r in range(R): # calc Φ for each section of rows:
            Phi_r = np.cov(Zt[M*r:(M*r+M)], rowvar=False) # Phi[r] is dxd
            # print(f"Phi_r is {Phi_r}")
            Phi_inv[r] = np.linalg.pinv(Phi_r + np.eye(d) * 1e-12) # np.finfo(float).eps=1e-16
            # inv doesnt work because argument may be singular
            # Phi_diag[r] = np.average(np.diag(Phi[r]))
        return Phi_inv

    @staticmethod
    @jit(nopython=True)
    def _calc_Phi_scalar_se(R, Zt):
        n = Zt.shape[0]
        M = int(n/R)
        Phi = np.zeros(R)
        for r in range(R): # calc Φ for each section of rows:
            # default is to calculate the variance of the flattened array
            Phi[r] = np.var(Zt[M*r:(M*r+M)]) # scalar
        return Phi

    @staticmethod
    @jit(nopython=True)
    def _calc_Vt_vector_se(A, Zt, Q):
        """10x faster than the non-jitted version."""
        n, d = Zt.shape
        L = A.shape[1]
        R, C = Q.shape[0:2] # Q is RxCxdxd
        M = int(n/R)
        N = int(L/C)
        Vt = np.zeros((L, d)) # shape of X0
        for j in range(L):
            Z_Q = np.zeros((n, d))
            for r in range(R):
                Z_Q[r*M:(r+1)*M] = Zt[r*M:(r+1)*M] @ Q[r, j//N]
            Vt[j] = Z_Q.T @ A[:,j] # nxd matrix times length-n vector
            # Line above gives non-contiguous array warning but runs faster than below:
            # for i in range(n):
            #     Vt[j] += Z_Q[i] * A[i,j]
        return Vt

    @staticmethod
    @jit(nopython=True)
    def _calc_Vt_scalar_se(A, Zt, Q):
        """10x faster than the non-jitted version."""
        n, d = Zt.shape
        L = A.shape[1]
        R, C = Q.shape # Q is RxC
        M = int(n/R)
        N = int(L/C)
        Vt = np.zeros((L, d)) # shape of X0
        for j in range(L):
            Z_Q = np.zeros((n, d))
            for r in range(R):
                Z_Q[r*M:(r+1)*M] = Zt[r*M:(r+1)*M] * Q[r, j//N]
            Vt[j] = Z_Q.T @ A[:,j] # nxd matrix times length-n vector
            # Line above gives non-contiguous array warning but runs faster than below:
            # for i in range(n):
            #     Vt[j] += Z_Q[i] * A[i,j]
        return Vt


    # @staticmethod
    # @jit(nopython=True)
    def _calc_Vt_vector_se_pablo(self, A, Zt, Q):
        """
        Pablo's version. A is nxL, Zt is nxd, Q is RxCxdxd.
        For z=30, d=720, this fn takes about 15mins and is faster than _calc_Vt_vector_se.
        Bottleneck is the repeat step.
        """
        n, d = Zt.shape
        L = A.shape[1]
        R, C = Q.shape[0:2] # Q is RxCxdxd
        M = int(n/R)
        N = int(L/C)
        Vt = np.zeros((L, d)) # shape of X0
        # numba doesnt support repeat keyword axis, or einsum
        for c in range(C):
            print(f"c={c}, C={C}")
            time_start = time.time()
            Qc = np.repeat(Q[:,c], repeats=M, axis=0) # nxdxd (30s)
            print(f"Repeat done in {time.time() - time_start} seconds")
            time_start = time.time()
            Z_Q = np.einsum('ij,ijk->ik', Zt, Qc) # nxd (9s)
            print(f"einsum done in {time.time() - time_start} seconds")
            time_start = time.time()
            Vt[N*c:(N*c + N)] = (A[:, N*c:(N*c + N )]).T @ Z_Q # Nxn times nxd -> Nxd (1s)
            print(f"Vt_c done in {time.time() - time_start} seconds")
        return Vt

    @staticmethod
    @jit(nopython=True)
    def _calc_Vt_vector_se_pablo_my(A, Zt, Q):
        """
        Updated version of _calc_Vt_vector_se_pablo.
        A is nxL, Zt is nxd, Q is RxCxdxd.
        For z=30, d=720, this fn takes about 1min and is faster than
        _calc_Vt_vector_se and _calc_Vt_vector_se_pablo.
        """
        n, d = Zt.shape
        L = A.shape[1]
        R, C = Q.shape[0:2] # Q is RxCxdxd
        M = int(n/R)
        N = int(L/C)
        Vt = np.zeros((L, d)) # shape of X0
        # numba doesnt support repeat keyword axis, or einsum
        for c in range(C):
            # print(f"c={c}, C={C}")
            # time_start = time.time()
            Z_Q = np.zeros((n, d))
            for r in range(R):
                Z_Q[r*M:(r+1)*M] = Zt[r*M:(r+1)*M] @ Q[r, c] # Mxd times dxd -> Mxd
            # print(f"Z_Q computed in {time.time() - time_start} seconds")
            # time_start = time.time()
            Vt[N*c:(N*c + N)] = (A[:, N*c:(N*c + N )]).T @ Z_Q # Nxn times nxd -> Nxd
            # print(f"Vt_c done in {time.time() - time_start} seconds")
        return Vt


    @staticmethod
    @jit(nopython=True)
    def _calc_G_cov_Q_vector_se(W, Phi_inv):
        """
        Phi_inv is Rxdxd, Q is RxCxdxd, W is RxC, G_cov is Cxdxd.
        Despite the for loop, this is 20x faster than the non-jitted version.
        """
        d = Phi_inv.shape[-1]
        R, C = W.shape
        G_cov = np.zeros((C, d, d))
        Q = np.zeros((R, C, d, d))
        for c in range(C):
            # G_cov[c] is the term in Q_rc, which is multiplied to Φ_inv to give Q_rc.
            # It is also the covariance matrix of the Gaussian noise at step t for column block c.
            # G_cov[c] = np.linalg.pinv(np.einsum('i,ijk->jk', W[:,c], Phi_inv)) # P[c] is dxd
            # G_cov[c] = np.linalg.pinv(np.dot(Phi_inv.T, W[:,c]).T) # doesnt work with numba
            # above: inv doesnt work because argument may be singular
            tmp_c = np.zeros((d, d))
            for r in range(R):
                tmp_c += W[r,c] * Phi_inv[r]
            # Line below rarely gives "Internal algorithm failed to converge" error:
            G_cov[c] = np.linalg.pinv(tmp_c) + np.eye(d) * 1e-12 # this is sometimes singular
            # G_cov[c] should be positive definite, but sometimes it is not:
            assert np.all(np.linalg.eigvals(G_cov[c]) > 0), \
                "G_cov[c] isnt positive definite. Possible reason: " + \
                "L,n too small such that the G_cov[c] estimate via avgs is inaccurate."
            for r in range(R):
                Q[r,c] = Phi_inv[r] @ G_cov[c] # Q[r,c] is dxd
        return G_cov, Q

    @staticmethod
    @jit(nopython=True)
    def _calc_G_cov_Q_scalar_se(W, Phi):
        C = W.shape[1]
        G_cov = np.zeros(C)
        for c in range(C):
            G_cov[c] = 1/np.sum(W[:,c]/Phi)
        # Q = 1/Phi.reshape(-1, 1) * G_cov.reshape(1, -1) # Q is RxC
        Q = np.outer(1/Phi, G_cov) # Q is RxC
        return G_cov, Q


def post_process_row(St_l, σ2_arr, ldpc_code):
    """
    Post AMP processing each row of St, effective observation,
    by running BP to recover the ldpc code in that row.
    Note St should be a numpy array instead of a traced Jax object.

    σ2_arr can be a scalar or a length-d vector. When it is scalar,
    use the same σ2 for all d entries in St_l.
    """
    assert St_l.shape == np.array(σ2_arr).shape or \
        np.array(σ2_arr).size == 1
    llr = ch2llr(St_l, σ2_arr)
    # (app,_) = ldpc_code.decode(llr, JOSSY_MAX_ITCOUNT)
    (app,_) = decode(ldpc_code.vdeg, ldpc_code.cdeg,
                        ldpc_code.intrlv, ldpc_code.Nv,
                        ldpc_code.Nc, ldpc_code.Nmsg, llr,
                        JOSSY_MAX_ITCOUNT)
    MAP_est_l = bpsk(app < 0.0)
    return MAP_est_l # length-d vector
