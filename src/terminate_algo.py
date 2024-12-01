from typing import Tuple
import numpy as np
from numba import jit

@jit(nopython=True)
def terminate(mse_t_arr, t, iter_max, mse_rtol=1e-4):
    # mse_rtol = 1e-4
    mse_atol = 1e-10
    if t<=1 or t >= iter_max-1:
        return False
    elif mse_t_arr[t] < mse_atol:
        return True
    else:
        return (np.abs(mse_t_arr[t] - mse_t_arr[t-1]) \
                / np.abs(mse_t_arr[t])) < mse_rtol

def calc_PUPE(X: np.array, Xest: np.array) -> float:
    """Per-user probability of error: 1/L\sum_{l\in[L]} p(xl \neq xl_hat)"""
    assert X.shape == Xest.shape, "X and Xest must have the same shape" # Lxr
    L = X.shape[0]
    return 1 / L * np.sum(np.sum(X != Xest, axis=1) > 0) # check row-wise

def PUPE_d1_to_d_general(PUPE_d1, d):
    """
    Probability of error for uncoded systems
    PUPE_d1: probability of error for d=1
    """
    return 1 - (1 - PUPE_d1)**d


def PUPE_d_general_to_d1(PUPE_d, d):
    """
    PUPE_d: probability of error for d
    """
    return 1 - (1 - PUPE_d)**(1/d)

@jit(nopython=True)
def calc_pMD_FA_AUE(X, Xest, quantise: bool=False, thres_arr: np.array=None) \
    -> Tuple[np.array, np.array, np.array]:
    """
    If quantise is True, then Xest is thresholded according to thres_list
    and quantised row-wise before calculating the probabilities of error.

    NOTE: list will soon be deprecated by numba so use thres_arr as np.array instead.
    NOTE: numba requires consistent output type from if and else: it cannot be arrays
    from one and scalars from the other.
    """
    assert X.shape == Xest.shape, "X and Xest must have the same shape"
    if quantise:
        assert thres_arr is not None, "Must provide an array of thresholds"
        num_thres = len(thres_arr)
        pMD_arr = np.zeros(num_thres)
        pFA_arr = np.zeros(num_thres)
        pAUE_arr = np.zeros(num_thres)
        for i_thres, thres in enumerate(thres_arr):
            Xest_quantised = rowwise_quant(Xest, thres)
            _, _, _, _, pMD_arr[i_thres], pFA_arr[i_thres], pAUE_arr[i_thres] = \
                count_errors(X, Xest_quantised)
        return pMD_arr, pFA_arr, pAUE_arr
    else:
        _, _, _, _, pMD, pFA, pAUE = count_errors(X, Xest)
        return np.array([pMD]), np.array([pFA]), np.array([pAUE])




# TODO: make a class called quantizer; move metrics into a separate file
@jit(nopython=True)
def entrywise_quant(X_amp, thres):
    """
    X_amp: Lxd signal estimate returned by the AMP algorithm
           after the algorithm has converged
    thres: positive constant

    This function checks each entry of X_amp:
    if entry>thres, declare +1; if entry<-thres, declare -1;
    otherwise declare 0

    X_est: output from the quantisation procedure described above
    """
    assert thres > 0, "Theshold must be positive"
    L, d = np.shape(X_amp)
    X_est = np.zeros_like(X_amp)
    X_est[X_amp > thres] = +1
    X_est[X_amp < -thres] = -1
    X_est[(-thres <= X_amp) & (X_amp <= thres)] = 0
    assert np.shape(X_est) == (L, d)
    return X_est

@jit(nopython=True)
def rowwise_quant(X_amp: np.array([[]]), thres: float) -> np.array:
    """
    X_amp: Lxd signal estimate returned by the AMP algorithm
           after the algorithm has converged
    thres: >0, a row-wise energy threshold

    For each row Xi of X_amp:
    if (l2-norm squared of Xi) < thres squared * d, declare an all-zero row;
    otherwise declare +/-1 entrywise.

    X_est: output from the quantisation procedure described above.
           guaranteed to be a valid codeword.
    """
    assert thres > 0, "Theshold must be positive"
    L, d = np.shape(X_amp)
    X_est = np.zeros_like(X_amp)
    exceeds_thres = np.sum(X_amp**2, axis=1) > thres**2 * d
    num_nz_rows = np.sum(exceeds_thres)
    X_est[exceeds_thres, :] = np.sign(X_amp[exceeds_thres, :])
    num_zeros_in_nz_rows = np.sum(X_est[exceeds_thres, :] == 0)
    if num_zeros_in_nz_rows > 0:
        # Nonzero rows contain zeros, assign +/-1 randomly
        active_X_est = X_est[exceeds_thres, :]
        # active_X_est[active_X_est == 0] = np.random.choice(
        #     [-1, 1], size=num_zeros_in_nz_rows
        # ) # not supported by numba
        # Numba does not cuurently support >1D bool array indexing:
        is_zero = active_X_est == 0
        active_X_est.ravel()[is_zero.ravel()] = (-1) ** (np.random.rand(num_zeros_in_nz_rows) > 0.5)
        X_est[exceeds_thres, :] = active_X_est
    assert X_est.shape == (L, d)
    assert np.sum(np.sum(X_est == 0, axis=1) == d) == L - num_nz_rows  # all-zero rows
    assert np.sum(np.sum(X_est != 0, axis=1) == d) == num_nz_rows  # nonzero rows
    return X_est

@jit(nopython=True)
def count_errors(X: np.array, X_est: np.array) -> Tuple[int, int, int, int, float, float, float]:
    """
    X, X_est: Lxd matrices
    Count number of missed detections (MD), false alarms (FA) and
    active user errors (AUE) in X_est against the ground truth X.
    """
    L, d = np.shape(X)
    assert np.shape(X) == np.shape(X_est)
    user_is_silent = np.sum(X != 0, axis=1) == 0
    user_is_active = np.logical_not(user_is_silent)
    num_active_users = np.sum(user_is_active)
    user_declared_silent = np.sum(X != 0, 1) == 0
    num_users_declared_active = L - np.sum(user_declared_silent)
    # User whose row is decoded correctly (incl active recovered as active
    # and silent recovered as silent):
    num_correct_users = np.sum(np.sum(X == X_est, axis=1) == d)  # sum along each row
    # Silent user whose decoded row contains nonzeros:
    num_FAs = np.sum(np.sum(X_est[user_is_silent, :] != 0, axis=1) > 0)
    # Active user whose decoded row is all-zero:
    num_MDs = np.sum(np.sum(X_est[user_is_active, :] == 0, axis=1) == d)
    # Active user whose decoded row contains nonzero and doesnt match the
    # transmitted row:
    row_contains_nonzeros = np.sum(X_est[user_is_active, :] != 0, axis=1) > 0
    row_doesnt_match = (
        np.sum(X_est[user_is_active, :] == X[user_is_active, :], axis=1) < d
    )
    num_AUEs = np.sum(row_contains_nonzeros & row_doesnt_match)
    assert num_correct_users + num_FAs + num_MDs + num_AUEs == L
    pMD = (
        num_MDs / num_active_users if num_active_users > 0 else 0
    )  # indicator function
    pFA = num_FAs / num_users_declared_active if num_users_declared_active > 0 else 0
    pAUE = num_AUEs / num_active_users if num_active_users > 0 else 0
    return num_correct_users, num_MDs, num_FAs, num_AUEs, pMD, pFA, pAUE
