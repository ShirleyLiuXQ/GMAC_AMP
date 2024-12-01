import logging
import numpy as np
from src.amp_jax import AMP_jax, AMP_SC_jax
from src.denoiser_jax import Denoiser_jax
from src.helper_fns import initialise_logger

from src.se_jax import SE_jax, SE_SC_jax
from src.terminate_algo import calc_PUPE

logger = logging.getLogger(__name__)

def binary_search(σ2_arr, target_error, f_σ2_to_error, log_file_name) \
    -> (int, float, float):
    """
    Perform binary search on a monotonically increasing array σ2_arr
    for fixed S.
    Ensure σ2_arr contains a fine grid, because this function returns
    the σ2 in σ2_arr that gives error closest to and smaller than
    target error.

    f_σ2_to_error: function that takes in σ2 and returns error (e.g. BER, epsTotal).
                 It is monotonically increasing in σ2
                 (higher noise -> lower SNR -> higher error).

    Returns:
    Index, value pair of σ2 that gives error closest to and smaller than
    target error.
    Also returns the error at the returned σ2.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered binary_search")
    assert np.all(σ2_arr == np.sort(σ2_arr)), \
        "σ2_arr must be monotonically increasing"
    low_idx, high_idx = 0, len(σ2_arr) - 1

    low_error = f_σ2_to_error(σ2_arr[low_idx]) # lower σ2, lower error
    high_error = f_σ2_to_error(σ2_arr[high_idx]) # higher σ2, higher error
    logger.debug(f"low_error={low_error}, high_error={high_error}")
    if target_error < low_error:
        logger.debug(f"low_error (max SNR)={low_error}, target_error={target_error}")
        logger.debug("No σ2 in range gives error < target error")
        return -1, -1, -1
    elif target_error >= high_error:
        logger.debug(f"high_error (min SNR)={high_error}, target_error={target_error}")
        logger.debug("All σ2 in range give error < target error")
        return high_idx, σ2_arr[high_idx], high_error
    else:
        logger.debug("There exists σ2 in range that gives error < target error")
        error_arr = -np.ones(len(σ2_arr)) # -1 means not yet calculated
        error_arr[low_idx] = low_error
        error_arr[high_idx] = high_error
        # When filled in, error_arr is monotonically increasing

        while low_idx <= high_idx:
            mid_idx = (low_idx + high_idx) // 2
            mid_σ2 = σ2_arr[mid_idx]
            mid_error = f_σ2_to_error(mid_σ2)
            error_arr[mid_idx] = mid_error

            if mid_error == target_error:
                logger.debug("Found exact match for target error: " + \
                             f"mid_error=target_error={target_error}")
                return mid_idx, mid_σ2, mid_error
            elif mid_error < target_error:
                if error_arr[mid_idx+1] > target_error:
                    # target_error lies between two adjacent EbN0s,
                    # return the EbN0_dB that gives error below target
                    logger.debug("Found approximate match for target error: \n" + \
                                 f"mid_error={mid_error}, " + \
                                 f"target_error={target_error} \n" + \
                                 f"error after mid_error={error_arr[mid_idx+1]}")
                    return mid_idx, mid_σ2, mid_error
                else:
                    low_idx = mid_idx +1 # Search the right half
            else:
                if 0 <= error_arr[mid_idx-1] < target_error:
                    # target_error lies between two adjacent EbN0s,
                    # return the EbN0_dB that gives error below target
                    logger.debug("Found approximate match for target error: \n" + \
                        f"error_arr[mid_idx-1]={error_arr[mid_idx-1]}, " + \
                        f"target_error={target_error} \n" + \
                        f"error before error_arr[mid_idx-1] i.e. mid_error={mid_error}")
                    return mid_idx-1, σ2_arr[mid_idx-1], error_arr[mid_idx-1]
                else:
                    high_idx = mid_idx -1 # Search the left half
    assert False, "Should not reach here"


def binary_search_EbN0dB_error_tol(low_EbN0dB, high_EbN0dB,
    target_error, error_tol, f_EbN0dB_to_error, log_file_name) \
    -> (float, float):
    """
    Perform binary search on the range [low_EbN0dB, high_EbN0dB], without
    predefining a grid of EbN0dB values, for the noise level leading to target_error.

    Search on EbN0dB instead of σ2 because we will plot EbN0dB on the x-axis,
    and the scaling between EbN0dB and σ2 is not linear.

    error_tol: stop search when the error is in [target_error - error_tol, target_error]

    f_EbN0dB_to_error: function that takes in EbN0dB and returns error
    (e.g. BER, epsTotal). It is monotonically decreasing in EbN0dB:
    (higher SNR -> lower noise -> lower error).

    Returns:
    σ2* that gives error* in [target_error - error_tol, target_error] and error*.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered binary_search_error_tol")
    assert low_EbN0dB < high_EbN0dB, "low_EbN0dB should be < high_EbN0dB"

    high_error = f_EbN0dB_to_error(low_EbN0dB) # lower SNR, higher error
    low_error = f_EbN0dB_to_error(high_EbN0dB) # higher SNR, lower error

    EbN0dB_list = [low_EbN0dB, high_EbN0dB]
    error_list = [high_error, low_error]
    if target_error < low_error:
        logger.debug("No EbN0dB in range gives error < target error")
        return -1, -1
    elif target_error >= high_error:
        logger.debug("All EbN0dB in range give error < target error")
        return low_EbN0dB, high_error
    else:
        logger.debug("There exists EbN0dB in range that gives error < target error")
        i_iter = 0
        while high_error >= low_error:
            mid_EbN0dB = (low_EbN0dB + high_EbN0dB) / 2
            mid_error = f_EbN0dB_to_error(mid_EbN0dB)
            EbN0dB_list.append(mid_EbN0dB)
            error_list.append(mid_error)
            # Due to the interplay between three types of errors, ascending EbN0dB
            # does not necessarily lead to descending error, but the vast majority
            # of the time it does.
            idx = np.argsort(EbN0dB_list) # indices that sort EbN0dB_list in ascending order
            error_list_sorted = np.array(error_list)[idx] # ascending EbN0dB -> descending error
            if np.all(np.diff(error_list_sorted) <= 0):
                logger.debug("error is monotonically decreasing with EbN0")
            else:
                logger.warn("error should be mostly monotonically decreasing with EbN0 but it isnt")
                logger.warn(f"error_list_sorted = {error_list_sorted}")

            if mid_error <= target_error and mid_error >= target_error - error_tol:
                logger.debug(f"binary_search_error_tol converged after {i_iter} iterations\n")
                logger.debug("Found approximate match for target error: \n" + \
                             f"mid_error={mid_error}, " + \
                             f"target_error={target_error}, " + \
                             f"error_tol={error_tol}")
                return mid_EbN0dB, mid_error
            elif mid_error < target_error:
                high_EbN0dB = mid_EbN0dB # Search the left half
            else:
                low_EbN0dB = mid_EbN0dB # Search the right half

            i_iter += 1
        assert False, "Should not reach here"


def find_critical_σ2_SE(δ_arr, d, σ2_arr, PUPE, log_file_name, \
    iter_max, num_G_samples, denoiser: Denoiser_jax, \
    W, δ_in_f: callable, iter_max_sc, num_G_samples_sc, \
        num_X0_samples: int=None, num_X0_samples_sc: int=None):
    initialise_logger(logger, log_file_name)

    assert np.all(δ_arr == np.sort(δ_arr)[::-1]), \
        "S_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    num_δ = len(δ_arr) # vary S via δ, (k, d are fixed)
    num_σ2 = len(σ2_arr)
    mse_arr = np.zeros((num_δ, num_σ2, iter_max))
    PUPE_arr = np.zeros((num_δ, num_σ2))
    BER_arr = np.zeros((num_δ, num_σ2))

    mse_arr_sc = np.zeros((num_δ, num_σ2, iter_max_sc))
    PUPE_arr_sc = np.zeros((num_δ, num_σ2))
    BER_arr_sc = np.zeros((num_δ, num_σ2))

    # Minimum σ2 needed for each given μ to achieve UER < target PUPE:
    critical_σ2_arr = -np.ones((num_δ, 2)) # 2 for iid and SC
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, δ in enumerate(δ_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        for i_σ2 in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2 [{i_σ2}/{num_σ2}] ==")
            σ2 = σ2_arr[i_σ2]
            logger.debug("Running iid SE...")
            se = SE_jax(δ, σ2, d, denoiser, log_file_name, iter_max, \
                    num_G_samples, num_X0_samples)
            _, mse_arr[i_δ, i_σ2], Tt_arr_uncoded = se.run()
            X0_uncoded, _, X_MAP_uncoded = se.last_iter_mc(Tt_arr_uncoded[-1])
            PUPE_arr[i_δ, i_σ2] = calc_PUPE(X0_uncoded, X_MAP_uncoded)
            BER_arr[i_δ, i_σ2] = np.mean(X_MAP_uncoded != X0_uncoded)

            logger.debug("Running SC SE...")
            δ_in = δ_in_f(δ)
            se_sc = SE_SC_jax(W, δ_in, σ2, d, denoiser, log_file_name,
                            iter_max_sc, num_G_samples_sc, num_X0_samples_sc)
            _, _, mse_arr_sc[i_δ, i_σ2], Pt_sc = se_sc.run()
            # NOTE: Pablo's Pt_sc is the Pt of the last iteration, not across all iterations.
            X0_sc, _, X_MAP_sc = se_sc.last_iter_mc(Pt_sc)
            PUPE_arr_sc[i_δ, i_σ2] = calc_PUPE(X0_sc, X_MAP_sc)
            BER_arr_sc[i_δ, i_σ2] = np.mean(X_MAP_sc != X0_sc)

            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i_δ, 0] == -1) and (PUPE_arr[i_δ, i_σ2] < PUPE):
                critical_σ2_arr[i_δ, 0] = σ2
            if (critical_σ2_arr[i_δ, 1] == -1) and (PUPE_arr_sc[i_δ, i_σ2] < PUPE):
                critical_σ2_arr[i_δ, 1] = σ2
            # Break the loop over σ2 if both are filled in:
            if np.all(critical_σ2_arr[i_δ] != -1):
                logger.debug(
                    f"critial σ2 for iid = {critical_σ2_arr[i_δ, 0]}\n"
                    + f"critial σ2 for sc = {critical_σ2_arr[i_δ, 1]}\n"
                )
                logger.debug(f"Pe_iid = {PUPE_arr[i_δ, i_σ2] }\n" + \
                      f"Pe_sc = {PUPE_arr_sc[i_δ, i_σ2]}\n")
                break
        # For next μ, scan σ2_arr from SC critical σ2 onwards for current μ:
        if critical_σ2_arr[i_δ, 1] != -1:
            start_σ2_idx = np.where(σ2_arr == critical_σ2_arr[i_δ, 1])[0][0]
            # When σ2 is fixed, above ensures next round still scans the
            # singleton array.
        else:
            break  # both iid and SC results wont show anything on the plot after this point

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2
    return mse_arr, mse_arr_sc, PUPE_arr, PUPE_arr_sc, \
        BER_arr, BER_arr_sc, critical_σ2_arr


def find_critical_σ2_iid_SE(δ_arr, d, σ2_arr, PUPE, log_file_name, \
    iter_max, num_G_samples, denoiser: Denoiser_jax, \
    W, δ_in_f: callable, iter_max_sc, num_G_samples_sc, \
        num_X0_samples: int=None, num_X0_samples_sc: int=None):
    initialise_logger(logger, log_file_name)

    assert np.all(δ_arr == np.sort(δ_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n is decreasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    num_δ = len(δ_arr) # vary S via δ, (k, d are fixed)
    num_σ2 = len(σ2_arr)
    mse_arr = np.zeros((num_δ, num_σ2, iter_max))
    PUPE_arr = np.zeros((num_δ, num_σ2))
    BER_arr = np.zeros((num_δ, num_σ2))

    mse_arr_sc = np.zeros((num_δ, num_σ2, iter_max_sc))
    PUPE_arr_sc = np.zeros((num_δ, num_σ2))
    BER_arr_sc = np.zeros((num_δ, num_σ2))

    # Minimum σ2 needed for each given μ to achieve UER < target PUPE:
    critical_σ2_arr = -np.ones((num_δ, 2)) # 2 for iid and SC
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, δ in enumerate(δ_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        for i_σ2 in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2 [{i_σ2}/{num_σ2}] ==")
            σ2 = σ2_arr[i_σ2]
            logger.debug("Running iid SE...")
            se = SE_jax(δ, σ2, d, denoiser, log_file_name, iter_max, \
                    num_G_samples, num_X0_samples)
            _, mse_arr[i_δ, i_σ2], Tt_arr_uncoded = se.run()
            X0_uncoded, _, X_MAP_uncoded = se.last_iter_mc(Tt_arr_uncoded[-1])
            PUPE_arr[i_δ, i_σ2] = calc_PUPE(X0_uncoded, X_MAP_uncoded)
            BER_arr[i_δ, i_σ2] = np.mean(X_MAP_uncoded != X0_uncoded)

            logger.debug("Running SC SE...")
            δ_in = δ_in_f(δ)
            se_sc = SE_SC_jax(W, δ_in, σ2, d, denoiser, log_file_name,
                            iter_max_sc, num_G_samples_sc, num_X0_samples_sc)
            _, _, mse_arr_sc[i_δ, i_σ2], Pt_sc = se_sc.run()
            # NOTE: Pablo's Pt_sc is the Pt of the last iteration, not across all iterations.
            X0_sc, _, X_MAP_sc = se_sc.last_iter_mc(Pt_sc)
            PUPE_arr_sc[i_δ, i_σ2] = calc_PUPE(X0_sc, X_MAP_sc)
            BER_arr_sc[i_δ, i_σ2] = np.mean(X_MAP_sc != X0_sc)

            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i_δ, 0] == -1) and (PUPE_arr[i_δ, i_σ2] < PUPE):
                critical_σ2_arr[i_δ, 0] = σ2
            if (critical_σ2_arr[i_δ, 1] == -1) and (PUPE_arr_sc[i_δ, i_σ2] < PUPE):
                critical_σ2_arr[i_δ, 1] = σ2
            # Break the loop over σ2 if both are filled in:
            if np.all(critical_σ2_arr[i_δ] != -1):
                logger.debug(
                    f"critial σ2 for iid = {critical_σ2_arr[i_δ, 0]}\n"
                    + f"critial σ2 for sc = {critical_σ2_arr[i_δ, 1]}\n"
                )
                logger.debug(f"Pe_iid = {PUPE_arr[i_δ, i_σ2] }\n" + \
                      f"Pe_sc = {PUPE_arr_sc[i_δ, i_σ2]}\n")
                break
        # For next μ, scan σ2_arr from SC critical σ2 onwards for current μ:
        if critical_σ2_arr[i_δ, 1] != -1:
            start_σ2_idx = np.where(σ2_arr == critical_σ2_arr[i_δ, 1])[0][0]
            # When σ2 is fixed, above ensures next round still scans the
            # singleton array.
        else:
            break  # both iid and SC results wont show anything on the plot after this point

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2
    return mse_arr, mse_arr_sc, PUPE_arr, PUPE_arr_sc, \
        BER_arr, BER_arr_sc, critical_σ2_arr


def find_critical_σ2_AMP(L, n_arr, d, σ2_arr, PUPE, log_file_name, \
    denoiser: Denoiser_jax, iter_max, W, iter_max_sc, \
        num_trials, A_type, estimate_Tt):
    """
    AMP is not dimensionless, so we fix L, vary n according to δ_arr.

    Register the minimum Eb/N0 needed for average Pe across trials
    to be < target PUPE.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered find_critical_σ2_AMP")
    assert np.all(n_arr == np.sort(n_arr)[::-1]), \
        "S_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    logger.debug(f"Creating empty result arrays")
    num_δ = len(n_arr) # vary S via δ, (k, d are fixed)
    num_σ2 = len(σ2_arr)
    mse_arr = np.zeros((num_δ, num_σ2, num_trials, iter_max))
    num_iter_converge = np.zeros((num_δ, num_σ2, num_trials))
    PUPE_arr = np.zeros((num_δ, num_σ2, num_trials))
    BER_arr = np.zeros((num_δ, num_σ2, num_trials))

    mse_arr_sc = np.zeros((num_δ, num_σ2, num_trials, iter_max_sc))
    num_iter_converge_sc = np.zeros((num_δ, num_σ2, num_trials))
    PUPE_arr_sc = np.zeros((num_δ, num_σ2, num_trials))
    BER_arr_sc = np.zeros((num_δ, num_σ2, num_trials))
    logger.debug("Finished creating empty result arrays")
    R, C = W.shape
    assert L % C == 0 # L/C should be integer

    # Minimum σ2 needed for each given μ to achieve UER < target PUPE:
    # PUPE is averaged across AMP trials.
    critical_σ2_arr = -np.ones((num_δ, 2)) # 2 for iid and SC
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, n in enumerate(n_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        assert n % R == 0  # n/R should be integer
        for i_σ2 in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2 [{i_σ2}/{num_σ2}] ==")
            σ2 = σ2_arr[i_σ2]
            logger.debug("Running iid AMP...")
            amp = AMP_jax(n, L, d, σ2, denoiser, log_file_name, iter_max,
                          num_trials, A_type, estimate_Tt)
            mse_arr[i_δ, i_σ2], num_iter_converge[i_δ, i_σ2], _,_,_, \
                PUPE_arr[i_δ, i_σ2], BER_arr[i_δ, i_σ2] = \
                    amp.run(run_bp_post_amp=True, calcPUPE=True, calcBER=True)

            logger.debug("Running SC AMP...")
            amp_sc = AMP_SC_jax(W, n, L, d, σ2, denoiser, log_file_name, \
                            iter_max_sc, num_trials, A_type, estimate_Tt)
            mse_arr_sc[i_δ, i_σ2], num_iter_converge_sc[i_δ, i_σ2], _,_,_, \
                PUPE_arr_sc[i_δ, i_σ2], BER_arr_sc[i_δ, i_σ2] = \
                    amp_sc.run(run_bp_post_amp=True, calcPUPE=True, calcBER=True)

            # If critical σ2 hasn't been filled in, and
            # average Pe across trials < target PUPE, fill it in:
            if (critical_σ2_arr[i_δ, 0] == -1) and \
                (np.mean(PUPE_arr[i_δ, i_σ2]) < PUPE):
                critical_σ2_arr[i_δ, 0] = σ2
            if (critical_σ2_arr[i_δ, 1] == -1) and \
                (np.mean(PUPE_arr_sc[i_δ, i_σ2]) < PUPE):
                critical_σ2_arr[i_δ, 1] = σ2
            # Break the loop over σ2 if both are filled in:
            if np.all(critical_σ2_arr[i_δ] != -1):
                logger.debug(
                    f"critial σ2 for iid = {critical_σ2_arr[i_δ, 0]}\n"
                    + f"critial σ2 for sc = {critical_σ2_arr[i_δ, 1]}\n"
                )
                logger.debug(f"Pe_iid = {np.mean(PUPE_arr[i_δ, i_σ2])}\n" + \
                      f"Pe_sc = {np.mean(PUPE_arr_sc[i_δ, i_σ2])}\n")
                break
        # For next μ, scan σ2_arr from SC critical σ2 onwards for current μ:
        if critical_σ2_arr[i_δ, 1] != -1:
            start_σ2_idx = np.where(σ2_arr == critical_σ2_arr[i_δ, 1])[0][0]
            # When σ2 is fixed, above ensures next round still scans the
            # singleton array.
        else:
            break  # both iid and SC results wont show anything on the plot after this point

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2
    return mse_arr, mse_arr_sc, PUPE_arr, PUPE_arr_sc, \
        BER_arr, BER_arr_sc, num_iter_converge, num_iter_converge_sc, \
            critical_σ2_arr



def find_critical_σ2_iid_AMP(L, n_arr, d, σ2_arr, BER, log_file_name, \
    denoiser: Denoiser_jax, iter_max, \
        num_trials, A_type, estimate_Tt):
    """
    AMP is not dimensionless, so we fix L, vary n according to δ_arr.

    Register the minimum Eb/N0 needed for average BER across trials
    to be < target BER.
    NOTE: since d differs for different coding schemes, BER is a fairer
    metric than PUPE.

    To run AMP with BP denoiser, define denoiser as 'ldpc-bp'.
    To run AMP with mmse-marginal denoiser, define denoiser as 'mmse-marginal'.
    BP postprocessing is always run after AMP.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered find_critical_σ2_iid_AMP")
    assert np.all(n_arr == np.sort(n_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n is decreasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    logger.debug(f"Creating empty result arrays")
    num_δ = len(n_arr) # vary S via δ, (k, d are fixed)
    S_arr = L / n_arr * denoiser.ldpc_code.K/denoiser.ldpc_code.N
    num_σ2 = len(σ2_arr)
    E = 1
    EbN0_arr = E / (2*σ2_arr) * denoiser.ldpc_code.N/denoiser.ldpc_code.K
    EbN0_dB_arr = 10*np.log10(EbN0_arr)
    mse_arr = np.zeros((num_δ, num_σ2, num_trials, iter_max))
    num_iter_converge = np.zeros((num_δ, num_σ2, num_trials))
    PUPE_arr = np.zeros((num_δ, num_σ2, num_trials))
    BER_arr = np.zeros((num_δ, num_σ2, num_trials))

    logger.debug("Finished creating empty result arrays")

    # Minimum σ2 needed for each given μ to achieve BER < target BER:
    # BER is averaged across AMP trials.
    critical_σ2_arr = -np.ones(num_δ)
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, n in enumerate(n_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}], S = {S_arr[i_δ]}=====")
        for i_σ2 in range(start_σ2_idx, num_σ2):
            σ2 = σ2_arr[i_σ2]
            logger.debug(f"== σ2={σ2} [{i_σ2}/{num_σ2}], EbN0_dB = {EbN0_dB_arr[i_σ2]} ==")
            logger.debug("Running iid AMP...")
            amp = AMP_jax(n, L, d, σ2, denoiser, log_file_name,
                          iter_max, num_trials, A_type, estimate_Tt)
            mse_arr[i_δ, i_σ2], num_iter_converge[i_δ, i_σ2], _,_,_, \
                PUPE_arr[i_δ, i_σ2], BER_arr[i_δ, i_σ2] = \
                    amp.run(run_bp_post_amp=True, calcPUPE=True, calcBER=True)

            # If critical σ2 hasn't been filled in, and
            # average BER across trials < target BER, fill it in:
            if (critical_σ2_arr[i_δ] == -1) and \
                (np.mean(BER_arr[i_δ, i_σ2]) < BER):
                critical_σ2_arr[i_δ] = σ2

            # Break the loop over σ2 if critical σ2 has been filled in:
            if critical_σ2_arr[i_δ] != -1:
                logger.debug(f"critial σ2 for iid = {critical_σ2_arr[i_δ]}\n")
                logger.debug(f"BER_iid = {np.mean(BER_arr[i_δ, i_σ2])}\n")
                break
        # For next μ, scan σ2_arr from critical σ2 onwards for current μ:
        if critical_σ2_arr[i_δ] != -1:
            start_σ2_idx = np.where(σ2_arr == critical_σ2_arr[i_δ])[0][0]
            # When σ2 is fixed, above ensures next round still scans the
            # singleton array.
        else:
            # Need higher EbN0 than provided range to achieve target BER for
            # given S, so cannot achieve target BER for any higher S either,
            # so break the loop over S.
            break  # results wont show anything on the plot after this point

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2
    return mse_arr, PUPE_arr, BER_arr, num_iter_converge, critical_σ2_arr


def binary_search_critical_σ2_iid_AMP(L, n_arr, d, σ2_arr, BER, log_file_name, \
    denoiser: Denoiser_jax, iter_max, \
        num_trials, A_type, estimate_Tt):
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered find_critical_σ2_iid_AMP")
    assert np.all(n_arr == np.sort(n_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n is decreasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    logger.debug(f"Creating empty result arrays")
    num_δ = len(n_arr) # vary S via δ, (k, d are fixed)
    S_arr = L / n_arr * denoiser.ldpc_code.K/denoiser.ldpc_code.N

    def f_σ2_to_EbN0_dB(σ2):
        E = 1
        EbN0 = E / (2*σ2) * denoiser.ldpc_code.N/denoiser.ldpc_code.K
        return 10*np.log10(EbN0)

    logger.debug("Finished creating empty result arrays")

    # Minimum σ2 needed for each given μ to achieve BER < target BER:
    # BER is averaged across AMP trials.
    critical_σ2_arr = -np.ones(num_δ)
    critical_BER_arr = -np.ones(num_δ)
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, n in enumerate(n_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}], S = {S_arr[i_δ]}=====")
        σ2_arr = σ2_arr[start_σ2_idx:]
        def f_σ2_to_BER(σ2):
            amp = AMP_jax(n, L, d, σ2, denoiser, log_file_name,
                          iter_max, num_trials, A_type, estimate_Tt)
            BER_arr = amp.run(run_bp_post_amp=True, calcPUPE=True, calcBER=True)[-1]
            return np.mean(BER_arr)

        critical_reversed_σ2_idx, critical_σ2_arr[i_δ], critical_BER_arr[i_δ] = \
            binary_search(σ2_arr[::-1], BER, f_σ2_to_BER, log_file_name)
        if critical_reversed_σ2_idx == -1:
            logger.debug('Did not find critical σ2 for achieving target BER')
            break # more challenging cases (with higher S) cannot achieve target BER either.
        else:
            logger.debug(f'critical σ2 = {critical_σ2_arr[i_δ]}')
            logger.debug(f'critical EbN0_dB ={f_σ2_to_EbN0_dB(critical_σ2_arr[i_δ])}')
            logger.debug(f"critial BER = {critical_BER_arr[i_δ]}\n")

            start_σ2_idx = len(σ2_arr)-1 - critical_reversed_σ2_idx
    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2

    return critical_σ2_arr, critical_BER_arr



def binary_search_critical_σ2_coding(n_or_δ_arr, rate_float, σ2_arr, BER,
                              log_file_name, f_BER):
    """
    For GMAC with coding. BER is the target.

    When f_BER uses AMP to return BER, n_or_δ_arr is n_arr, and
    f_BER takes in two args: σ2 and n, returns BER.

    When f_BER uses SE to return BER, n_or_δ_arr is δ_arr, and
    f_BER takes in two args: σ2 and δ, returns BER.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered binary_search_critical_σ2_coding")
    assert np.all(n_or_δ_arr == np.sort(n_or_δ_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n or δ is decreasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    logger.debug(f"Creating empty result arrays")
    num_δ = len(n_or_δ_arr) # vary S via δ, (k, d are fixed)

    def f_σ2_to_EbN0_dB(σ2):
        E = 1
        EbN0 = E / (2*σ2) * 1/rate_float
        return 10*np.log10(EbN0)

    logger.debug("Finished creating empty result arrays")

    # Minimum σ2 needed for each given μ to achieve BER < target BER:
    # BER is averaged across AMP trials.
    critical_σ2_arr = -np.ones(num_δ)
    critical_BER_arr = -np.ones(num_δ)
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, n_or_δ in enumerate(n_or_δ_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        σ2_arr = σ2_arr[start_σ2_idx:]
        def f_σ2_to_BER(σ2):
            return f_BER(σ2, n_or_δ)
        critical_reversed_σ2_idx, critical_σ2_arr[i_δ], critical_BER_arr[i_δ] = \
            binary_search(σ2_arr[::-1], BER, f_σ2_to_BER, log_file_name)
        if critical_reversed_σ2_idx == -1:
            logger.debug('Did not find critical σ2 for achieving target BER')
            break # more challenging cases (with higher S) cannot achieve target BER either.
        else:
            logger.debug(f'critical σ2 = {critical_σ2_arr[i_δ]}')
            logger.debug(f'critical EbN0_dB ={f_σ2_to_EbN0_dB(critical_σ2_arr[i_δ])}')
            logger.debug(f"critial BER = {critical_BER_arr[i_δ]}\n")

            start_σ2_idx = len(σ2_arr)-1 - critical_reversed_σ2_idx
    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2

    return critical_σ2_arr, critical_BER_arr


def binary_search_critical_σ2_random(n_or_δ_arr, σ2_arr, epsTotal,
                              log_file_name, f_epsTotal):
    """
    For GMAC with random user activity.
    Specify and fix α, d, (and L in the case of AMP) externally.
    epsTotal: target total error probability = max(pMD,pFA)+pAUE.

    When f_epsTotal uses AMP to return epsTotal, n_or_δ_arr is AMP_n_arr, and
    f_epsTotal takes in two args: σ2 and n, returns epsTotal.

    When f_epsTotal uses SE to return epsTotal, n_or_δ_arr is δ_arr, and
    f_epsTotal takes in two args: σ2 and δ, returns epsTotal.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered binary_search_critical_σ2_random")
    assert np.all(n_or_δ_arr == np.sort(n_or_δ_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n or δ is decreasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    logger.debug(f"Creating empty result arrays")
    num_δ = len(n_or_δ_arr) # vary S via δ, (k, d are fixed)

    def f_σ2_to_EbN0_dB(σ2):
        E = 1
        EbN0 = E / (2*σ2) # silent users do not contribute energy towards Y
        return 10*np.log10(EbN0)

    logger.debug("Finished creating empty result arrays")

    # Minimum SNR needed for each given S to achieve epsTotal < target epsTotal:
    # epsTotal is averaged across AMP trials.
    critical_σ2_arr = -np.ones(num_δ)
    critical_eps_arr = -np.ones(num_δ)
    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i_δ, n_or_δ in enumerate(n_or_δ_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        σ2_arr = σ2_arr[start_σ2_idx:]
        def f_σ2_to_epsTotal(σ2):
            return f_epsTotal(σ2, n_or_δ)
        critical_reversed_σ2_idx, critical_σ2_arr[i_δ], critical_eps_arr[i_δ] = \
            binary_search(σ2_arr[::-1], epsTotal, f_σ2_to_epsTotal, log_file_name)
        if critical_reversed_σ2_idx == -1:
            logger.debug('Did not find critical σ2 which achieves target epsTotal')
            break # more challenging cases (with higher S) cannot achieve target epsTotal either.
        else:
            logger.debug(f'critical σ2 = {critical_σ2_arr[i_δ]}')
            logger.debug(f'critical EbN0_dB ={f_σ2_to_EbN0_dB(critical_σ2_arr[i_δ])}')
            logger.debug(f"critial epsTotal = {critical_eps_arr[i_δ]}\n")

            start_σ2_idx = len(σ2_arr)-1 - critical_reversed_σ2_idx
    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan # meaning critical σ2 is lower than the
                                                    # minimum in the provided set of σ2

    return critical_σ2_arr, critical_eps_arr


def binary_search_critical_EbN0dB_random(n_or_δ_arr, \
    low_EbN0dB, high_EbN0dB, epsTotal, epsTotal_tol, \
    f_epsTotal, log_file_name):
    """
    For GMAC with random user activity.
    Specify and fix α, d, (and L in the case of AMP) externally.
    low_EbN0dB, high_EbN0dB: search range for EbN0dB to find the SNR
    that gives target epsTotal.

    epsTotal: target total error probability = max(pMD,pFA)+pAUE.
    epsTotal_tol: tolerance for epsTotal. Stop binary search when
    the epsTotal found is in [epsTotal-epsTotal_tol, epsTotal]

    When f_epsTotal uses AMP to return epsTotal, n_or_δ_arr is AMP_n_arr, and
    f_epsTotal takes in two args: EbN0dB and n, returns epsTotal.

    When f_epsTotal uses SE to return epsTotal, n_or_δ_arr is δ_arr, and
    f_epsTotal takes in two args: EbN0dB and δ, returns epsTotal.
    """
    initialise_logger(logger, log_file_name)
    logger.debug(f"Entered binary_search_critical_EbN0dB_random")
    assert np.all(n_or_δ_arr == np.sort(n_or_δ_arr)[::-1]), \
        "S_arr must be monotonically increasing i.e. n or δ is decreasing"
    assert low_EbN0dB < high_EbN0dB, \
        "low_EbN0dB must be < high_EbN0dB"

    num_δ = len(n_or_δ_arr) # vary S via δ, (k, d are fixed)

    # Minimum SNR needed for each given S to achieve epsTotal < target epsTotal:
    # epsTotal is averaged across AMP trials.
    critical_EbN0dB_arr = -np.ones(num_δ)
    critical_eps_arr = -np.ones(num_δ)
    for i_δ, n_or_δ in enumerate(n_or_δ_arr):
        logger.debug(f"===== δ [{i_δ}/{num_δ}] =====")
        def f_EbN0dB_to_epsTotal(EbN0dB):
            return f_epsTotal(EbN0dB, n_or_δ)
        critical_EbN0dB_arr[i_δ], critical_eps_arr[i_δ] = \
            binary_search_EbN0dB_error_tol(low_EbN0dB, high_EbN0dB,
            epsTotal, epsTotal_tol, f_EbN0dB_to_epsTotal, log_file_name)
        if critical_EbN0dB_arr[i_δ] == -1:
            logger.debug("Did not find critical σ2 that achieves target epsTotal, \n" + \
                "so more challenging cases (with higher S) cannot achieve target epsTotal either.")
            break
        else:
            logger.debug(f'critical EbN0_dB ={critical_EbN0dB_arr[i_δ]},\n')
            logger.debug(f"critial epsTotal = {critical_eps_arr[i_δ]}\n")
            low_EbN0dB = critical_EbN0dB_arr[i_δ] # next S (higher) needs higher EbN0dB
    # Erase the -1 entries in critical_EbN0dB_arr:
    critical_EbN0dB_arr[critical_EbN0dB_arr == -1] = np.nan
    # meaning critical EbN0dB is higher than the maximum in [low_EbN0dB, high_EbN0dB]

    return critical_EbN0dB_arr, critical_eps_arr
