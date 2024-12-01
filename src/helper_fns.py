import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # colourmaps
from datetime import datetime
from numba import jit
import jax.numpy as jnp

# Functions for
# - sanity checks,
# - reporting results (e.g. printing iteration numbers, plotting graphs)


def log_B_1(k):
    """
    Natural log of (B-1) when B is large, assuming K=1.
    Recall k = log2(B) + log2(K), so when K=1, B=2**k
    """
    factor = jnp.log2(jnp.e) # rescale to base-e
    return k/factor + jnp.log(1-2**(-k))


def ldpc_rate_str2float(s):
    """Convert a string to a float, e.g. '1/2' --> 0.5"""
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


@jit(nopython=True)
def create_random_covariance_matrix(d):
    """Create a random valid covariance matrix of size dxd."""
    sqrt = np.tril(np.random.randn(d, d))
    np.fill_diagonal(sqrt, np.abs(np.diag(sqrt)))  # diag entries must be +ve
    Sigma = sqrt @ sqrt.T  # symmetric, +ve semidefinite
    return Sigma

# @jit(nopython=True) # for some reason, this throws errors
def is_valid_covariance_matrix(Sigma: np.array) -> bool:
    """
    Check if Sigma is a valid covariance matrix.

    - A matrix is invertible if it doesn't have any zero eigenvalues;
      +ve or -ve eigenvalues are allowed.
    - A covariance matrix however, by definition, must be symmetric
      +ve semidefinite.
    """
    if Sigma.ndim == 0:  # scalar
        return Sigma >= 0
    else:
        is_square = Sigma.ndim == 2 and (Sigma.shape[0] == Sigma.shape[1])
        is_symm = np.max(Sigma - Sigma.T) < 1e-6 # np.allclose(Sigma, Sigma.T) not supported by numba
        # is_symm = np.all(Sigma == Sigma.T) # too stringent, always fails
        Sigma = Sigma + 1e-12 * np.eye(Sigma.shape[0])
        is_pos_semidef = np.all(np.linalg.eigvals(Sigma) >= 0)
        # without small identity, eigenvalues may contain spurious complex parts
        # assert np.linalg.matrix_rank(Sigma) == Sigma.shape[0]
        return is_square and is_symm and is_pos_semidef

def initialise_logger(logger, log_file_name: str):
    """
    In the script where this function is called from, create a logger object:
    logger = logging.getLogger(__name__)
    and pass it to this function.
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not logger.hasHandlers():
            fh = logging.FileHandler(log_file_name)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)


def print_iter(name_str: str, idx: int, idx_max: int, val=None, idx_step: int = 1):
    """
    Print to screen the value of the iterable of interest if the
    index of the iterable idx is a multiple of the prespecified idx_step.
    """
    assert idx >= 0 and idx_max >= idx
    if idx % idx_step == 0:
        if val is None:
            print(f"=== Running {name_str} [{idx}/{idx_max}] ===")
        else:
            print(f"=== Running {name_str} = {val} [{idx}/{idx_max}] ===")


def print_elapsed_time(t_stop, t_start):
    """t_start and t_stop should be returned from perf_counter()"""
    t_elap = (t_stop - t_start) / 60
    print(f"Elapsed time in mins: {t_elap}")
    return t_elap


def timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def extend_plot(critical_EbN0_dB_arr, S_arr, EbN0_dB_max_to_plot):
    """
    Extends the plot to the rightmost EbN0 by repeating the last S value.
    critical_EbN0_dB_arr may contain nan values.
    """
    critical_EbN0_dB_to_plot = critical_EbN0_dB_arr[~np.isnan(critical_EbN0_dB_arr)]

    S_to_plot = S_arr[~np.isnan(critical_EbN0_dB_arr)]
    critical_EbN0_dB_to_plot = np.append(critical_EbN0_dB_to_plot, EbN0_dB_max_to_plot)
    S_to_plot = np.append(S_to_plot, S_to_plot[-1])
    return critical_EbN0_dB_to_plot, S_to_plot

def correct_transition_plot(S_arr, critical_EbN0_dB_arr):
    """
    Manually correct transition plot S v Eb/N0 for a given error rate.
    If a larger S requires a smaller EbN0 to achieve, replace the EbN0 of
    the smaller S with the EbN0 of the larger S:
    """
    n = len(S_arr)
    for i in range(1, n):
        if critical_EbN0_dB_arr[n-i] < \
            critical_EbN0_dB_arr[n-i-1]:
            critical_EbN0_dB_arr[n-i-1] = \
                critical_EbN0_dB_arr[n-i]
    return critical_EbN0_dB_arr

def file_name(metric_str: str, setting_str: str, file_format_str: str):
    """
    Create a file name in the format of metric_setting_timestamp, where
    metric is the quantity to plot (in 2D, it's plotted along the y axis),
    setting_str specifies the parameters fixed for the experiments to
    present in the plot
    """
    timestamp_str = timestamp()
    file_name_str = (
        metric_str + "_" + setting_str + "_" + timestamp_str + file_format_str
    )
    # Replace spaces with underscores;
    # consecutive spaces --> consecutive underscores:
    file_name_str = file_name_str.replace(" ", "_")
    assert not (" " in file_name_str)
    return file_name_str


def plot_error_vs_some_param(
    param_arr,
    err_arr,
    err_arr_label: str,
    param_str: str,
    err_metric_str: str,
    setting_str: str,
    fig_file_name_str: str,
    extra_err_arr=None,
    extra_err_arr_label: str = None,
    save_fig=False,
):
    """
    Inputs:
    param_arr: data to plot along the x axis. Must have >1 entry.
    err_arr: data to plot along the y axis.
             stores the algorithm performance according to some error metric.
             If err_arr is 2D, its rows correspond to different runs/
             experiments and its columns to different values that a parameter
             of interest take in each run.
             Num of rows can = 1 (in which case error_arr is 1D), but num of
             columns must be > 1.
             Num columns of err_arr = num entries in param_arr.
    param_str, err_metric_str: x, y labels of the plot
    setting_str: title of the plot specifying parameter setting
    file_name_str: name of the file to save the plot to

    This function plots the average error across runs (rows).
    When num of runs (rows) > 1, we plot error bars of +/- std.
    """
    assert np.ndim(param_arr) == 1
    assert 1 <= np.ndim(err_arr) <= 2
    # fig = plt.figure()  # create a figure object
    # ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # ax.plot([1, 2, 3, 4])
    # ax.set_ylabel('some numbers')
    plt.figure()
    if err_arr.ndim == 1:
        assert len(param_arr) == len(err_arr)
        plt.plot(param_arr, err_arr, label=err_arr_label)
    else:
        assert len(param_arr) == err_arr.shape[1]
        num_param_vals = len(param_arr)
        mean_err_arr = np.mean(err_arr, axis=0)
        assert mean_err_arr.shape == (num_param_vals,)
        std_err_arr = np.std(err_arr, axis=0)
        assert std_err_arr.shape == (num_param_vals,)
        plt.errorbar(param_arr, mean_err_arr, yerr=std_err_arr, label=err_arr_label)
    if extra_err_arr is not None:
        assert np.ndim(extra_err_arr) == 1
        assert len(param_arr) == len(extra_err_arr)
        plt.plot(param_arr, extra_err_arr, label=extra_err_arr_label, linewidth=2)
    plt.legend()
    plt.xlabel(param_str)
    plt.ylabel(err_metric_str)
    plt.title(setting_str + "error bars = +/-1std", fontsize=10)
    if save_fig:
        plt.savefig(fig_file_name_str)
    plt.show()


def contour_map(
    param0_arr,
    param1_arr,
    data_arr,
    param0_str: str,
    param1_str: str,
    data_str: str,
    setting_str: str,
    fig_file_name_str: str,
    save_fig=False,
):
    assert param0_arr.ndim == param1_arr.ndim == 1
    num_param0 = len(param0_arr)
    num_param1 = len(param1_arr)
    assert num_param0 > 1 and num_param1 > 1
    assert data_arr.ndim == 2
    assert data_arr.shape == (num_param0, num_param1)

    # contourf requires its 1st argument to match the num columns
    # and 2nd to match the num rows in the 2D data array to render.
    plt.figure()
    plt.contourf(param0_arr, param1_arr, data_arr.T, cmap=cm.coolwarm)
    # -- This meshgrid approach produces the same contour map:
    # param0, param1 = np.meshgrid(mu_arr, sigma2_arr)
    # plt.contourf(param0, param1, mean_final_mse_amp_arr.T, cmap=cm.coolwarm)
    # --
    plt.colorbar()
    plt.xlabel(param0_str)
    plt.ylabel(param1_str)
    plt.title(setting_str + data_str, fontsize=7)
    if save_fig:
        plt.savefig(fig_file_name_str)
    plt.show()
