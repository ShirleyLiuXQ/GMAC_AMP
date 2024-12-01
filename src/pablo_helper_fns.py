import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # colourmaps
from datetime import datetime
#from numba import jit

# Functions for
# - sanity checks (e.g. check function input dims),
# - enforcing stopping criterion of loops,
# - reporting results (e.g. printing iteration numbers, plotting graphs)


def calc_percent_diff(x0, xt, xt_1):
    """
    x0, xt, xt_1 can be arrays.
    Calculate |xt - x(t-1)| as a percentage of |xt - x0|
    """
    input_are_scalar = np.isscalar(x0) and np.isscalar(xt) and np.isscalar(xt_1)
    input_match_shape = x0.shape == xt.shape == xt_1.shape
    assert input_are_scalar or input_match_shape
    assert np.logical_not(np.any(abs(xt - x0) == 0))  # no zeros
    return abs(xt - xt_1) / abs(xt - x0)


def terminate_loop(x0, xt, xt_1, rtol: float):
    """
    x0, x1, ..., x(t-1), xt is a sequence output from a loop.
    This function returns a boolean to indicate whether or not
    to terminate the loop. The stopping criterion is specified
    by the relative tolerance rtol=|xt - x(t-1)|/ |xt - x0|.

    The sequence may be non-monotonic i.e. it may fluctuate.
    """
    assert np.isscalar(rtol) and rtol > 0
    percent_diff = calc_percent_diff(x0, xt, xt_1)
    return percent_diff <= rtol


def check_cov_mat(Sigma):
    """Check if Sigma is a valid covariance matrix."""
    if np.ndim(Sigma) == 0:  # scalar
        assert Sigma >= 0
    else:
        assert Sigma.ndim == 2
        assert Sigma.shape[0] == Sigma.shape[1]
        assert np.all(np.linalg.eigvals(Sigma) >= 0)  # +ve semidefinite
        # assert np.linalg.matrix_rank(Sigma) == Sigma.shape[0]
        assert np.allclose(Sigma, Sigma.T)


def scalar_to_arr2d(x):
    """
    x is a 2D square array or a scalar.
    Returns x itself if x is a 2D array or
    a 2D array containing x if x is a scalar
    """
    if np.ndim(x) == 0:
        y = np.array([[x]])
        assert y.ndim == 2 and y.shape == (1, 1)
    else:
        assert x.shape[0] == x.shape[1]
        y = x
    return y


def arr2d_to_scalar(x):
    """
    x is a 2D square array or a scalar.
    Returns x itself if x is a scalar or
    squeeze out axes with dim 1 and return the scalar
    if x is a 2D array
    """
    if x.ndim == 2:
        assert x.shape[0] == x.shape[1]
        y = np.squeeze(x)
    else:
        assert np.ndim(x) == 0
        y = x
    return y


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


def print_exit_loop_mssg(num_iter_conv: int):
    """Print to screen that we have exited from the loop"""
    print(f"=== Exited loop after {num_iter_conv} iterations. ===")


def print_init_n_final_vals(var_name_str: str, var_arr):
    """Print to screen the first and last entry in vector var_mat."""
    assert var_arr.ndim == 1
    print(
        f"Initial {var_name_str} = {var_arr[0]} \n"
        + f"Final {var_name_str} = {var_arr[-1]}"
    )


def print_elapsed_time(t_stop, t_start):
    """t_start and t_stop should be returned from perf_counter()"""
    t_elap = t_stop - t_start
    print(f"Elapsed time in mins: {(t_elap)/60}")
    return t_elap


def calc_n_print_percent_diff(scalar_mse_amp_arr, scalar_mse_se_arr):
    """The last axes of the two input arrays correpond to num iterations."""
    mse_amp_percent_diff = calc_percent_diff(
        scalar_mse_amp_arr[..., 0],
        scalar_mse_amp_arr[..., -1],
        scalar_mse_amp_arr[..., -2],
    )
    max_mse_amp_percent_diff = np.max(mse_amp_percent_diff)
    print(f"max last-step diff in mse_amp = {max_mse_amp_percent_diff} of total diff")

    mse_se_percent_diff = calc_percent_diff(
        scalar_mse_se_arr[..., 0],
        scalar_mse_se_arr[..., -1],
        scalar_mse_se_arr[..., -2],
    )
    max_mse_se_percent_diff = np.max(mse_se_percent_diff)
    print(f"max last-step diff in mse_se = {max_mse_se_percent_diff} of total diff")
    return max_mse_amp_percent_diff, max_mse_se_percent_diff


def timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


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
    plt.title(setting_str + data_str, fontsize=10)
    if save_fig:
        plt.savefig(fig_file_name_str)
    plt.show()
