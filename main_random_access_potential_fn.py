import logging
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import argparse
from src.helper_fns import timestamp
from src.potential_fn_RA_K1 import F_RA_all_ψ_K1, pErrors_K1
import jax.numpy as jnp

# Random Access (RA) potential function assuming SPARC-based coding scheme, K=1.
# k=6, use EbN0_min=2, EbN0_max=8, num_EbN0=20, mu_a=0.2, α=0.7 or 1, sectionwise and entrywise η
# k=50, use EbN0_min=2, EbN0_max=8, num_EbN0=20, mu_a=0.03, entrywise η only
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot RA potential functions (not the ach bounds).")
    parser.add_argument('--k', type=int, default=6,
                        help="User payload in bits, e.g. 6.")
    parser.add_argument('--alpha', type=float, default=1,
        help="Fraction of users active, e.g. 0.7.")
    parser.add_argument('--num_Zs_pot_fn', type=int, default=50000,
        help="Number of Z samples for calculating the potential function. 10000 gives smooth curves. " +
            "Too few samples e.g.20000 doesnt ensure global minimum is located correctly. " +
            "(i.e. minimum of sectionwise η may lie on the right of minimum of entrywise η.)")
    parser.add_argument('--num_Zs_pError', type=int, default=50000,
                        help="Number of Z samples for calculating the total error probability.")
    parser.add_argument('--EbN0_min', type=float, default=5.1,
        help="Minimum Eb/N0 in dB.")
    parser.add_argument('--EbN0_max', type=float, default=5.5,
        help="Maximum Eb/N0 in dB.")
    parser.add_argument('--num_EbN0', type=int, default=30,
        help="Number of Eb/N0 values to search.")
    parser.add_argument('--mu_a', type=float, default=0.2,
        help="μa = μ*α, where μ is the fraction of users active.")
    parser.add_argument('--η_type_list', action='append', type=str, required=True,
                           help="List of η_type values: sectionwise or entrywise.")
    parser.add_argument('--apply_exp_trick', action='store_true',
        help="Apply the exp trick to avoid overflow. By default, "
        "dont pass in anything so the exp trick isnt applied."
        "Otherwise, pass in --apply_exp_trick.")
    parser.add_argument("--save_path", type=str, default="./results/figs/",
                           help="Path to save the log and data files.")
    args = parser.parse_args()

    # global parameters:
    α = args.alpha
    E = 1
    k = args.k # user payload in bits, which = log2(B), so B=2**k, because K=1 by default.
    B = 2 ** k
    μa = args.mu_a # 0.1 # α*L/n = μ*α
    μ  = μa / args.alpha
    file_name = f'RA_pot_fn_k={k}_α={α}_μa={μa}_' + timestamp()
    log_file_name = args.save_path + file_name + '.log'
    data_file_name = args.save_path + file_name + '.npz'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logfile = logging.FileHandler(log_file_name) #, mode="w")  # overwrite
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    logger.info(f"k: {k}, α: {α}")
    logger.info(f"μa: {μa}, μ: {μ}")

    # ψ is the variable to minimise the potential function over:
    NUM_ψ = 1500  # coarser grid than this gives incorrect minima;
                    # finer grid sometimes identifies extreme values
                    # as local minima
    # IMPORTANT: grid near zero should be fine in order to show
    # grandular behaviour of mse, PUPE or other error metrics
    # near zero e.g. 10^{-3}. ψ corresponds to mse.

    NUM_ψ_NEAR_ZERO = 100

    ψ_over_E_near_zero_arr = 10 ** jnp.linspace(-8, -1, NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.linspace(1e-1, 1, NUM_ψ - NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.concatenate((ψ_over_E_near_zero_arr, ψ_over_E_arr))

    ψ_arr = jnp.array(ψ_over_E_arr * E, dtype=jnp.float32)  # [0, E]
    num_ψ = len(ψ_arr)

    EbN0_min = args.EbN0_min
    EbN0_max = args.EbN0_max
    num_EbN0 = args.num_EbN0 # 60
    EbN0_dB_arr = np.linspace(EbN0_min, EbN0_max, num_EbN0)
    EbN0_arr = 10 ** (EbN0_dB_arr / 10)
    σ2_arr = E / (2*k) / EbN0_arr # SPARC-based coding scheme differs from non-SPARC one
    logger.info(f"σ2_arr: {σ2_arr}")

    num_Zs_pot_fn = args.num_Zs_pot_fn # 50000
    num_Zs_pError = args.num_Zs_pError # 20000
    logger.info(f"num_Zs_pot_fn: {num_Zs_pot_fn}, num_Zs_pError: {num_Zs_pError}")
    # η_type_list = ["sectionwise", "entrywise"]
    η_type_list = args.η_type_list
    apply_exp_trick = args.apply_exp_trick
    logger.debug(f"η_type_list: {η_type_list}")
    logger.debug(f"apply_exp_trick: {apply_exp_trick}")

    F_arr_dict = {}
    ψ_idx_arr_dict = {}
    pErrors_arr_dict = {}

    logger.info("\n========= Global minimum of the potential function [Liu et al.] ========")
    for η_type in η_type_list:
        logger.info(f"η_type: {η_type}")
        F_arr_all_σ2 = np.zeros((num_EbN0, num_ψ))
        ψ_idx_arr = np.zeros(num_EbN0, dtype=int)
        pErrors_arr = np.zeros((num_EbN0, 3)) # stores the three types of errors
        # at the global minimum ψ
        for i_σ2, σ2 in enumerate(σ2_arr):
            F_arr_all_σ2[i_σ2, :] = F_RA_all_ψ_K1(α, B, E, μ, σ2, ψ_arr,
                num_Zs_pot_fn, η_type)
            sc_idx = jnp.argmin(F_arr_all_σ2[i_σ2, :])
            # sc_idx = identify_minima_argrelmin(F_arr_all_σ2[i_σ2, :])[0] # less accurate
            ψ_idx_arr[i_σ2] = sc_idx
            τ = lambda idx: σ2 + μ * ψ_arr[idx]
            pMD, pFA, pAUE = pErrors_K1(α, τ(sc_idx), B, E,
                num_Zs_pError, apply_exp_trick)
            logger.debug(f"pMD: {pMD}, pFA: {pFA}, pAUE: {pAUE}")
            pErrors_arr[i_σ2, :] = [pMD, pFA, pAUE]

        F_arr_dict[η_type] = F_arr_all_σ2
        ψ_idx_arr_dict[η_type] = ψ_idx_arr
        pErrors_arr_dict[η_type] = pErrors_arr
    logger.debug(f"ψ_idx_arr_dict: {ψ_idx_arr_dict}")
    if len(η_type_list) == 2:
        assert np.all(ψ_idx_arr_dict['sectionwise'] <= ψ_idx_arr_dict['entrywise']), \
            "sectionwise η should always have lower mse than entrywise η."

    logger.info("=== Plotting ===")
    if len(η_type_list) == 1:
        linestyle_list = ['-']
        scatter_marker_list = ['o']
    elif len(η_type_list) == 2:
        linestyle_list = ['-', '--']
        scatter_marker_list = ['x','o']
    plt.rcParams.update({'font.size': 12})

    # Plot the potential function for five differet σ2 values, against the normalised mse ψ/E:
    logger.info("Plotting potential function against normalised mse ψ/E (full)")
    num_σ2_to_plot = 5
    interval_σ2 = num_EbN0 / num_σ2_to_plot
    plt.figure()
    for i_η_type, (η_type, linestyle) in enumerate(zip(η_type_list, linestyle_list)):
        num_plots = 0
        for i_σ2, σ2 in enumerate(σ2_arr):
            if i_σ2 % interval_σ2 == 0:
                F_arr = F_arr_dict[η_type][i_σ2] # length-num_ψ
                # Recall ψ is mse, so divide by E to get normalised mse:
                plt.plot(ψ_arr/E, F_arr,
                    label=f'{η_type}, EbN0={int(EbN0_dB_arr[i_σ2]*100)/100}dB',
                    color=f"C{num_plots}", linestyle=linestyle)
                # Mark the global minimum:
                sc_idx = ψ_idx_arr_dict[η_type][i_σ2]
                plt.scatter(
                    ψ_arr[sc_idx]/E, F_arr[sc_idx], marker="x", color=f"C{num_plots}")
                num_plots += 1
        assert num_plots == num_σ2_to_plot
    plt.legend()
    plt.xlabel(r"normalised MSE $\psi/E$")
    plt.ylabel(r"potential function $\mathcal{F}$")
    plt.title(f"K=1,k={k},E={E},μa={μa},α={args.alpha},\nnum_Zs_pot_fn={num_Zs_pot_fn},num_Zs_pError={num_Zs_pError}")
    plt.xscale("log")
    plt.grid()
    fig_file_name = args.save_path + file_name + '_pot_fn_v_ψ_full' + '.pdf'
    plt.savefig(fig_file_name)

    logger.info("Plotting potential function against normalised mse ψ/E")
    cmap = matplotlib.colormaps.get_cmap('plasma')
    colour_list = [cmap(j) for j in np.linspace(0, 0.8, num_σ2_to_plot)]
    plt.figure()
    for i_η_type, (η_type, linestyle, scatter_marker) in \
        enumerate(zip(η_type_list, linestyle_list, scatter_marker_list)):
        num_plots = 0
        for i_σ2, σ2 in enumerate(σ2_arr):
            if i_σ2 % interval_σ2 == 0:
            # if i_σ2 in i_σ2_to_plot:
                F_arr = F_arr_dict[η_type][i_σ2] # length-num_ψ
                # Recall ψ is mse, so divide by E to get normalised mse:
                # "+3" below ensures the colours of these plots differ from those of
                # the plots for the 3 types of errors:
                if i_η_type == 0:
                    plt.plot(ψ_arr/E, F_arr,
                        label=r'$E_b/N_0$' + f'={int(EbN0_dB_arr[i_σ2]*100)/100}dB',
                        color=colour_list[num_plots][0:3], linestyle=linestyle, alpha=0.6)
                else:
                    # No legend:
                    plt.plot(ψ_arr/E, F_arr,
                        color=colour_list[num_plots][0:3], linestyle=linestyle, alpha=0.6)
                # Mark the global minimum:
                sc_idx = ψ_idx_arr_dict[η_type][i_σ2]
                plt.scatter(
                    ψ_arr[sc_idx]/E, F_arr[sc_idx], marker=scatter_marker,
                    color=colour_list[num_plots][0:3], s=40)
                num_plots += 1
        assert num_plots == num_σ2_to_plot
    # Specify linestyles in black in legend:
    for i_η_type, (η_type, linestyle, scatter_marker) in \
        enumerate(zip(η_type_list, linestyle_list, scatter_marker_list)):
        if η_type == "sectionwise":
            label = r'$\mathcal{F}_{Bayes}$'
        elif η_type == "entrywise":
            label = r'$\mathcal{F}_{marginal}$'
        plt.plot([ψ_arr[0]/E], [np.max(F_arr_dict[η_type])],
                 color="black", linestyle=linestyle, label=label, marker=scatter_marker)
    plt.xlabel(r"normalised MSE $\psi/E$")
    plt.ylabel(r"potential function $\mathcal{F}$")
    plt.xscale("log")
    plt.grid()
    plt.legend(loc='upper left', ncols=2)
    # plt.ylim([2.5, 4.6])
    # plt.ylim([3.5, 6.3])
    plt.xlim([1e-8, 1])
    plt.tight_layout()
    fig_file_name1 = args.save_path + file_name + '_pot_fn_v_ψ' + '.pdf'
    plt.savefig(fig_file_name1)

    logger.info("Plotting errors against EbN0 (full)")
    # Plot errors against EbN0:
    plt.figure()
    for i_η_type, (η_type, linestyle) in enumerate(zip(η_type_list, linestyle_list)):
        for i in range(3):
            plt.plot(EbN0_dB_arr, pErrors_arr_dict[η_type][:, i],
                label=f'{η_type} ' + [r"$\varepsilon_{MD}$",
                                    r"$\varepsilon_{FA}$",
                                    r"$\varepsilon_{AUE}$"][i],
                color=f"C{i}", linestyle=linestyle, marker='o')
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("error probabilities")
    plt.title(f"K=1,k={k},E={E},μa={μa},α={args.alpha},\nnum_Zs_pot_fn={num_Zs_pot_fn},num_Zs_pError={num_Zs_pError}")
    fig_file_name2 = args.save_path + file_name + '_errors_v_EbN0_full' + '.pdf'
    plt.savefig(fig_file_name2)

    logger.info("Plotting errors against EbN0")
    plt.figure()
    for i in range(3):
        for i_η_type, (η_type, linestyle) in enumerate(zip(η_type_list, linestyle_list)):
            if i_η_type == 0:
                plt.plot(EbN0_dB_arr, pErrors_arr_dict[η_type][:, i],
                    label= [r"$\varepsilon_{MD}$",
                                    r"$\varepsilon_{FA}$",
                                    r"$\varepsilon_{AUE}$"][i],
                    color=f"C{i}", linestyle=linestyle)
            else:
                # No legend:
                plt.plot(EbN0_dB_arr, pErrors_arr_dict[η_type][:, i],
                    color=f"C{i}", linestyle=linestyle)
    # Specify linestyles in black in legend:
    for i_η_type, (η_type, linestyle) in enumerate(zip(η_type_list, linestyle_list)):
        if η_type == "sectionwise":
            label = r'AMP with $\eta_t^{Bayes}$'
        elif η_type == "entrywise":
            label = r'AMP with $\eta_t^{marginal}$'
        plt.plot([EbN0_dB_arr[0]], [np.max(pErrors_arr_dict[η_type])],
                 color="black", linestyle=linestyle, label=label)
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("error probabilities")
    plt.tight_layout()
    plt.xlim([EbN0_min, EbN0_max])
    fig_file_name3 = args.save_path + file_name + '_errors_v_EbN0' + '.pdf'
    plt.savefig(fig_file_name3)


    # save data:
    np.savez(data_file_name, μa=μa,
             k=k, α=α, E=E,
             EbN0_dB_arr=EbN0_dB_arr, num_EbN0=num_EbN0, σ2_arr=σ2_arr,
             η_type_list=η_type_list, apply_exp_trick=apply_exp_trick,
             num_Zs_pot_fn=num_Zs_pot_fn, num_Zs_pError=num_Zs_pError,
             ψ_arr=ψ_arr,
             F_arr_dict=F_arr_dict, ψ_idx_arr_dict=ψ_idx_arr_dict,
            pErrors_arr_dict=pErrors_arr_dict)
    logger.info(f"==== Data saved, reached the end ====")
