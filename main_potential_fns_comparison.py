from matplotlib import pyplot as plt
import numpy as np
import argparse
import datetime
from src.potential_fn import find_critical_σ2_global_only
from src.potential_fn_kowshik import find_kowshik_critical_σ2
from src.zadik_bounds import gmac_achieve_zadik19, gmac_converse_comb

# This script allows any user-defined k, PUPE, and EbN0_dB_arr, S.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        "Plot Kowshik's ach bound, ach bound by Liu et al., and Zadik's ach+conv bounds.")
    parser.add_argument('--k', type=int, default=60, help=
        "User payload in bits. Uncoded is 1, Hamming code is 4." +
        "This gives accurate results for k up to 100." + \
        "May be numerically unstable for k>100.")
    parser.add_argument('--BER', type=float, default=1e-3/2,
                        help="e.g. BER=1e-4. BER=PUPE/2")
    parser.add_argument('--save_path', type=str, default='./results/',
                        help='Path to the results folder.')
    args = parser.parse_args()

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # global parameters:
    E = 1
    k = args.k # user payload in bits, which = log2(B), so B=2**k
    # If K=2, then user payload k = log2(B)+1
    num_EbN0 = 40
    if k == 1:
        PUPE = args.BER
    else:
        PUPE = args.BER * 2 # because ach bounds are based on random coding

    if k == 1: # uncoded
        EbN0_dB_arr = np.linspace(8, 17, num_EbN0)
    elif k == 4: # Hamming code in my scheme.
        EbN0_dB_arr = np.linspace(4, 16, num_EbN0)
    elif k <= 62: # LDPC
        EbN0_dB_arr = np.linspace(0.1, 16, num_EbN0)
    else:
        raise ValueError("k > 62 only works with higher machine precision.")

    EbN0_dB_arr = np.linspace(13.7, 14, 10)
    EbN0_arr = 10 ** (EbN0_dB_arr / 10)
    σ2_arr = E / (2*k) / EbN0_arr # SPARC-based coding scheme differs from non-SPARC one
    print(f"σ2_arr: {σ2_arr}")
    if k > 4:
        S_arr = np.concatenate([np.linspace(0.01, 2, 20), np.linspace(2, 4, 5)])
    else:
        S_arr = np.linspace(0.01, 4, 40)
    S_arr = np.linspace(0.01, 1.3, 20)
    print(f"S_arr: {S_arr}")
    μ_arr = S_arr / k
    print(f"μ_arr: {μ_arr}")

    if True:
        print("\n========= Global minimum of the potential function [by Hsieh et al. 2022] ========")
        if k == 1:
            B_hsieh = 1
            K_hsieh = 2
            assert np.log2(B_hsieh) + np.log2(K_hsieh) == k
            ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr = \
                find_critical_σ2_global_only(K_hsieh, B_hsieh, E, μ_arr, σ2_arr, PUPE,
                    log_file_name="results/test_hsieh_" + timestamp + ".log")
            critical_EbN0_dB_arr_hsieh = 10 * np.log10(E / (2*k) / critical_σ2_arr)
            print(f"critical_EbN0_dB_arr: {critical_EbN0_dB_arr_hsieh}")
            plt.figure()
            plt.plot(critical_EbN0_dB_arr_hsieh, S_arr,
                    label=f'Hsieh, ach, K={K_hsieh}, B={B_hsieh}')

            # Another way of ensuring 4 bit payload is via:
            K_hsieh1 = 1
            B_hsieh1 = int(2**k)
            assert np.log2(B_hsieh1) + np.log2(K_hsieh1) == k
            ψ_arr1, F_arr1, min_F_idx_arr1, min_F_Pe_arr1, critical_σ2_arr1 = \
                find_critical_σ2_global_only(K_hsieh1, B_hsieh1, E, μ_arr, σ2_arr, PUPE,
                    log_file_name="results/test_hsieh_K1_" + timestamp + ".log")
            critical_EbN0_dB_arr_hsieh1 = 10 * np.log10(E / (2*k) / critical_σ2_arr1)
            plt.plot(critical_EbN0_dB_arr_hsieh1, S_arr,
                    label=f'Hsieh, ach, K={K_hsieh1}, B={B_hsieh1}')
        elif k == 4:
            K_hsieh = 2
            B_hsieh = int(2**(k - np.log2(K_hsieh)))
            assert np.log2(B_hsieh) + np.log2(K_hsieh) == k
            ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr = \
                find_critical_σ2_global_only(K_hsieh, B_hsieh, E, μ_arr, σ2_arr, PUPE,
                    log_file_name="results/test_hsieh_K2_" + timestamp + ".log")
            critical_EbN0_dB_arr_hsieh = 10 * np.log10(E / (2*k) / critical_σ2_arr)
            plt.figure()
            plt.plot(critical_EbN0_dB_arr_hsieh, S_arr, label=f'Hsieh, ach, K={K_hsieh}, B={B_hsieh}')
            # Another way of ensuring 4 bit payload is via:
            K_hsieh1 = 1
            B_hsieh1 = int(2**k)
            assert np.log2(B_hsieh1) + np.log2(K_hsieh1) == k
            ψ_arr1, F_arr1, min_F_idx_arr1, min_F_Pe_arr1, critical_σ2_arr1 = \
                find_critical_σ2_global_only(K_hsieh1, B_hsieh1, E, μ_arr, σ2_arr, PUPE,
                    log_file_name="results/test_hsieh_K1_" + timestamp + ".log")
            critical_EbN0_dB_arr_hsieh1 = 10 * np.log10(E / (2*k) / critical_σ2_arr1)
            plt.plot(critical_EbN0_dB_arr_hsieh1, S_arr, label=f'Hsieh1, ach, K={K_hsieh1}, B={B_hsieh1}')
        else:
            K_hsieh, K_hsieh1, B_hsieh, B_hsieh1 = None, None, None, None
            critical_EbN0_dB_arr_hsieh = None
            critical_EbN0_dB_arr_hsieh1 = None
            plt.figure()

    K_kowshik = 1 # Kowshik uses no modulation
    if True:
        print("========= Use entrywise MAP decoding as hard-decision step (as in [Kowshik 2022]) ========")
        ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr_entrywise = \
            find_kowshik_critical_σ2(K_kowshik, k, E, μ_arr, σ2_arr, PUPE,
            hard_dec_step = "entrywise",
            log_file_name=f"results/test_kowshik_entrywise_" + timestamp + ".log")
        critical_EbN0_dB_arr_entrywise = 10 * np.log10(E / (2*k) / critical_σ2_arr_entrywise)
        print(f"critical_EbN0_dB_arr: {critical_EbN0_dB_arr_entrywise}")

        plt.plot(critical_EbN0_dB_arr_entrywise, S_arr, label='Kowshik entrywise, ach')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Spectral efficiency S')
        plt.grid()

    print("\n========= Use sectionwise MAP decoding as hard-decision step " + \
        "(previous steps same as [Kowshik 2022]) ========")
    ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr_sectionwise = \
        find_kowshik_critical_σ2(K_kowshik, k, E, μ_arr, σ2_arr, PUPE,
        hard_dec_step = "sectionwise",
        log_file_name="results/test_kowshik_sectionwise_" + timestamp + ".log")
    critical_EbN0_dB_arr_sectionwise = 10 * np.log10(E / (2*k) / critical_σ2_arr_sectionwise)
    print(f"critical_EbN0_dB_arr: {critical_EbN0_dB_arr_sectionwise}")

    plt.plot(critical_EbN0_dB_arr_sectionwise, S_arr, label='Kowshik sectionwise, ach')
    plt.legend()


    print("\n========= Achievability and converse bounds [Zadik] ========")
    # This ach scheme uses random codebooks to encode k bits per msg:
    B_zadik = 2 ** k # total number of msgs
    PUPE = args.BER * 2
    EbN0_ach_arr = gmac_achieve_zadik19(B_zadik, PUPE, μ_arr, EbN0_dB_arr)
    EbN0_conv_arr = gmac_converse_comb(B_zadik, PUPE, μ_arr)
    # The outputs above are in normal unit not dB
    EbN0_dB_ach_arr = 10*np.log10(EbN0_ach_arr)
    EbN0_dB_conv_arr = 10*np.log10(EbN0_conv_arr)

    plt.plot(EbN0_dB_ach_arr, S_arr, label='Zadik ach')
    plt.plot(EbN0_dB_conv_arr, S_arr, label='Zadik conv')
    plt.legend()
    plt.title(f'K_kowshik={K_kowshik}, E={E}, k={k}, BER={args.BER}, PUPE={PUPE}')
    plt.savefig(args.save_path + "test_kowshik_v_zadik_" + timestamp + ".pdf")

    # save data:
    np.savez(args.save_path + f"test_kowshik_v_zadik_k={k}_" + timestamp + ".npz",
             k=k, PUPE=PUPE, BER=args.BER, K_kowshik=K_kowshik, E=E,
             K_hsieh=K_hsieh, B_hsieh=B_hsieh, K_hsieh1=K_hsieh1, B_hsieh1=B_hsieh1,
        critical_EbN0_dB_arr_entrywise=critical_EbN0_dB_arr_entrywise,
        critical_EbN0_dB_arr_sectionwise=critical_EbN0_dB_arr_sectionwise,
        EbN0_dB_ach_arr=EbN0_dB_ach_arr, EbN0_dB_conv_arr=EbN0_dB_conv_arr,
        critical_EbN0_dB_arr_hsieh=critical_EbN0_dB_arr_hsieh,
        critical_EbN0_dB_arr_hsieh1=critical_EbN0_dB_arr_hsieh1,
        S_arr=S_arr, EbN0_dB_arr=EbN0_dB_arr, μ_arr=μ_arr)
