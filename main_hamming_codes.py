import argparse
import logging
import numpy as np
from src.denoiser_jax import Denoiser_jax
from src.helper_fns import timestamp
from src.init_model import calc_δ_for_fixed_S, calc_δ_in_from_δ, create_W, create_codebook, scale_σ2_for_fixed_EbN0
from src.se_jax import SE_SC_jax, SE_jax
from src.terminate_algo import calc_PUPE

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--alpha', type=float, default=0,
                           help='Probability that a user is silent." \
                           "alpha=0 means all users are active.')
    argparser.add_argument('--BER', type=float, default=1e-4,
                            help='Target BER')
    argparser.add_argument('--EbN0_dB', type=float, default=8,
                            help='Fix EbN0_dB and vary S.')
    argparser.add_argument('--S', type=float, default=2,
                           help='Spectral efficiency S to run SE at.')
    argparser.add_argument('--S_idx', type=int, default=0, \
                        help='Index of spectral efficiency S in the array of mu values.')
    argparser.add_argument("--save_path", type=str, default="./results/",
                           help="Path to save the log and data files.")
    args = argparser.parse_args()

    file_name = 'hamming_S_jax_idx_' + "{:02d}".format(args.S_idx) + '_' + \
        timestamp()
    log_file_name = args.save_path + file_name + '.log'
    data_file_name = args.save_path + file_name + '.npz'

    logger = logging.getLogger(__name__)  # create as a global variable
    # Set threshold to DEBUG, so all messages were printed
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)  # higher severity level than DEBUG
    console.setFormatter(formatter)
    logger.addHandler(console)

    logfile = logging.FileHandler(log_file_name) #, mode="w")  # overwrite
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    denoiser_type = 'mmse'
    iter_max = 150
    iter_max_sc = 600
    lam, omega = 20, 4 # 60, 11 # 40, 6
    W = create_W(lam, omega)
    R, C = W.shape
    δ_in_f = lambda δ: calc_δ_in_from_δ(δ, lam, omega)

    EbN0 = 10 ** (args.EbN0_dB/10)
    E = 1
    σ2 = E / (2 * EbN0) # uncoded Eb/N0 = E/(2σ2)
    # Use the same range of Spectral density (i.e. effective user density) for all schemes:
    S_arr = np.array([args.S])
    num_S = len(S_arr)

    # Uncoded:
    # (without sparsity, mse is the same for any d because the columns of X are effectively independent)
    d_uncoded = 1
    cb_uncoded = create_codebook(d_uncoded, code=None)
    δ_arr_uncoded = calc_δ_for_fixed_S(S_arr, d_uncoded, code=None)
    mse_se_arr_uncoded = np.zeros((num_S, iter_max))
    PUPE_arr_uncoded = np.zeros(num_S)
    BER_arr_uncoded = np.zeros(num_S)

    # Hamming code:
    d_hamm = 7
    cb_hamm = create_codebook(d_hamm, code='Hamming')
    δ_arr_hamm = calc_δ_for_fixed_S(S_arr, d_hamm, code='hamming')
    σ2_hamm = scale_σ2_for_fixed_EbN0(σ2, d_hamm, code='hamming')
    mse_se_arr_hamm = np.zeros((num_S, iter_max))
    PUPE_arr_hamm = np.zeros(num_S)
    BER_arr_hamm = np.zeros(num_S)
    mse_se_arr_hamm_sc = np.zeros((num_S, iter_max_sc))
    PUPE_arr_hamm_sc = np.zeros(num_S)
    BER_arr_hamm_sc = np.zeros(num_S)


    logger.info(f"== Uncoded ==")
    num_G_samples_uncoded = 20000
    total_num_bits_uncoded = num_G_samples_uncoded * d_uncoded
    assert total_num_bits_uncoded > 1/args.BER, \
    f"Increase num_G_samples to get total_num_bits = {total_num_bits_uncoded} > 1/BER = {1/args.BER}"
    # Without sparsity, mse should be the same for any d because the columns of X are effectively independent.

    logger.info(f"== Uncoded ==")
    denoiser = Denoiser_jax(args.alpha, cb_uncoded, denoiser_type)
    for i_δ, δ in enumerate(δ_arr_uncoded):
        logger.info(f"== δ = {δ} [{i_δ}/{num_S}]==")
        se_uncoded = SE_jax(δ, σ2, d_uncoded, denoiser, log_file_name,
                            iter_max, num_G_samples_uncoded)
        _, mse_se_arr_uncoded[i_δ], Tt_uncoded = se_uncoded.run()
        X0_uncoded, _, X_MAP_uncoded = se_uncoded.last_iter_mc(Tt_uncoded)
        PUPE_arr_uncoded[i_δ] = calc_PUPE(X0_uncoded, X_MAP_uncoded)
        BER_arr_uncoded[i_δ] = np.mean(X_MAP_uncoded != X0_uncoded)

    logger.info(f"== Hamming ==")
    num_G_samples = 40000
    total_num_bits = num_G_samples * d_hamm
    assert total_num_bits > 1/args.BER, \
        f"Increase num_G_samples to get total_num_bits = {total_num_bits} > 1/BER = {1/args.BER}"
    num_G_samples_sc = 4000
    total_num_bits_sc = num_G_samples_sc * C * d_hamm
    assert total_num_bits_sc > 1/args.BER, \
        f"Increase num_G_samples_sc to get total_num_bits_sc = {total_num_bits_sc} > 1/BER = {1/args.BER}"
    for i_δ, δ in enumerate(δ_arr_hamm):
        logger.info(f"== δ = {δ} [{i_δ}/{num_S}]==")
        denoiser = Denoiser_jax(args.alpha, cb_hamm, denoiser_type)
        se_hamm = SE_jax(δ, σ2_hamm, d_hamm, denoiser, log_file_name,
                     iter_max, num_G_samples)
        _, mse_se_arr_hamm[i_δ], Tt_hamm = se_hamm.run()
        X0_hamm, _, X_MAP_hamm = se_hamm.last_iter_mc(Tt_hamm)
        PUPE_arr_hamm[i_δ] = calc_PUPE(X0_hamm, X_MAP_hamm)
        BER_arr_hamm[i_δ] = np.mean(X_MAP_hamm != X0_hamm)

        δ_in = δ_in_f(δ)
        se_hamm_sc = SE_SC_jax(W, δ_in, σ2_hamm, d_hamm, denoiser, log_file_name,
                    iter_max_sc, num_G_samples_sc) # all codewords are used for E_X0
        _,_, mse_se_arr_hamm_sc[i_δ], Pt_hamm_sc = se_hamm_sc.run()
        X0_hamm_sc, _, X_MAP_hamm_sc = se_hamm_sc.last_iter_mc(Pt_hamm_sc)
        PUPE_arr_hamm_sc[i_δ] = calc_PUPE(X0_hamm_sc, X_MAP_hamm_sc)
        BER_arr_hamm_sc[i_δ] = np.mean(X_MAP_hamm_sc != X0_hamm_sc)

    logger.info(f"=== Finished expts for all codes ===")
    logger.info(f"=== Saving data to {data_file_name} ===")
    np.savez(data_file_name, alpha=args.alpha, lam=lam, omega=omega, \
             S=args.S, S_idx=args.S_idx, \
             file_name=file_name, EbN0_dB=args.EbN0_dB, \
             S_arr=S_arr, d_hamm=d_hamm, \
                σ2=σ2, σ2_hamm=σ2_hamm, \
                iter_max=iter_max, iter_max_sc=iter_max_sc, \
                mse_se_arr_uncoded=mse_se_arr_uncoded, \
                mse_se_arr_hamm=mse_se_arr_hamm, mse_se_arr_hamm_sc=mse_se_arr_hamm_sc, \
                PUPE_arr_uncoded=PUPE_arr_uncoded, \
                PUPE_arr_hamm=PUPE_arr_hamm, PUPE_arr_hamm_sc=PUPE_arr_hamm_sc, \
                BER_arr_uncoded=BER_arr_uncoded, \
                BER_arr_hamm=BER_arr_hamm, BER_arr_hamm_sc=BER_arr_hamm_sc)

    logger.info(f"=== Data saved ===")
    logger.info(f"=== Reached the end of script ===")
