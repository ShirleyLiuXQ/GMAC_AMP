import argparse
import logging
import numpy as np
from ldpc_jossy.py.ldpc import code
from src.EbN0_S import binary_search_critical_σ2_coding
from src.amp_jax import AMP_jax
from src.denoiser_jax import Denoiser_jax
from src.helper_fns import ldpc_rate_str2float, timestamp
from src.init_model import calc_δ_for_fixed_S, calc_δ_in_from_δ, create_W, scale_σ2_for_fixed_EbN0
from src.se_jax import SE_SC_jax, SE_jax


# Compare LDPC with a given rate, with iid or SC design on a S versus EbN0 plot.
# Run AMP or SE, but not both.
# The default is to run binary search over all EbN0 for each S value.
# num_all_S and num_all_EbN0 specify the grid for the search.
if __name__ == "__main__":
    # Since we have a long series of S values to experiment with, each job in
    # the job array will run a subset of the S values.
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--z', type=int, default=30, \
                           help="z parameter of the LDPC code. Blocklength=24z.")
    argparser.add_argument('--rate_str', type=str, default='1/2', \
                           help="Rate of the LDPC code, e.g. '1/2', '2/3', '3/4', '5/6'")
    argparser.add_argument('--L', type=int, default=2000, help="Number of users.")
    argparser.add_argument('--BER', type=float, default=1e-4, \
                           help="Target BER.")
    argparser.add_argument('--num_trials', type=int, default=2, \
                            help="Number of AMP trials to average over.")
    argparser.add_argument('--denoiser_type', type=str, default='ldpc-bp', \
                           help="Type of denoiser to use. " + \
                            "Choose 'ldpc-bp' or 'mmse-marginal' for AMP with BP denoiser " + \
                            "or AMP with iid-marginal denoiser then BP.")
    argparser.add_argument('--bp_max_itcount', type=int, default=2, \
                            help="Number of BP iterations per AMP iteration")
    argparser.add_argument('--postprocessing', action='store_true', \
                            help="Whether to run BP after AMP. store_true by default" +
                            "i.e. NO postprocessing by default. No need to pass in anything." +
                            "To run postprocessing, pass in --postprocessing")
    argparser.add_argument('--SE_or_AMP', type=str, default='SE', \
                            help="Whether to use SE or AMP for the plot.")
    argparser.add_argument('--iid_or_SC', type=str, default='SC', \
                            help="Whether to use iid or SC design matrix.")
    argparser.add_argument('--S_min', type=float, default=0.05, \
                            help="Minimum S to simulate.")
    argparser.add_argument('--S_max', type=float, default=2.5, \
                            help="Maximum S to simulate.")
    argparser.add_argument('--num_all_S', type=int, default=20, \
                            help="Number of S values to simulate.")
    argparser.add_argument('--EbN0_dB_min', type=float, default=4, \
                            help="Minimum EbN0_dB to simulate.")
    argparser.add_argument('--EbN0_dB_max', type=float, default=16, \
                            help="Maximum EbN0_dB to simulate.")
    argparser.add_argument('--num_all_EbN0', type=int, default=20, \
                            help="Number of EbN0_dB values to simulate.")
    argparser.add_argument('--S_EbN0_comb_idx', type=int, default=0,
                           help="Index referring which combination of S and EbN0 to use.")
    argparser.add_argument('--num_G_samples', type=int, default=4000, \
                            help="Number of G samples for iid-SE.")
    argparser.add_argument('--num_G_samples_sc', type=int, default=200, \
                            help="Number of G samples for SC-SE.")
    argparser.add_argument('--num_X0_samples_sc', type=int, default=2, \
                            help="Number of X0 samples for SC-SE.")
    argparser.add_argument('--use_allzero_X0', action='store_true', \
        help="Whether to use all-zero X0 in SE (both iid-SE and SC-SE for LDPC codes only) " +
            "store_true by default " +
            "i.e. do NOT use all-zero X0 by default, use random X0 samples. " +
            "No need to pass in anything. " +
            "To use all-zero X0, pass in --use_allzero_X0")
    argparser.add_argument('--iter_max', type=int, default=150, \
                            help="Maximum number of iterations for SE.")
    argparser.add_argument('--iter_max_sc', type=int, default=200, \
                            help="Maximum number of iterations for SC-SE.")
    argparser.add_argument('--lam', type=int, default=40, \
                            help="Lambda parameter of the SC design matrix.")
    argparser.add_argument('--omega', type=int, default=6, \
                            help="omega parameter of the SC design matrix.")
    argparser.add_argument("--save_path", type=str, default="./results/",
                           help="Path to save the log and data files.")
    args = argparser.parse_args()

    file_name = 'ldpc_EbN0_vs_S_BER_idx_' + "{:02d}".format(args.S_EbN0_comb_idx) + '_' + timestamp()
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

    logger.debug(f"target BER = {args.BER}")
    logger.debug(f'postprocessing = {args.postprocessing}')
    logger.debug(f'use_allzero_X0 = {args.use_allzero_X0}')

    # Compare Kuan's uncoded case (K=2, B=1) with our coded case (K=2, B=1)
    # i.e. random CDMA.
    # Fix UER, plot the minimum Eb/N0 needed to achieve varying spectral efficiency.
    K = 2 # binary modulation
    B = 1 # number of columns is each section
    # log2(M) = log2(B) + log2(K) = num bits encoded in the spreading
    # sequences + num bits encoded in the modulation symbols
    M = 2 ** (np.log2(B) + np.log2(K)) # number of msgs
    k = np.log2(M) # user payload per transmission
    alpha = 0 # all users are active
    E = 1
    # Spectral efficiency = total # user bits/ total # channel uses = (Lk)/(nd) = μ*k
    # User density μ = # users/ total # channel uses = L/(nd)
    # ========================================================================
    num_all_S = args.num_all_S
    num_jobs_to_cover_all_S = num_all_S
    assert num_all_S % num_jobs_to_cover_all_S == 0
    num_S_per_job = num_all_S // num_jobs_to_cover_all_S
    num_all_EbN0 = args.num_all_EbN0
    num_jobs_to_cover_all_EbN0 = 1
    assert num_all_EbN0 % num_jobs_to_cover_all_EbN0 == 0
    num_EbN0_per_job = num_all_EbN0 // num_jobs_to_cover_all_EbN0

    assert 0 <= args.S_EbN0_comb_idx < \
        num_jobs_to_cover_all_S * num_jobs_to_cover_all_EbN0

    S_arr_idx = args.S_EbN0_comb_idx // num_jobs_to_cover_all_EbN0 # block idx
    EbN0_arr_idx = args.S_EbN0_comb_idx % num_jobs_to_cover_all_EbN0 # idx within block
    # e.g. if S_EbN0_comb_idx=11, num_jobs_to_cover_all_EbN0=2, then S_arr_idx=5, EbN0_arr_idx=1
    # higher S_EbN0_comb_idx refers to larger S values.

    S_min, S_max = args.S_min, args.S_max
    S_all_arr = np.linspace(S_min, S_max, num_all_S)
    logger.debug(f'S_all_arr={S_all_arr}')
    S_arr = S_all_arr[S_arr_idx*num_S_per_job:
                      (S_arr_idx+1)*num_S_per_job]
    num_S = len(S_arr)
    logger.debug(f'S_arr={S_arr}')

    EbN0_dB_min, EbN0_dB_max = args.EbN0_dB_min, args.EbN0_dB_max
    EbN0_dB_all_arr = np.linspace(EbN0_dB_min, EbN0_dB_max, num_all_EbN0)
    EbN0_dB_arr = EbN0_dB_all_arr[EbN0_arr_idx*num_EbN0_per_job:
                                  (EbN0_arr_idx+1)*num_EbN0_per_job]
    logger.debug(f'EbN0_dB_arr={EbN0_dB_arr}')
    num_EbN0 = len(EbN0_dB_arr)
    EbN0_arr = 10 ** (EbN0_dB_arr/10)
    # For uncoded system, Eb/N0 = ||c||^2 * E/(2σ2) = E/(2σ2)
    σ2_arr = E/EbN0_arr/2 # for uncoded system
    # ========================================================================
    lam, omega = args.lam, args.omega
    W = create_W(lam, omega)
    R, C = W.shape
    logger.debug(f'For SC design, R={R}, C={C}')
    δ_in_f = lambda δ: calc_δ_in_from_δ(δ, lam, omega)

    iter_max = args.iter_max
    iter_max_sc = args.iter_max_sc
    rate_float = ldpc_rate_str2float(args.rate_str)
    if args.z == 27 or args.z == 54 or args.z == 81:
        ldpc_code = code('802.11n', args.rate_str, args.z, 'A')
    else:
        ldpc_code = code('802.16', args.rate_str, args.z, 'A')
    assert ldpc_code.N == 24 * args.z
    logger.debug(f"LDPC code successfully created.")

    logger.debug(f"Creating Denoiser for LDPC code. denoiser_type={args.denoiser_type}")
    if args.denoiser_type == 'ldpc-bp':
        logger.debug(f"bp_iter = {args.bp_max_itcount}")
    denoiser = Denoiser_jax(alpha, type=args.denoiser_type,
                ldpc_code=ldpc_code, bp_max_itcount=args.bp_max_itcount)
    logger.debug(f"Denoiser created")
    # Use the following denoiser for debugging locally:
    # denoiser = Denoiser(alpha, d=ldpc_code.N, codebook=None, type='mmse-marginal',
    #                     ldpc_code=ldpc_code, bp_max_itcount=JOSSY_MAX_ITCOUNT)
    σ2_arr_ldpc = scale_σ2_for_fixed_EbN0(σ2_arr, code='ldpc', ldpc_rate=rate_float)

    assert args.L % C == 0, 'L/C should be integer'
    A_type = 'gaussian'
    estimate_Tt = True # empirically via AMP, SE is slow

    if args.SE_or_AMP == 'AMP':
        logger.debug(f"== Running AMP ==")
        logger.debug(f"L={args.L}, num_trials={args.num_trials}")
        # S scales inversely with n, so we cannot pick evenly spaced n values
        # which are integer multiples of R and then scale S accordingly.
        # We want evenly spaced S values.
        n_all_arr_ldpc_amp = args.L/S_all_arr * rate_float
        # Now make n_arr_acc_to_S integer multiples of R:
        n_all_arr_ldpc_amp = ( (n_all_arr_ldpc_amp // R + 1) * R).astype(int)
        assert len(n_all_arr_ldpc_amp) == num_all_S
        # Array of n is monotonically decreasing:
        assert np.all(n_all_arr_ldpc_amp == np.sort(n_all_arr_ldpc_amp)[::-1])

        n_arr_ldpc_amp = n_all_arr_ldpc_amp[S_arr_idx*num_S_per_job:
                        (S_arr_idx+1)*num_S_per_job]
        assert np.all(n_arr_ldpc_amp > 0)
        S_arr_ldpc_amp = args.L / n_arr_ldpc_amp *rate_float
        logger.debug(f'S_arr_ldpc_amp={S_arr_ldpc_amp}')

        num_G_samples_sc, num_X0_samples_sc, num_G_samples, num_X0_samples = None, None, None, None
        total_num_bits = args.L * ldpc_code.N # for both iid and SC
        assert total_num_bits > 1/args.BER, \
            f"Increase L to ensure total_num_bits={total_num_bits} > 1/args.BER={1/args.BER}"
        if args.iid_or_SC == 'iid':
            logger.debug(f"== LDPC iid-AMP ==")
            def f_σ2_n_to_BER(σ2, n):
                logger.debug(f'postprocessing = {args.postprocessing}')
                amp = AMP_jax(n, args.L, ldpc_code.N, σ2, denoiser, log_file_name,
                                iter_max, args.num_trials, A_type, estimate_Tt)
                BER_arr = amp.run(run_bp_post_amp=args.postprocessing, calcBER=True)[-1]
                return np.mean(BER_arr)
            critical_σ2_arr_iid, critical_BER_arr_iid = binary_search_critical_σ2_coding(
                n_arr_ldpc_amp, rate_float, σ2_arr_ldpc, args.BER, log_file_name, f_σ2_n_to_BER)
            logger.debug(f"Skipping SC-AMP")
            critical_σ2_arr_sc, critical_BER_arr_sc = None, None
        else:
            raise ValueError(f"args.iid_or_SC={args.iid_or_SC} AMP not included in this script")
    else:
        logger.debug(f"== Running SE ==")
        S_arr_ldpc_amp = None
        δ_arr_ldpc = calc_δ_for_fixed_S(S=S_arr, code='ldpc', ldpc_rate=rate_float)

        num_G_samples = args.num_G_samples # 4000 # fewer than 1000 samples gives jagged plots
        total_num_bits = num_G_samples * ldpc_code.N # fed into last_iter_mc
        assert total_num_bits > 1/args.BER, \
            f"Increase num_G_samples to ensure "+ \
                f"total_num_bits={total_num_bits} > 1/args.BER={1/args.BER}"
        num_X0_samples = 1
        logger.debug(f"num_G_samples={num_G_samples}, num_X0_samples={num_X0_samples}")

        num_G_samples_sc = args.num_G_samples_sc # 600
        total_num_bits_sc = num_G_samples_sc * C * ldpc_code.N # fed into last_iter_mc
        assert total_num_bits_sc > 1/args.BER, \
            f"Increase num_G_samples_sc to ensure "+ \
                f"total_num_bits_sc={total_num_bits_sc} > 1/args.BER={1/args.BER}"
        num_X0_samples_sc = args.num_X0_samples_sc # 1, 2
        logger.debug(f"num_G_samples_sc={num_G_samples_sc}, num_X0_samples_sc={num_X0_samples_sc}")
        if args.iid_or_SC == 'iid':
            logger.debug(f"Running iid-SE")
            def f_σ2_δ_to_BER(σ2, δ):
                se = SE_jax(δ, σ2, ldpc_code.N, denoiser, log_file_name,
                    iter_max, num_G_samples, num_X0_samples, args.use_allzero_X0)
                Tt = se.run()[-1]
                X0, _, X_MAP = se.last_iter_mc(Tt)
                BER_tmp = np.mean(X_MAP != X0)
                logger.debug(f"BER={BER_tmp}")
                return BER_tmp
            critical_σ2_arr_iid, critical_BER_arr_iid = binary_search_critical_σ2_coding(
                δ_arr_ldpc, rate_float, σ2_arr_ldpc, args.BER, log_file_name, f_σ2_δ_to_BER)
            logger.debug(f"Skipping SC-SE")
            critical_σ2_arr_sc, critical_BER_arr_sc = None, None
        else:
            logger.debug(f"Running SC-SE")
            last_iter_num_G_samples = 1000 # need to be large enough so that 1/(C*num_Gsamples) < PUPE
            # 1000*C=20000 which is enough to capture 1e-4 PUPE
            def f_σ2_δ_to_BER_sc(σ2, δ):
                δ_in = δ_in_f(δ)
                se_sc = SE_SC_jax(W, δ_in, σ2, ldpc_code.N, denoiser, log_file_name,
                    iter_max_sc, num_G_samples_sc, num_X0_samples_sc, args.use_allzero_X0)
                Pt_sc = se_sc.run()[-1]
                X0_sc, _, X_MAP_sc = se_sc.last_iter_mc(Pt_sc, last_iter_num_G_samples)
                BER_tmp = np.mean(X_MAP_sc != X0_sc)
                logger.debug(f"BER={BER_tmp}")
                return BER_tmp
            critical_σ2_arr_sc, critical_BER_arr_sc = binary_search_critical_σ2_coding(
                δ_arr_ldpc, rate_float, σ2_arr_ldpc, args.BER, log_file_name, f_σ2_δ_to_BER_sc)
            logger.debug(f"Skipping iid-SE")
            critical_σ2_arr_iid, critical_BER_arr_iid = None, None

    logger.debug("==== Saving data ====")
    np.savez(data_file_name, S_EbN0_comb_idx=args.S_EbN0_comb_idx, \
        S_arr=S_arr, EbN0_dB_arr=EbN0_dB_arr, \
        S_arr_ldpc_amp=S_arr_ldpc_amp, \
        rate_str=args.rate_str, z=args.z, L=args.L, \
        BER=args.BER, bp_max_itcount = args.bp_max_itcount, denoiser_type=args.denoiser_type, \
        postprocessing=args.postprocessing, SE_or_AMP=args.SE_or_AMP, \
        lam=lam, omega=omega, num_trials=args.num_trials, \
        iter_max=iter_max, iter_max_sc=iter_max_sc, \
        num_all_S=num_all_S, num_jobs_to_cover_all_S=num_jobs_to_cover_all_S, \
        num_all_EbN0=num_all_EbN0, num_jobs_to_cover_all_EbN0=num_jobs_to_cover_all_EbN0, \
        S_min=S_min, S_max=S_max, EbN0_dB_min=EbN0_dB_min, EbN0_dB_max=EbN0_dB_max, \
        num_G_samples=num_G_samples, num_X0_samples=num_X0_samples, \
        num_G_samples_sc=num_G_samples_sc, num_X0_samples_sc=num_X0_samples_sc, \
        critical_σ2_arr_iid=critical_σ2_arr_iid, \
        critical_BER_arr_iid=critical_BER_arr_iid, \
        critical_σ2_arr_sc=critical_σ2_arr_sc, \
        critical_BER_arr_sc=critical_BER_arr_sc)


    logger.debug(f"== End of script ==")
