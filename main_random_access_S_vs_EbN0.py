import argparse
import logging
import numpy as np
from src.EbN0_S import binary_search_critical_σ2_random
from src.amp_jax import AMP_jax
from src.denoiser_jax import Denoiser_jax

from src.helper_fns import timestamp
from src.init_model import calc_δ_in_from_δ, create_W, create_codebook
from src.se_jax import SE_SC_jax
from src.terminate_algo import calc_pMD_FA_AUE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        "For fixed S, finds minimal EbN0 that achieves target error." + \
        "Note we dont allow dividing range of EbN0 into sub-ranges.")
    parser.add_argument('--L', type=int, default=100, help="Number of users.")
    parser.add_argument('--d', type=int, default=60,
                           help="Number of bits transmitted per user.")
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Probability that a user is silent. alpha=0 means all users are active.')
    parser.add_argument('--denoiser_type', type=str, default='mmse-marginal', \
                           help="Type of denoiser to use. ")
    parser.add_argument('--last_step', type=str, default='MAP', \
        help='The way to calculate errors upon convergence. Options: MAP or threshold.')
    parser.add_argument('--epsTotal', type=float, default=0.01, \
                           help="Target total error probability = max(pFA+pMD)+pAUE.")
    parser.add_argument('--lam', type=int, default=50,
                        help="param of base matrix for SC design, R=lam+omega-1,C=lam.")
    parser.add_argument('--omega', type=int, default=11,
                        help="param of base matrix for SC design, R=lam+omega-1,C=lam.")
    parser.add_argument('--num_S', type=int, default=20, help="Number of S values to experiment with.")
    parser.add_argument('--S_min', type=float, default=0.05, help="Minimum S.")
    parser.add_argument('--S_max', type=float, default=1.5, help="Maximum S.")
    parser.add_argument('--num_EbN0', type=int, default=25, help="Number of EbN0 values to experiment with.")
    parser.add_argument('--EbN0_dB_min', type=float, default=6, help="Minimum EbN0 in dB.")
    parser.add_argument('--EbN0_dB_max', type=float, default=10, help="Maximum EbN0 in dB.")
    parser.add_argument('--num_G_samples_sc', type=int, default=2000, \
                            help="Number of G samples to use for SC SE.")
    parser.add_argument('--iter_max_sc', type=int, default=600, \
                            help="Maximum number of iterations for SC SE.")
    parser.add_argument('--job_idx', type=int, default=1, help="Job idx.")
    parser.add_argument('--num_jobs_to_cover_all_S', type=int, default=20,
                           help="Number of jobs to cover all S, each job is indexed by an job_idx.")
    parser.add_argument('--num_trials', type=int, default=2, \
                            help="Number of AMP trials to average over.")
    parser.add_argument('--SE_or_AMP', type=str, default='SE', \
                            help="Whether to use SE or AMP for the plot.")
    parser.add_argument("--save_path", type=str, default="./results/",
                           help="Path to save the log and data files.")
    args = parser.parse_args()

    file_name = 'random_access_S_EbN0_jax_idx_' + "{:02d}".format(args.job_idx) + '_' + timestamp()
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

    # num_S = 10, 20
    num_S = args.num_S
    # S_min, S_max = 0.05, 3, 0.05, 1.5
    S_min, S_max = args.S_min, args.S_max
    S_arr = np.linspace(S_min, S_max, num_S)
    assert num_S % args.num_jobs_to_cover_all_S == 0, \
        "num_S should divide into num_jobs_to_cover_all_S"
    num_S_per_job = int(num_S / args.num_jobs_to_cover_all_S)
    S_arr = S_arr[args.job_idx * num_S_per_job: (args.job_idx+1) * num_S_per_job]

    num_EbN0 = args.num_EbN0 # 25
    # num_EbN0 = 10
    # EbN0_dB_min, EbN0_dB_max = 2, 6
    EbN0_dB_min, EbN0_dB_max = args.EbN0_dB_min, args.EbN0_dB_max
    EbN0_dB_arr = np.linspace(EbN0_dB_min, EbN0_dB_max, num_EbN0)
    EbN0_arr = 10 ** (EbN0_dB_arr/10)
    E = 1
    σ2_arr = E/EbN0_arr/2

    iter_max = 50
    iter_max_sc = args.iter_max_sc # 600 # need roughly 2x base matrix size (40,45) to converge

    lam, omega = args.lam, args.omega # 50, 11 # 40, 6
    W = create_W(lam, omega)
    R, C = W.shape
    logger.debug(f'For SC design, R={R}, C={C}')
    δ_in_f = lambda δ: calc_δ_in_from_δ(δ, lam, omega)

    assert args.L % C == 0, 'L/C should be integer'
    A_type = 'gaussian'
    estimate_Tt = True # empirically via AMP, SE is slow
    thres = 0.5 * E # to be used when last_step = 'threshold'

    def obj(pMD, pFA, pAUE):
        # Inputs are all scalars:
        assert np.isscalar(pMD) and np.isscalar(pFA) and np.isscalar(pAUE)
        return np.max([pMD, pFA]) + pAUE
    if args.denoiser_type == 'mmse':
        cb = create_codebook(args.d)
    elif args.denoiser_type == 'mmse-marginal' or \
            args.denoiser_type == 'thres':
        cb = None # force denoiser to create d=1 codebook (uncoded case)
    else:
        raise ValueError(f"Unknown denoiser type: {args.denoiser_type}")
    denoiser = Denoiser_jax(args.alpha, cb, args.denoiser_type)
    if args.SE_or_AMP == 'AMP':
        n_arr_amp = args.L * (1-args.alpha) / S_arr
        # Now make n_arr integer multiples of R:
        n_arr_amp = ( (n_arr_amp // R + 1) * R).astype(int)
        assert len(n_arr_amp) == num_S_per_job
        # Array of n is monotonically decreasing:
        assert np.all(n_arr_amp == np.sort(n_arr_amp)[::-1])
        assert np.all(n_arr_amp > 0)
        S_arr_amp = args.L * (1-args.alpha) / n_arr_amp
        assert args.L > 1/args.epsTotal, \
            f"Increase L to ensure L={args.L} > 1/args.epsTotal={1/args.epsTotal}"
        num_G_samples, num_G_samples_sc = None, None
        last_iter_num_G_samp_per_trial, last_iter_num_G_samp_sc_per_trial = None, None
        last_iter_num_trials = None

        logger.debug("\n== iid AMP ==")
        def f_σ2_n_to_eps_iid(σ2, n):
            amp = AMP_jax(n, args.L, args.d, σ2, denoiser, log_file_name,
                            iter_max, args.num_trials, A_type, estimate_Tt)
            pMD_arr, pFA_arr, pAUE_arr = amp.run(calc_pErrors=True)[2:5]
            pMD = np.mean(pMD_arr)
            pFA = np.mean(pFA_arr)
            pAUE = np.mean(pAUE_arr)
            return obj(pMD, pFA, pAUE)

        critical_σ2_arr_iid, critical_eps_arr_iid = binary_search_critical_σ2_random(
            n_arr_amp, σ2_arr, args.epsTotal,
            log_file_name, f_σ2_n_to_eps_iid)
        critical_σ2_arr_sc, critical_eps_arr_sc = None, None
    else:
        S_arr_amp = None
        δ_arr = (1-args.alpha)/S_arr
        last_iter_num_trials = 20
        logger.debug("\n== iid SE ==")
        num_G_samples = 1500 # and by default SE uses all possible values of X0
        last_iter_num_G_samp_per_trial = (5000 // C) * C
        last_iter_tot_num_G_samp = last_iter_num_G_samp_per_trial * last_iter_num_trials
        critical_σ2_arr_iid, critical_eps_arr_iid = None, None

        logger.debug("\n== SC SE ==")
        # d=6: 500 is sufficient; d=100: 2000 should be sufficient
        num_G_samples_sc = args.num_G_samples_sc # 500 # and by default SE uses all possible values of X0
        last_iter_num_G_samp_sc_per_trial = int(last_iter_num_G_samp_per_trial / C)
        last_iter_tot_num_G_samp_sc = last_iter_num_G_samp_per_trial * last_iter_num_trials

        # Above ensures error granularity of iid-SE and SC-SE are the same.
        assert last_iter_tot_num_G_samp_sc > 1/args.epsTotal, \
            f"Increase last_iter_tot_num_G_samp_sc to ensure "+ \
            f"last_iter_tot_num_G_samp_sc={last_iter_tot_num_G_samp_sc} > 1/args.epsTotal={1/args.epsTotal}"
        def f_σ2_δ_to_eps_sc(σ2, δ):
            δ_in = δ_in_f(δ)
            se_sc = SE_SC_jax(W, δ_in, σ2, args.d, denoiser, log_file_name,
                        iter_max_sc, num_G_samples_sc) # all codewords are used for E_X0
            Pt_sc = se_sc.run()[-1]
            pMD_arr = np.zeros(last_iter_num_trials)
            pFA_arr = np.zeros(last_iter_num_trials)
            pAUE_arr = np.zeros(last_iter_num_trials)
            for i in range(last_iter_num_trials):
                X0_sc, Xt_sc, X_MAP_sc = se_sc.last_iter_mc(Pt_sc, last_iter_num_G_samp_sc_per_trial)
                Xt_sc, X_MAP_sc = np.array(Xt_sc), np.array(X_MAP_sc)
                pMD, pFA, pAUE = calc_pMD_FA_AUE(X0_sc, X_MAP_sc) \
                    if args.last_step == 'MAP' else \
                    calc_pMD_FA_AUE(X0_sc, Xt_sc, quantise=True, thres_arr = np.array([thres]))
                pMD_arr[i], pFA_arr[i], pAUE_arr[i] = pMD.item(), pFA.item(), pAUE.item()
            pMD = np.mean(pMD_arr)
            pFA = np.mean(pFA_arr)
            pAUE = np.mean(pAUE_arr)
            return obj(pMD, pFA, pAUE)

        critical_σ2_arr_sc, critical_eps_arr_sc = binary_search_critical_σ2_random(
            δ_arr, σ2_arr, args.epsTotal, log_file_name, f_σ2_δ_to_eps_sc)


    logger.debug("==== Saving data ====")
    np.savez(data_file_name, L=args.L, d=args.d, alpha=args.alpha,
             denoiser_type=args.denoiser_type, last_step=args.last_step,
             epsTotal=args.epsTotal, job_idx=args.job_idx, \
             num_trials=args.num_trials, SE_or_AMP=args.SE_or_AMP,
            S_arr=S_arr, EbN0_dB_arr=EbN0_dB_arr, \
            num_S=num_S, num_EbN0=num_EbN0, num_S_per_job=num_S_per_job, \
            S_min=S_min, S_max=S_max, EbN0_dB_min=EbN0_dB_min, EbN0_dB_max=EbN0_dB_max, \
            S_arr_amp=S_arr_amp, \
            lam=lam, omega=omega, \
            iter_max=iter_max, iter_max_sc=iter_max_sc, \
            num_G_samples=num_G_samples, num_G_samples_sc=num_G_samples_sc, \
            last_iter_num_G_samp_per_trial=last_iter_num_G_samp_per_trial, \
            last_iter_num_G_samp_sc_per_trial=last_iter_num_G_samp_sc_per_trial, \
            last_iter_num_trials=last_iter_num_trials, \
            critical_σ2_arr_iid=critical_σ2_arr_iid, \
            critical_eps_arr_iid=critical_eps_arr_iid, \
            critical_σ2_arr_sc=critical_σ2_arr_sc, \
            critical_eps_arr_sc=critical_eps_arr_sc)

    logger.debug(f"== End of script ==")
