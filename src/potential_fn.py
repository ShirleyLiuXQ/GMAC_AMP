import logging
import numpy as np
import jax
import jax.scipy as jsp
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
from src.helper_fns import log_B_1
from tqdm import tqdm


# import jax.scipy.integrate.trapezoid as jax_trapz
# import jax.numpy.trapz as jax_trapz

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ψ is the variable to minimise the potential function over:
NUM_ψ = 1000  # coarser grid than this gives incorrect minima;
                # finer grid sometimes identifies extreme values
                # as local minima
# IMPORTANT: grid near zero should be fine in order to show
# grandular behaviour of mse, PUPE or other error metrics
# near zero e.g. 10^{-3}. ψ corresponds to mse.

NUM_ψ_NEAR_ZERO = 100

def F(μ, σ2, ψ, K, B, E):
    """
    Kuan's potential function in Hsieh2022.
    μ, σ2, ψ are scalars, inputs to the potential function.

    μ is the user density μ = L/ total number of channel uses.
    σ2 is the noise variance.
    ψ is the argument of the potential function, between 0 and E.

    B: number of spreading sequences for each user. B=1 reduces to CDMA where
    each user has only one spreading/signature sequence.

    K is the number of modulation symbols. For my purpose, assume K=2 fixed
    (i.e. binary modulation).
    """
    # assert B == 1 and K == 2, "Only compatible with B=1 and K=2"
    τ = σ2 + μ * ψ
    extra_term = 1 / (2 * μ) * (jnp.log(τ / σ2) - μ * ψ / τ) # term besides the mutual info term
    if B == 1 and K == 2:
        mutual_info = I_trapz_B1_K2(K, B, E, τ)
    else:
        mutual_info = I(K, B, E, τ)
    return mutual_info + extra_term


F_all_ψ = jax.jit(
    jax.vmap(F, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "B"],
)  # length-num_ψ vector
F_all_ψ_σ2 = jax.jit(
    jax.vmap(F_all_ψ, in_axes=(None, 0, None, None, None, None), out_axes=0),
    static_argnames=["K", "B"],
)  # num_σ2 x num_ψ matrix
F_all_ψ_σ2_μ = jax.jit(
    jax.vmap(F_all_ψ_σ2, in_axes=(0, None, None, None, None, None), out_axes=0),
    static_argnames=["K", "B"],
)  # num_μ x num_σ2 x num_ψ tensor


def I(K, B, E, τ):
    """
    Returns I(X; S) for a given τ by evaluating the expectation E_Z[.].
    This mutual info uses base-e logarithms.

    General B>=1, X_sec is one-hot with K-ary modulation.
    K=1 corresponds to no modulation (p1 in Kuans paper).
    K=2 corresponds to binary modulation (p2 in Kuans paper).
    At the moment, only supports K=1 or K=2.

    τ encompasses the information about ψ.
    We use Monte Carlo integration to approximate the integral over the
    Gaussian vector Z of length B. JAX doesnt have multidimensional integration.

    NOTE: this function may be numerically unstable for large B.
    """
    # num_samples should be >> B because we are sampling length-B Gaussian vectors.
    # Result may be inaccurate/ unstable and potentially highly misleading if
    # num_samples isnt large enough.
    num_Z_samples = 100000
    def E_arg(Z):
        """
        Returns the argument of the expectation E_Z[.]. Z is an array of length B.

        Uses sumlogexp trick to avoid numerical overflow.
        """
        if K == 1:
            exp_args = jnp.sqrt(E / τ) * Z
            assert exp_args.shape == (B,)
            exp_args = exp_args.at[0].add(E / τ)
            return jsp.special.logsumexp(exp_args)
        elif K == 2:
            exp_args = jnp.concatenate((jnp.sqrt(E / τ) * Z, -jnp.sqrt(E / τ) * Z))
            assert exp_args.shape == (2 * B,)
            exp_args = exp_args.at[0].add(E / τ)
            exp_args = exp_args.at[B].add(-E / τ)
            return jsp.special.logsumexp(exp_args) - jnp.log(2)
            # 2nd term due to cosh has a factor of 1/2
        else:
            raise NotImplementedError("Not implemented for K>2")

    num_Z_per_batch = int(100000/B) # to ensure the num_Z_per_batch x B matrix doesnt
    # cause memory overflow
    num_batches = num_Z_samples // num_Z_per_batch + 1
    E_term = 0
    for i in range(num_batches):
        print(f"batch i={i}")
        # Same key generates same set of samples:
        Z_samples = random.multivariate_normal(
            key=random.PRNGKey(i), mean=jnp.zeros(B),
            cov=jnp.eye(B), shape=(num_Z_per_batch,)
        )
        assert Z_samples.shape == (num_Z_per_batch, B)
        E_arg_samples = jax.vmap(E_arg, in_axes=0, out_axes=0)(Z_samples)
        assert E_arg_samples.shape == (num_Z_per_batch,)
        E_term = E_term + jnp.mean(E_arg_samples)
    assert i == num_batches - 1
    assert num_Z_per_batch * num_batches >= num_Z_samples
    E_term = E_term / num_batches
    mi = E / τ + jnp.log(B) - E_term
    return mi


I_all_τ = jax.jit(
    jax.vmap(I, in_axes=(None, None, None, 0), out_axes=0), static_argnames=["K", "B"]
)


def I_trapz_B1_K2(K, B, E, τ):
    """
    Same as I(K, B, E, τ) but uses trapezoidal rule instead of MC integration.
    This is much faster than I(K, B, E, τ) and equally accurate.
    This mutual info uses base-e logarithms.

    Limited to B=1, so we only need to evaluate a 1D integral.
    Limited to K=2.
    Z is a scalar standard Gaussian.
    """
    assert B == 1 and K == 2, "Only compatible with B=1 and K=2"

    def integrand(Z):
        exp_arg = E / τ + jnp.sqrt(E / τ) * Z
        exp_args = jnp.array([exp_arg, -exp_arg])
        log_cosh = jsp.special.logsumexp(exp_args) - jnp.log(2)
        # 2nd term due to cosh has a factor of 1/2
        return jsp.stats.norm.pdf(Z) * log_cosh
    integrand_all_Z = jax.jit(jax.vmap(integrand, in_axes=0, out_axes=0))

    num_Z = 100000
    Z_arr = jnp.linspace(-1000, 1000, num_Z)
    # mi = E / τ - jnp.trapz(integrand_all_Z(Z_arr), Z_arr)
    # scipy.integrate.trapezoid is not recognised by JAX, needs to use jsp version:
    # mi = E / τ - jsp.integrate.trapezoid(integrand_all_Z(Z_arr), Z_arr)
    mi = E / τ -  jnp.trapz(integrand_all_Z(Z_arr), Z_arr)

    # jax.scipy.integrate.trapezoid somehow is not recognised
    # jnp.trapz is deprecated but it works
    return mi



# Derivative of F(μ, σ2, ψ, K, B, E) wrt ψ:
dFdψ = jax.jit(jax.grad(F, argnums=2), static_argnames=["K", "B"])
dFdψ_all_ψ = jax.jit(
    jax.vmap(dFdψ, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "B"],
)

# Second derivative of F(μ, σ2, ψ, K, B, E) wrt ψ:
d2Fdψ2 = jax.jit(jax.grad(dFdψ, argnums=2), static_argnames=["K", "B"])
d2Fdψ2_all_ψ = jax.jit(
    jax.vmap(d2Fdψ2, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "B"],
)


def Pe(τ, K, B, E, k=None):
    """
    PUPE as a function of τ.

    K: number of modulation symbols. Can only be K=1 or 2 for now.
    B: >=1. number of spreading sequences for each user.
    eqn (21) of Kuan's GMAC paper.
    """
    assert K == 1 or K == 2, "Not implemented for K>2"
    assert B >= 1, "B must be >=1"

    large_B_thres = 2**50
    if K == 1 and B == 1:
        return 0
    if K == 2 and B == 1:
        return jsp.stats.norm.cdf(-jnp.sqrt(E / τ))
    num_samples = 200000  # 50000000 was used to pass tests
    print(f"num_samples of scalar Z = {num_samples}")

    # Same key generates same set of samples:
    Z_samples = random.normal(
        key=random.PRNGKey(0), shape=(num_samples,) # scalar standard normal
    )
    assert Z_samples.shape == (num_samples,)
    if B < large_B_thres:
        if K == 1:
            def E_arg(Z):
                """Argument of the expectation E_Z[.] where Z is scalar standard Gaussian."""
                Φ = jsp.stats.norm.cdf(jnp.sqrt(E / τ) + Z)
                return Φ ** (B - 1)
        elif K == 2:
            def E_arg(Z):
                """Argument of the expectation E_Z[.] where Z is scalar standard Gaussian."""
                Φ = jsp.stats.norm.cdf(-(jnp.sqrt(E / τ) + Z))
                # Ensure integrand for z<-\sqrt{E/τ} is zero:
                return (1 - 2 * Φ) ** (B - 1) * (Z > -jnp.sqrt(E / τ))
        res = 1 - jnp.mean(jax.vmap(E_arg, in_axes=0, out_axes=0)(Z_samples))
    else: # B >= large_B_thres
        if K == 1:
            ################## Truncated Taylor expansion of (1-Q)^{B-1} ##################
            # This method is inaccurate.
            # def E_arg(Z):
            #     """Use binomial expansion around 1."""
            #     Φ = jsp.stats.norm.cdf(jnp.sqrt(E / τ) + Z)
            #     Q = 1 - Φ
            #     return Q # should be small
            # E_Z = jnp.mean(jax.vmap(E_arg, in_axes=0, out_axes=0)(Z_samples))
            # res = jnp.exp(log_B_1(k) + jnp.log(E_Z))
            ##############################################################################
            ################## Below uses lim_n→∞(1+x/n)^n=e^x ############################
            # This method is more accurate.
            def E_arg(Z):
                Q = jsp.stats.norm.sf(jnp.sqrt(E / τ) + Z) # survival function=1-cdf,
                # it is more accurate than 1-cdf
                tmp = jnp.exp(log_B_1(k) + jnp.log(Q)) # (B-1) * Q
                return jnp.exp(-tmp) # exp(-(B-1)Q)
            res = 1 - jnp.mean(jax.vmap(E_arg, in_axes=0, out_axes=0)(Z_samples))
            ##############################################################################
        else:
            raise NotImplementedError("Not implemented for K>1 and large B")

    return res
Pe_all_τ = jax.jit(
    jax.vmap(Pe, in_axes=(None, None, None, 0), out_axes=0), static_argnames=["K", "B"]
)


def identify_minima(F_arr, dFdψ_arr, d2Fdψ2_arr):
    """
    Identify the largest minimum and global minimum of function F(ψ),
    and return the indices of the minima.

    Each minimum
    - either locates in the middle of the interval, with
      zero gradient and positive curvature;
    - or locates at the ends of the interval, with
      nonzero gradient and any curvature.
    """
    ############### collect minima in the middle of the interval ###############
    # diff returns an array of length len(f)-1:
    # Below gives the idx before each zero crossing:
    zero_cross_idx = jnp.argwhere(jnp.diff(jnp.sign(dFdψ_arr))).flatten()
    # remove zero crossings that are maxima:
    local_min_idx = zero_cross_idx[d2Fdψ2_arr[zero_cross_idx] > 0]
    idx_tol = NUM_ψ_NEAR_ZERO + 0.03 * NUM_ψ  # distance away from the actual minimum. To accommodate
    # inaccuracies in MC estimates.
    # Small deviation doesnt change ψ much so doesnt change Pe much.
    # remove extremes because they dont count as middle minima:
    local_min_idx = local_min_idx[local_min_idx > idx_tol]
    local_min_idx = local_min_idx[local_min_idx < len(F_arr) - idx_tol]

    ############### collect minima at the ends of the interval #################
    # Check whether two extremes are global minima (they cannot be local minima)
    # and if they are add to the set of candidate minima:
    global_min_idx = jnp.argmin(F_arr)
    # I discovered that zero in df may not exactly correspond to a minimum in f.
    # JAX AD is slightly inaccurate.
    global_min_in_middle = (
        jnp.min(jnp.abs(global_min_idx - local_min_idx)) <= idx_tol
        if local_min_idx.size > 0
        else False
    )
    # Below need to match the tolerance used in local_min_idx exactly:
    global_min_at_extremes = (
        global_min_idx <= idx_tol
        or global_min_idx >= len(F_arr) - idx_tol
    )  # left and right to each extreme:
    if (global_min_in_middle or global_min_at_extremes):
        additional_min_idx = [global_min_idx] if global_min_at_extremes else []
        ############## collect all minima ##############
        # Below works with empty arrays or lists
        minima_idx = jnp.concatenate(
            (local_min_idx, jnp.array(additional_min_idx)), dtype=int
        )
    else:
        # Rare case: f oscillates due to inaccurate numerical integration,
        # so that global minimum is not at the extremes and neither does
        # it coincide with the local minima with zero gradient:
        # Trust global minimum instead of local minima with zero gradient:
        minima_idx = jnp.array([global_min_idx], dtype=int)
    assert minima_idx.size > 0

    f_at_minima = F_arr[minima_idx]
    # Below works even if minima_idx is a singleton:
    largest_minimum_idx = minima_idx[jnp.argmax(f_at_minima)]
    global_minimum_idx = minima_idx[jnp.argmin(f_at_minima)]
    return largest_minimum_idx, global_minimum_idx


def identify_minima_argrelmin(F_arr):
    """
    Unlike identify_minima, this function uses F_arr
    (i.e. F evaluated over a fine grid) and argrelmin from scipy
    to identify largest minimum and global minimum of F.

    NOTE: this function returns global minimum fine, but when F is
    not smooth locally, argrelmin returns non-existent local minima,
    so this function shouldnt be used to find local minima.

    This function however is much faster than identify_minima in
    finding the global minimum + it doesnt require derivatives.
    """
    minima_idx = argrelmin(F_arr, order=1)[0] # order=1 means
    # we compare every point with its immediate neighbours
    # to see if it is a local minimum. order=2 means a point
    # is a minimum if it is < neighbours 2 points away.

    if len(minima_idx) == 0:
        if F_arr[0] < F_arr[-1]:
            largest_minimum_idx = 0
            global_minimum_idx = 0
        elif F_arr[0] > F_arr[-1]:
            largest_minimum_idx = len(F_arr) - 1
            global_minimum_idx = len(F_arr) - 1
        else:
            raise ValueError("F_arr[0] = F_arr[-1]")
    else:
        F_at_minima = F_arr[minima_idx]
        largest_minimum_idx = minima_idx[jnp.argmax(F_at_minima)]
        global_minimum_idx = minima_idx[jnp.argmin(F_at_minima)]
        # Global minimum may be endpoints (assuming there arent multiple global minima):
        if F_arr[0] < F_arr[global_minimum_idx]:
            global_minimum_idx = 0
        if F_arr[-1] < F_arr[global_minimum_idx]:
            global_minimum_idx = len(F_arr) - 1
    return largest_minimum_idx, global_minimum_idx


def reproduce_kuan_Fig4_2():
    """This function reproduces Fig.4.2 in Kuan's thesis.
    K=1, B=2, mu=2, no modulation. This no longer runs because
    our F is now restricted to B==1 and K==2 for speed.
    """
    K = 1
    B = 2
    μ = 2
    E = 10  # this scales with τ so its value on its own doesnt matter
    num_ψ = 1000
    ψ_over_E_arr = jnp.linspace(1e-4, 1, num_ψ)
    # ψ_arr = np.linspace(1e-4, E, num_ψ) # [0, E]
    ψ_arr = ψ_over_E_arr * E
    EbN0_dB_arr = jnp.array([8, 9.1, 10.06, 12, 15.8])
    # σ2_arr = np.array([1e-3, 1e-2, 1e-1, 0.5, 1, 2, 3])
    # EbN0_arr = E/ (2 * σ2_arr * np.log2(B))
    # EbN0_dB_arr = 10 * np.log10(EbN0_arr)
    EbN0_arr = 10 ** (EbN0_dB_arr / 10)
    σ2_arr = E / (EbN0_arr * 2 * jnp.log2(B))

    F_arr = np.zeros((len(σ2_arr), num_ψ))
    dFdψ_arr = np.zeros((len(σ2_arr), num_ψ))
    d2Fdψ2_arr = np.zeros((len(σ2_arr), num_ψ))
    for i, σ2 in enumerate(σ2_arr):
        print(f"== σ2={σ2} [{i}/{len(σ2_arr)}]==")
        F_arr[i] = F_all_ψ(μ, σ2, ψ_arr, K, B, E)
        dFdψ_arr[i] = dFdψ_all_ψ(μ, σ2, ψ_arr, K, B, E)
        d2Fdψ2_arr[i] = d2Fdψ2_all_ψ(μ, σ2, ψ_arr, K, B, E)

    #################### Plot ############################
    plt.figure()
    for i, σ2 in enumerate(σ2_arr):
        plt.plot(
            ψ_arr / E,
            F_arr[i],
            label=f"σ2={int(σ2*100)/100}, "
            + r"$E_b/N_0$="
            + f"{int(EbN0_dB_arr[i]*100)/100}dB",
            c=f"C{i}",
        )
        # mark minima of F:
        if False:
            min_idx = jnp.argmin(F_arr[i])
            plt.scatter(
                ψ_arr[min_idx] / E,
                F_arr[i, min_idx],
                c=f"C{i}",
                marker="o",
                markersize=4,
            )
        minima_idx = identify_minima(F_arr[i], dFdψ_arr[i], d2Fdψ2_arr[i])
        print(f"minima: {minima_idx}")
        minima_idx_argrelmin = identify_minima_argrelmin(F_arr[i])
        print(f"minima_argrelmin: {minima_idx_argrelmin}")
        if len(minima_idx_argrelmin) > 0:
            tmp_idx = minima_idx_argrelmin[0]
            plt.scatter(
                ψ_arr[tmp_idx] / E, F_arr[i, tmp_idx], marker="x", c=f"C{i}"
            )
    plt.legend()
    plt.xlabel(r"normalised MSE $\psi/E$")
    plt.ylabel(r"potential function $\mathcal{F}$")
    plt.title(f"K={K}, B={B}, E={E}")
    plt.xscale("log")
    plt.grid()
    plt.savefig("results/Kuan_thesis_fig4_2_potential_fn.pdf")
    plt.show()


def find_critical_σ2(K, B, E, μ_arr, σ2_arr, PUPE=1, log_file_name=None):
    """
    This employs Kuan's potential function for B=1, K=2.
    (because F_all_ψ only supports B=1, K=2)

    For each μ, fix signal power E, decrease σ2 and stop at the first σ2
    that gives Pe < target PUPE asymptotically for both iid and SC designs.

    μ_arr: monotonically increasing array of user densities.
    σ2_arr: monotonically decreasing array of noise variances.

    Can also cope with fixed σ2 (i.e. σ2_arr contains only 1 entry).

    PUPE=1 by default essentially means that we dont care about PUPE
    and just save the Pe for each μ and σ2, because PUPE always < 1.
    """
    assert np.all(μ_arr == np.sort(μ_arr)), \
        "μ_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ψ_over_E_near_zero_arr = 10 ** jnp.linspace(-8, -1, NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.linspace(1e-1, 1, NUM_ψ - NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.concatenate((ψ_over_E_near_zero_arr, ψ_over_E_arr))
    ψ_arr = jnp.array(ψ_over_E_arr * E, dtype=jnp.float32)  # [0, E]

    num_μ = len(μ_arr)
    num_σ2 = len(σ2_arr)

    F_arr = np.zeros((num_μ, num_σ2, NUM_ψ))
    dFdψ_arr = np.zeros((num_μ, num_σ2, NUM_ψ))
    d2Fdψ2_arr = np.zeros((num_μ, num_σ2, NUM_ψ))

    min_F_idx_arr = np.zeros((num_μ, num_σ2, 2))  # ψ indices at the minima of F
    min_F_Pe_arr = np.zeros((num_μ, num_σ2, 2))  # Pe corresponding to the minima of F
    critical_σ2_arr = -np.ones((num_μ, 2))  # σ2 needed for each given μ to
    # achieve UER < target PUPE

    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i in tqdm(range(num_μ)):
        logger.debug(f"==== μ={μ_arr[i]} [{i}/{num_μ}] ====")
        for j in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2={σ2_arr[j]} [{j}/{num_σ2}] ==")
            logger.debug("Starting F_all_ψ")
            F_arr[i, j] = F_all_ψ(
                μ_arr[i], σ2_arr[j], ψ_arr, K, B, E
            )  # length num_ψ vector
            # print(f"F_arr[i, j] = {F_arr[i, j]}")
            dFdψ_arr[i, j] = dFdψ_all_ψ(μ_arr[i], σ2_arr[j], ψ_arr, K, B, E)
            d2Fdψ2_arr[i, j] = d2Fdψ2_all_ψ(μ_arr[i], σ2_arr[j], ψ_arr, K, B, E)
            iid_idx, sc_idx = identify_minima(
                F_arr[i, j], dFdψ_arr[i, j], d2Fdψ2_arr[i, j]
            )
            τ = lambda idx: σ2_arr[j] + μ_arr[i] * ψ_arr[idx]
            Pe_iid, Pe_sc = Pe(τ(iid_idx), K, B, E), \
                Pe(τ(sc_idx), K, B, E)
            assert Pe_iid >= Pe_sc, "Pe_iid must be >= Pe_sc\n" + \
                "This may be because the minima of F are not identified correctly.\n" + \
                "One possibility is the grid of ψ is too fine, such that the minima " + \
                "at the extremes are identified as minima in the middle of the interval.\n"

            min_F_idx_arr[i, j] = [iid_idx, sc_idx]
            min_F_Pe_arr[i, j] = [Pe_iid, Pe_sc]
            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i, 0] == -1) and (Pe_iid < PUPE):
                critical_σ2_arr[i, 0] = σ2_arr[j]
            if (critical_σ2_arr[i, 1] == -1) and (Pe_sc < PUPE):
                critical_σ2_arr[i, 1] = σ2_arr[j]
            # Break the loop over σ2 if both are filled in:
            if np.all(critical_σ2_arr[i] != -1):
                logger.debug(
                    f"critial σ2 for iid = {critical_σ2_arr[i, 0]}\n"
                    + f"critial σ2 for sc = {critical_σ2_arr[i, 1]}\n"
                )
                logger.debug(f"Pe_iid = {Pe_iid}\nPe_sc = {Pe_sc}\n")
                break
        # For next μ, scan σ2_arr from SC critical σ2 onwards for current μ:
        if critical_σ2_arr[i, 1] != -1:
            start_σ2_idx = np.where(σ2_arr == critical_σ2_arr[i, 1])[0][0]
            # When σ2 is fixed, above ensures next round still scans the
            # singleton array.
        else:
            break  # both iid and SC results wont show anything on the plot after this point

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan
    return (
        ψ_arr,
        F_arr,
        dFdψ_arr,
        d2Fdψ2_arr,
        min_F_idx_arr,
        min_F_Pe_arr,
        critical_σ2_arr,
    )



def find_critical_σ2_global_only(K, B, E, μ_arr, σ2_arr, PUPE=1, log_file_name=None):
    """
    This employs Kuan's potential function for B=1, K=2 (uncoded case),
    so payload is k=log2(B)+log2(K)=1. (because F_all_ψ only supports B=1, K=2)

    For each μ, fix signal power E, decrease σ2 and stop at the first σ2
    that gives Pe < target PUPE asymptotically for SC designs, and only SC designs.

    μ_arr: monotonically increasing array of user densities.
    σ2_arr: monotonically decreasing array of noise variances.

    Can also cope with fixed σ2 (i.e. σ2_arr contains only 1 entry).

    PUPE=1 by default essentially means that we dont care about PUPE
    and just save the Pe for each μ and σ2, because PUPE always < 1.
    """
    assert np.all(μ_arr == np.sort(μ_arr)), \
        "μ_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ψ_over_E_near_zero_arr = 10 ** jnp.linspace(-8, -1, NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.linspace(1e-1, 1, NUM_ψ - NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.concatenate((ψ_over_E_near_zero_arr, ψ_over_E_arr))
    ψ_arr = jnp.array(ψ_over_E_arr * E, dtype=jnp.float32)  # [0, E]

    num_μ = len(μ_arr)
    num_σ2 = len(σ2_arr)

    F_arr = np.zeros((num_μ, num_σ2, NUM_ψ))

    min_F_idx_arr = np.zeros((num_μ, num_σ2))  # ψ indices at the minima of F
    min_F_Pe_arr = np.zeros((num_μ, num_σ2))  # Pe corresponding to the minima of F
    critical_σ2_arr = -np.ones(num_μ)  # σ2 needed for each given μ to
    # achieve UER < target PUPE

    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i in tqdm(range(num_μ)):
        logger.debug(f"==== μ={μ_arr[i]} [{i}/{num_μ}] ====")
        for j in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2={σ2_arr[j]} [{j}/{num_σ2}] ==")
            logger.debug("Starting F_all_ψ")
            F_arr[i, j] = F_all_ψ(
                μ_arr[i], σ2_arr[j], ψ_arr, K, B, E
            )  # length num_ψ vector
            logger.debug("Finished F_all_ψ")
            sc_idx = jnp.argmin(F_arr[i, j])
            min_F_idx_arr[i, j] = sc_idx
            τ = lambda idx: σ2_arr[j] + μ_arr[i] * ψ_arr[idx]
            sc_Pe = Pe(τ(sc_idx), K, B, E)
            min_F_Pe_arr[i, j] = sc_Pe

            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i] == -1) and (sc_Pe < PUPE):
                critical_σ2_arr[i] = σ2_arr[j]
                logger.debug(f"Critical σ2 for SC-AMP: {critical_σ2_arr[i]}" + \
                             f"sc_Pe: {sc_Pe}, PUPE: {PUPE}")
                k = np.log2(B) + np.log2(K)
                critical_EbN0_dB = 10 * np.log10(E / (2*k) / critical_σ2_arr[i])
                logger.debug(f"Critical EbN0 (dB) for SC-AMP: {critical_EbN0_dB}")
                start_σ2_idx = j  # start scanning σ2 from this index for the next μ
                break
        if critical_σ2_arr[i] == -1:
            logger.debug(f"Critical σ2 for SC-AMP: not found in range for μ={μ_arr[i]}")
            start_σ2_idx = j

    # Erase the -1 entries in critical_σ2_arr:
    critical_σ2_arr[critical_σ2_arr == -1] = np.nan
    return ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr
