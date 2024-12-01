import logging
import numpy as np
import jax
jax.config.update("jax_enable_x64", True) # seems to accommodate larger B (64 bits instead of the default 32)
jax.config.update("jax_debug_nans", True) # break on nans
import jax.scipy as jsp
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

from src.potential_fn import Pe
from src.helper_fns import log_B_1


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ψ is the variable to minimise the potential function over:
NUM_ψ = 2000  # coarser grid than this gives incorrect minima;
                # finer grid sometimes identifies extreme values
                # as local minima
# IMPORTANT: grid near zero should be fine in order to show
# grandular behaviour of mse, PUPE or other error metrics
# near zero e.g. 10^{-3}. ψ corresponds to mse.

NUM_ψ_NEAR_ZERO = 100

# NOTE: 16/04/2024:
# I_trapz_marginal_K1 and I_trapz_marginal_K1_nodrv are two implementations
# of the same mutual info assuming marginal denoiser. The former ensures numerical
# stability in the derivative of I wrt ψ, while the latter doesnt. Although we
# only ever need to compute the potential function F itself, not its derivative
# in determining the critical σ2, we use the former which seems to be more stable.

# NOTE: jitting Fm_all_ψ leads to overflow errors when B>2**(63-1).
# i.e. B cannot be represented as int64. The error msg I often see is:
# OverflowError: An overflow was encountered while parsing an argument to a jitted
# computation, whose argument path is x1.
# ... raise OverflowError(f"Python int {value} too large to convert to {dtype}")
# OverflowError: Python int 1125899906842624 too large to convert to int32
#
# Without jax.jit, the functions work fine, most likely due to implicit
# value-dependent semantics like numpy does.
def F_marginal(μ, σ2, ψ, K, k, E):
    """
    Kowshik's potential function in [Kowshik 2022] where the mutual info term
    is computed using the entrywise marginal distribution of X_sec.
    Require B>1 because B=1 implies each section is always 1.

    NOTE: this function overflows for large k (e.g. k=360). In such cases
    use Zadik's ach bound instead which is tight for large k.
    """
    assert K == 1, "Only compatible with K=1"
    B = 2**k
    assert B > 1, "B>1"
    τ = σ2 + μ * ψ
    res = I_trapz_marginal_K1(K, k, E, τ) + 1/(2*μ*B) * (jnp.log(τ/σ2) - μ*ψ/τ)
    # res = I_trapz_marginal_K1_nodrv(K, k, E, τ) + 1/(2*μ*B) * (jnp.log(τ/σ2) - μ*ψ/τ)
    # When B is large, res may underflow so we scale res up, because
    # the utility of F (its minima) doesnt change with constant scaling factors.
    # if k > 100:
    #     res = res * 1e5
    return res


def I_trapz_marginal_K1(K, k, E, τ):
    """
    K=1 i.e. no modulation.
    X_sec = √Ee meaning the entry-wise marginal distribution is
    X_sec(i)=√E w.p. 1/B or zero w.p. 1-1/B.
    This function computes I(X_sec(i); S) where S=X_sec(i)+√τZ,
    and it uses base-e logarithms.

    Recall B=2**k.
    """
    assert K==1, "Only compatible with K=1"
    B = 2**k
    assert B>1, "B>1"
    H_X = entropy([1/B, 1-1/B]) # base-e
    # print(f"H_X: {H_X}")

    def integrand_1(Z):
        """Integrand in the first term of H(X|S)"""
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        # exp_args = jnp.array([0, log_B_1(k)+exp_arg])
        # for numerical stability in not only F but its derivatives, factor (B-1) out:
        exp_args = jnp.array([-log_B_1(k), exp_arg])
        ln_term = jsp.special.logsumexp(exp_args, axis=0) + log_B_1(k)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_1_all_Z = jax.jit(jax.vmap(integrand_1, in_axes=0, out_axes=0))
    # integrand_1_all_Z = jax.vmap(integrand_1, in_axes=0, out_axes=0)

    def integrand_2(Z):
        """Integrand in the second term of H(X|S)"""
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        exp_args = jnp.array([exp_arg, log_B_1(k)])
        ln_term = jsp.special.logsumexp(exp_args, axis=0)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_2_all_Z = jax.jit(jax.vmap(integrand_2, in_axes=0, out_axes=0))
    # integrand_2_all_Z = jax.vmap(integrand_2, in_axes=0, out_axes=0)

    num_Z = 80000
    # -100 to 100 isnt enough to give non-trivial F.
    Z_arr = jnp.linspace(-10000, 10000, num_Z)
    term_1 = 1/B * jnp.trapz(integrand_1_all_Z(Z_arr), Z_arr)
    # term_1 = 1/B * jsp.integrate.trapezoid(integrand_1_all_Z(Z_arr), Z_arr)
    term_2 = (1-1/B) * (jnp.trapz(integrand_2_all_Z(Z_arr), Z_arr) - log_B_1(k))
    # term_2 = (1-1/B) * (jsp.integrate.trapezoid(integrand_2_all_Z(Z_arr), Z_arr)
                        #  - log_B_1(k))
    # For some reason, my laptop runs jsp.integrate.trapezoid without problems,
    # but HPC doesnt recognize jsp.integrate.trapezoid, only jnp.trapz.
    mi = H_X - (term_1 + term_2)
    return mi

Fm_all_ψ = jax.jit(
    jax.vmap(F_marginal, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "k"],
)  # length-num_ψ vector

# Fm_all_ψ = jax.vmap(F_marginal, in_axes=(None, None, 0, None, None, None), out_axes=0)  # length-num_ψ vector


# Derivative of F_marginal(μ, σ2, ψ, K, B, E) wrt ψ:
dFmdψ = jax.jit(jax.grad(F_marginal, argnums=2), static_argnames=["K", "k"])
dFmdψ_all_ψ = jax.jit(
    jax.vmap(dFmdψ, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "k"],
)

# dFmdψ = jax.grad(F_marginal, argnums=2)
# dFmdψ_all_ψ = jax.vmap(dFmdψ, in_axes=(None, None, 0, None, None, None), out_axes=0)

# Second derivative of F_marginal(μ, σ2, ψ, K, B, E) wrt ψ:
d2Fmdψ2 = jax.jit(jax.grad(dFmdψ, argnums=2), static_argnames=["K", "k"])
d2Fmdψ2_all_ψ = jax.jit(
    jax.vmap(d2Fmdψ2, in_axes=(None, None, 0, None, None, None), out_axes=0),
    static_argnames=["K", "k"],
)

def I_trapz_marginal_K1_nodrv(K, k, E, τ):
    """
    Compared to I_trapz_marginal_K1, this function doesnt worry
    about numerical stability in the derivative of I or F.

    K=1 i.e. no modulation.
    X_sec = √Ee meaning the entry-wise marginal distribution is
    X_sec(i)=√E w.p. 1/B or zero w.p. 1-1/B.
    This function computes I(X_sec(i); S) where S=X_sec(i)+√τZ,
    and it uses base-e logarithms.

    Recall B=2**k.
    """
    assert K==1, "Only compatible with K=1"
    B = 2**k
    assert B>1, "B>1"
    H_X = entropy([1/B, 1-1/B]) # base-e
    # print(f"H_X: {H_X}")

    def integrand_1(Z):
        """Integrand in the first term of H(X|S)"""
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        exp_args = jnp.array([0, log_B_1(k)+exp_arg])
        ln_term = jsp.special.logsumexp(exp_args, axis=0)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_1_all_Z = jax.jit(jax.vmap(integrand_1, in_axes=0, out_axes=0))

    def integrand_2(Z):
        """Integrand in the second term of H(X|S)"""
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        exp_args = jnp.array([exp_arg-log_B_1(k), 0])
        ln_term = jsp.special.logsumexp(exp_args, axis=0)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_2_all_Z = jax.jit(jax.vmap(integrand_2, in_axes=0, out_axes=0))

    num_Z = 50000
    # -100 to 100 isnt enough to give non-trivial F.
    Z_arr = jnp.linspace(-10000, 10000, num_Z)
    # jsp.quad?
    term_1 = 1/B * jnp.trapz(integrand_1_all_Z(Z_arr), Z_arr)
    # term_1 = 1/B * jsp.integrate.trapezoid(integrand_1_all_Z(Z_arr), Z_arr)
    term_2 = (1-1/B) * (jnp.trapz(integrand_2_all_Z(Z_arr), Z_arr))
    # term_2 = (1-1/B) * (jsp.integrate.trapezoid(integrand_2_all_Z(Z_arr), Z_arr)
                        #  - log_B_1(k))
    # For some reason, my laptop runs jsp.integrate.trapezoid without problems,
    # but HPC doesnt recognize jsp.integrate.trapezoid, only jnp.trapz.
    mi = H_X - (term_1 + term_2)
    return mi

def Pe_marginal(τ, K, k, E):
    """PUPE given the hard decision step uses entrywise MAP decoding."""
    # B = 2**k
    # assert B>1, "B>1"
    assert k >= 1, "k>=1, because require B=2^k>1"
    assert K == 1, "Only compatible with K=1"
    # MAP classification threshold: S > thres -> X=√E, S < thres -> X=0
    thres = τ * log_B_1(k) / jnp.sqrt(E) + jnp.sqrt(E)/2
    term_1 = jsp.stats.norm.cdf((thres - jnp.sqrt(E)) / jnp.sqrt(τ))
    # term_2 = (B-1) * (1 - jsp.stats.norm.cdf(thres / jnp.sqrt(τ))) # overflows
    # for large B, use log instead:
    tmp = 1 - jsp.stats.norm.cdf(thres / jnp.sqrt(τ))
    term_2 = jnp.exp(log_B_1(k) + jnp.log(tmp)) # turns out the rate at which
    # tmp decreases to zero surpasses the rate at which log_B_1(k) increases to infinity
    return term_1 + term_2


def find_kowshik_critical_σ2(K, k, E, μ_arr, σ2_arr, PUPE=1,
        hard_dec_step = "entrywise", log_file_name=None):
    """
    For the purpose of getting achievability bounds, only need to find
    global minimum instead of all local minima, so we will simply compute
    F itself without evaluating its derivatives.

    hard_dec_step: "entrywise" or "sectionwise" MAP decoding in the
    hard decision step after SC-AMP has converged.
    """
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file_name) #, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    assert np.all(μ_arr == np.sort(μ_arr)), \
        "μ_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    if hard_dec_step == "entrywise":
        logger.debug("Using entrywise MAP decoding as hard-decision step")
        Pe_f = Pe_marginal # inputs are τ, K, k, E
    elif hard_dec_step == "sectionwise":
        logger.debug("Using sectionwise MAP decoding as hard-decision step")
        Pe_f = lambda τ, K, k, E: Pe(τ, K, 2**k, E, k)
    else:
        raise ValueError("hard_dec_step must be 'entrywise' or 'sectionwise'")

    ψ_over_E_near_zero_arr = 10 ** jnp.linspace(-8, -1, NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.linspace(1e-1, 1, NUM_ψ - NUM_ψ_NEAR_ZERO)
    ψ_over_E_arr = jnp.concatenate((ψ_over_E_near_zero_arr, ψ_over_E_arr))

    ψ_arr = jnp.array(ψ_over_E_arr * E, dtype=jnp.float32)  # [0, E]

    num_μ = len(μ_arr)
    num_σ2 = len(σ2_arr)
    F_arr = np.zeros((num_μ, num_σ2, NUM_ψ))
    min_F_idx_arr = np.zeros((num_μ, num_σ2))  # ψ indices at the global minimum of F
    min_F_Pe_arr = np.zeros((num_μ, num_σ2))  # Pe corresponding to the global minimum of F
    critical_σ2_arr = -np.ones(num_μ)  # σ2 needed for each given μ to
    # achieve UER < target PUPE

    start_σ2_idx = 0  # For first μ, scan σ2 from the beginning of σ2_arr
    for i in tqdm(range(num_μ)):
        logger.debug(f"==== μ={μ_arr[i]} [{i}/{num_μ}] ====")
        for j in range(start_σ2_idx, num_σ2):
            logger.debug(f"== σ2={σ2_arr[j]} [{j}/{num_σ2}] ==")
            F_arr[i, j] = Fm_all_ψ(
                μ_arr[i], σ2_arr[j], ψ_arr, K, k, E)  # length num_ψ vector
            sc_idx = jnp.argmin(F_arr[i, j])
            min_F_idx_arr[i, j] = sc_idx
            τ = lambda idx: σ2_arr[j] + μ_arr[i] * ψ_arr[idx]
            sc_Pe = Pe_f(τ(sc_idx), K, k, E)
            min_F_Pe_arr[i, j] = sc_Pe
            if False:
                plt.figure()
                plt.plot(ψ_arr, F_arr[i, j])
                plt.scatter(ψ_arr[sc_idx], F_arr[i, j, sc_idx], color='red', marker='o')
                plt.xlabel("ψ")
                plt.ylabel("F")

            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i] == -1) and (sc_Pe < PUPE):
                critical_σ2_arr[i] = σ2_arr[j]
                logger.debug(f"Critical σ2 for SC-AMP: {critical_σ2_arr[i]}" + \
                             f"sc_Pe: {sc_Pe}, PUPE: {PUPE}")
                critical_EbN0_dB = 10 * np.log10(E / (2*k) / critical_σ2_arr[i])
                logger.debug(f"Critical EbN0 (dB) for SC-AMP: {critical_EbN0_dB}")
                start_σ2_idx = j  # start scanning σ2 from this index for the next μ
                break
        if critical_σ2_arr[i] == -1:
            logger.debug(f"Critical σ2 for SC-AMP: not found in range for μ={μ_arr[i]}")
            start_σ2_idx = j

    critical_σ2_arr[critical_σ2_arr == -1] = np.nan
    return ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr
