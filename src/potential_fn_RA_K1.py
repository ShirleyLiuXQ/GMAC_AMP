# Potential function to provide ach bounds for random access paper, with K=1.
# K=2 gives overcomplicated pMD, pFA, pAUE expressions.

# For some reason, jsp.integrate.trapezoid doesnt work on HPC, so I have to use
# jnp.trapz instead. Both work on my laptop.
import logging
import jax
jax.config.update("jax_enable_x64", True) # seems to accommodate larger B (64 bits instead of the default 32)
import jax.scipy as jsp
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Unlike in denoiser.py, α here refers to the fraction of active users.
# denoiser.py uses the old notation where α is the fraction of silent users.

# ψ is the variable to minimise the potential function over:
NUM_ψ = 1500  # coarser grid than this gives incorrect minima;
                # finer grid sometimes identifies extreme values
                # as local minima
# IMPORTANT: grid near zero should be fine in order to show
# grandular behaviour of mse, PUPE or other error metrics
# near zero e.g. 10^{-3}. ψ corresponds to mse.

NUM_ψ_NEAR_ZERO = 100

# Analogous to potential_fn_kowshik.py, this potential function can only
# handle up to B=2**(63-1) i.e. k=62 without numerical overflow.

def F_RA_K1(α, B, E, μ, σ2, ψ, num_Zs=100000, η_type: str = "sectionwise"):
    """The potential function for Random Access."""
    τ = σ2 + μ * ψ
    extra_term = 1 / (2 * μ) * (jnp.log(τ / σ2) - μ * ψ / τ) # term besides the mutual info term
    if η_type == "sectionwise":
        mi = I_sectionwise_K1(α, B, E, τ, num_vector_Zs=num_Zs)
    elif η_type == "entrywise":
        mi = B*I_entrywise_K1(α, B, E, τ, num_scalar_Zs=num_Zs)
    else:
        raise ValueError("Invalid η_type")
    return mi + extra_term
F_RA_all_ψ_K1 = jax.jit(
    jax.vmap(F_RA_K1, in_axes=(None, None, None, None, None, 0, None, None),
             out_axes=0), static_argnames=["α", "B", "num_Zs", "η_type"],
)  # length-num_ψ vector

def I_sectionwise_K1(α, B, E, τ, num_vector_Zs=100000):
    """
    Computes I(X_sec; S) where X_sec, Z are both length-B vectors, S=X_sec+√τZ.
    We use base-e logarithms.
    K=1 NO modulation.
    """
    assert B>1, "B>1"
    if α < 1:
        H_X = -(1-α)*jnp.log(1-α) + α*jnp.log(B/α) # base-e H(X_sec)
    else:
        H_X = jnp.log(B)

    def E_arg1(Z):
        """
        Integrand in the first term of -H(X_sec|S), capturing the case when user is silent.
        Z is length-B standard noise vector.
        """
        if α < 1:
            exp_arg = jnp.sqrt(E / τ) * Z.reshape(1,-1)
            exp_args = jnp.append(exp_arg - E / (2*τ), 0)
            assert len(exp_args) == B+1
            b = jnp.append(α/B/(1-α) * jnp.ones(B), 1) # logsum{be^a}
            assert len(b) == B+1
            ln_term = jsp.special.logsumexp(a=exp_args, b=b)
            return -ln_term
        else:
            return 0 # all users active, the term corresponding
            # to silent users is zero
    E_arg1_all = jax.jit(jax.vmap(E_arg1, in_axes=0, out_axes=0))

    def E_arg2(Z):
        """
        Integrand in the second term of -H(X_sec|S), capturing the case when user is active.
        Z is length-B standard noise vector.
        """
        # Z1 term has +E/(2τ) and the rest have -E/(2τ):
        # Z1 is the distinct term, just like in Kuans potential fn implementation,
        # otherwise this wont match Kuans potential fn when α=1.
        exp_arg = jnp.sqrt(E / τ) * Z.reshape(1,-1) + E/(2*τ) * \
            jnp.concatenate((jnp.array([1]), -jnp.ones(B-1))).reshape(1,-1)
        exp_args = jnp.append(exp_arg, 0)
        assert len(exp_args) == B+1
        b = jnp.append(α/B * jnp.ones(B), 1-α) # logsum{be^a}
        ln_term = jsp.special.logsumexp(a=exp_args, b=b)
        return -ln_term
    E_arg2_all = jax.jit(jax.vmap(E_arg2, in_axes=0, out_axes=0))

    num_Z_per_batch = np.max([int(10000/B), 1]) # to ensure the
    # num_Z_per_batch x B matrix doesnt cause memory overflow.
    # Also ensure num_Z_per_batch is at least 1
    num_batches = num_vector_Zs // num_Z_per_batch + 1
    neg_HX_givenS = 0
    print(f"num_batches: {num_batches}")
    for i in tqdm(range(num_batches)):
        # Same key generates same set of samples:
        Z_samples = random.multivariate_normal(
            key=random.PRNGKey(i), mean=jnp.zeros(B),
            cov=jnp.eye(B), shape=(num_Z_per_batch,)
        )
        assert Z_samples.shape == (num_Z_per_batch, B)
        E_arg1_all_Zs = E_arg1_all(Z_samples)
        assert E_arg1_all_Zs.shape == (num_Z_per_batch,)
        E_arg2_all_Zs = E_arg2_all(Z_samples)
        assert E_arg2_all_Zs.shape == (num_Z_per_batch,)
        term_1 = (1-α) * jnp.mean(E_arg1_all_Zs)
        term_2 = α * (jnp.log(α/B) + E/(2*τ) + jnp.mean(E_arg2_all_Zs))
        neg_HX_givenS = neg_HX_givenS + (term_1 + term_2)
    assert i == num_batches-1
    assert num_Z_per_batch * num_batches >= num_vector_Zs
    neg_HX_givenS = neg_HX_givenS / num_batches
    # -H(X_sec|S) = (term_1 + term_2)
    mi = H_X + neg_HX_givenS
    return mi


def I_entrywise_K1(α, B, E, τ, num_scalar_Zs=50000):
    """
    Computes I(X; S) where X, Z are both scalars, S=X+√τZ.
    We use base-e logarithms.
    K=1 binary modulation.
    """
    assert B>1, "B>1"
    H_X = entropy([α/B, 1-α/B]) # base-e H(X)
    # print(f"H_X: {H_X}")

    def integrand_1(Z):
        """
        Integrand in the first term of -H(X|S), capturing the case where
        a specific entry is zero in the one-hot vector.
        Z is standard normal scalar variable.
        """
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        # exp_args = jnp.array([0, exp_arg-jnp.log(B/α-1)]).reshape(1,2)
        # ln_term = jsp.special.logsumexp(exp_args)
        exp_args = jnp.array([exp_arg, 0]).reshape(1,2)
        b = jnp.array([α/(B-α), 1]).reshape(1,-1)
        ln_term = jsp.special.logsumexp(a=exp_args, b=b)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_1_all_Z = jax.jit(jax.vmap(integrand_1, in_axes=0, out_axes=0))
    # integrand_1_all_Z = jax.vmap(integrand_1, in_axes=0, out_axes=0)

    def integrand_2(Z):
        """
        Integrand in the second term of -H(X|S), capturing the case where
        a specific entry is one in the one-hot vector.
        Z is standard normal scalar variable.
        """
        exp_arg = jnp.sqrt(E / τ) * Z - E / (2*τ)
        # exp_args = jnp.array([0, exp_arg+jnp.log(B/α-1)]).reshape(1,2)
        # ln_term = jsp.special.logsumexp(exp_args)
        exp_args = jnp.array([exp_arg, 0]).reshape(1,2)
        b = jnp.array([B/α-1, 1]).reshape(1,-1)
        ln_term = jsp.special.logsumexp(a=exp_args, b=b)
        return jsp.stats.norm.pdf(Z) * ln_term
    integrand_2_all_Z = jax.jit(jax.vmap(integrand_2, in_axes=0, out_axes=0))
    # integrand_2_all_Z = jax.vmap(integrand_2, in_axes=0, out_axes=0)

    # The integrand has to have scalar args:
    Z_arr = jnp.linspace(-10000, 10000, num_scalar_Zs)
    term_1 = -(1-α/B) * jnp.trapz(integrand_1_all_Z(Z_arr), Z_arr)
    # term_1 = -(1-α/B) * jsp.integrate.trapezoid(integrand_1_all_Z(Z_arr), Z_arr)
    term_2 = -α/B * jnp.trapz(integrand_2_all_Z(Z_arr), Z_arr)
    # term_2 = -α/B * jsp.integrate.trapezoid(integrand_2_all_Z(Z_arr), Z_arr)
    # For some reason, my laptop runs jsp.integrate.trapezoid without problems,
    # but HPC doesnt recognize jsp.integrate.trapezoid, only jnp.trapz.
    neg_HX_givenS = term_1 + term_2
    mi = H_X + neg_HX_givenS
    return mi

def ξ_f(B, α, E, τ):
    """A shorthand used in pMD, pFA, pAUE expressions."""
    return jnp.log((1-α) * B / α) / jnp.sqrt(E/τ)

def pErrors_K1(α, τ, B, E, num_scalar_Zs: int=50000, apply_exp_trick: bool = False):
    """
    pMD, pFA, pAUE in the sectionwise hard-decision step for K=1.

    apply_exp_trick: by default doesnt apply the (1-Q(sf_thres))^B≈exp(-Q(sf_thres)B) trick
    for small Q and large B. Can choose True when B is large, and the arg of Q, ξ - tmp_arg > sf_thres.
    """
    sf_thres = 10 # threshold below which we use (1-Q(sf_thres))^B instead of exp(-Q(sf_thres)B)
    if α == 0: # No active users
        return 0, 0, 0
    else:
        tmp_arg = jnp.sqrt(E/τ)/2
        Z_arr = jnp.linspace(-10000, 10000, num_scalar_Zs) # (-1000, 1000)
        # is not enough to avoid numerical issues (e.g. returns pAUE<0)
        if α == 1: # All users active
            pMD, pFA = 0, 0
            def integrand_pAUE_α0(Z):
                """
                Integrand in the expectation term in pAUE when α=0.
                Z is a sample from the scalar standard normal distribution.
                ξ will be - infinity.
                """
                arg = jsp.stats.norm.cdf(Z + tmp_arg*2) ** (B-1)
                # arg = jnp.clip(arg, 0, 1) # Due to numerical errors arg may be slightly >1
                return jsp.stats.norm.pdf(Z) * arg
            integrand_pAUE_α0_all_Z = jax.jit(jax.vmap(integrand_pAUE_α0,
                                            in_axes=0, out_axes=0))
            # pAUE = 1 - jsp.integrate.trapezoid(integrand_pAUE_α0_all_Z(Z_arr), Z_arr)
            pAUE = 1 - jnp.trapz(integrand_pAUE_α0_all_Z(Z_arr), Z_arr)
        else: # α is between 0 and 1
            ξ = ξ_f(B, α, E, τ)
            # term1 = jsp.stats.norm.cdf(ξ - tmp_arg)
            Q1 = jsp.stats.norm.sf(ξ - tmp_arg)
            term1 = 1 - Q1
            # term2 = jsp.stats.norm.cdf(ξ + tmp_arg)
            Q2 = jsp.stats.norm.sf(ξ + tmp_arg)
            term2 = 1 - Q2
            # if jnp.log2(B) <= 20:
            if apply_exp_trick and ξ - tmp_arg > sf_thres: # Q1, Q2 are very small
                pMD = term1 * jnp.exp(-Q2*(B-1))
                tmp_denom_pFA = 1-jnp.exp(-Q2*B)
            else:
                pMD = term1 * term2 ** (B-1)
                tmp_denom_pFA = 1-term2**B

            if tmp_denom_pFA == 0: # Avoid division by zero
                pFA = 0
            else:
                denom_pFA = α * (1-pMD) / ((1-α) * tmp_denom_pFA) + 1
                pFA = 1/denom_pFA

            if apply_exp_trick and ξ - tmp_arg > sf_thres:
                def integrand_pAUE(Z):
                    """
                    Integrand in the expectation term in pAUE. Z is a sample from the
                    scalar standard normal distribution.
                    """
                    thres = jnp.max(jnp.array([ξ+tmp_arg, Z + tmp_arg*2]))
                    Q = jsp.stats.norm.sf(thres)
                    # if np.log2(B) <= 20: # jnp.log2(B) doesnt work with jit
                    arg = jnp.exp(-Q*(B-1))
                    return jsp.stats.norm.pdf(Z) * arg
                integrand_pAUE_all_Z = jax.vmap(integrand_pAUE, in_axes=0, out_axes=0)
            else:
                def integrand_pAUE(Z):
                    """
                    Integrand in the expectation term in pAUE. Z is a sample from the
                    scalar standard normal distribution.
                    """
                    thres = jnp.max(jnp.array([ξ+tmp_arg, Z + tmp_arg*2]))
                    Q = jsp.stats.norm.sf(thres)
                    arg = (1-Q) ** (B-1)
                    # NOTE: jsp.stats.norm.cdf(thres) is sometimes slightly >1.
                    # Survival function below also has the same issue.
                    # arg = (1-jsp.stats.norm.sf(thres)) ** (B-1)
                    # arg = jnp.clip(arg, 0, 1) # Due to numerical errors arg may be slightly negative (seems ok now).
                    return jsp.stats.norm.pdf(Z) * arg
                integrand_pAUE_all_Z = jax.jit(jax.vmap(integrand_pAUE, in_axes=0, out_axes=0))

            # pAUE requires integral over Z, scalar standard normal variable:
            # pAUE = 1 - jsp.integrate.trapezoid(integrand_pAUE_all_Z(Z_arr), Z_arr)
            pAUE = 1 - jnp.trapz(integrand_pAUE_all_Z(Z_arr), Z_arr)
        pAUE = jnp.clip(pAUE, 0, 1) # Due to numerical inaccuracies pAUE may be slightly negative
        assert pMD >= 0 and pFA >= 0 and pAUE >= 0, \
            f"pMD={pMD}, pFA={pFA}, pAUE={pAUE} should be non-negative"
        assert pMD <= 1 and pFA <= 1 and pAUE <= 1, \
            f"pMD={pMD}, pFA={pFA}, pAUE={pAUE} should be at most 1"
        return pMD, pFA, pAUE

def find_RA_K1_critical_σ2(α, B, E, μ_arr, σ2_arr, pError=1,
        num_Zs_pot_fn=100000, num_Zs_pError=10000,
        η_type: str = "sectionwise", apply_exp_trick: bool=False, log_file_name=None):
    """
    For the purpose of getting achievability bounds, only need to find
    global minimum instead of all local minima, so we will simply compute
    F itself without evaluating its derivatives.

    μ = L/total_num_channel_uses.
    μ_a = L*α/total_num_channel_uses.

    NOTE: pError refers to max{pMD, pFA} + pAUE.
    num_Zs_pot_fn: number Z samples (scalar or vector) for potential function evaluation.
    num_Zs_pError: number Z samples (scalar) for pMD, pFA, pAUE evaluation.
    """
    if not logger.hasHandlers():
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    assert np.all(μ_arr == np.sort(μ_arr)), \
        "μ_arr must be monotonically increasing"
    assert np.all(σ2_arr == np.sort(σ2_arr)[::-1]), \
        "σ2_arr must be monotonically decreasing"

    k = int(np.log2(B)) # user payload in bits, K=1.
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
            F_arr[i, j] = F_RA_all_ψ_K1(α, B, E,
                μ_arr[i], σ2_arr[j], ψ_arr, num_Zs_pot_fn, η_type)  # length num_ψ vector
            sc_idx = jnp.argmin(F_arr[i, j])
            min_F_idx_arr[i, j] = sc_idx
            τ = lambda idx: σ2_arr[j] + μ_arr[i] * ψ_arr[idx]
            pMD, pFA, pAUE = pErrors_K1(α, τ(sc_idx), B, E, num_Zs_pError, apply_exp_trick)
            sc_Pe = max(pMD, pFA) + pAUE
            min_F_Pe_arr[i, j] = sc_Pe
            logger.debug(f"pMD: {pMD}, pFA: {pFA}, pAUE: {pAUE}, Pe: {sc_Pe}")
            if False:
                plt.figure()
                plt.plot(ψ_arr, F_arr[i, j])
                plt.scatter(ψ_arr[sc_idx], F_arr[i, j, sc_idx], color='red', marker='o')
                plt.xlabel("ψ")
                plt.ylabel("F")

            # If critical σ2 hasn't been filled in, and Pe < target PUPE, fill it in:
            if (critical_σ2_arr[i] == -1) and (sc_Pe < pError):
                critical_σ2_arr[i] = σ2_arr[j]
                logger.debug(f"Critical σ2 for SC-AMP: {critical_σ2_arr[i]} " + \
                             f"Pe: {sc_Pe}, target Pe: {pError}")
                critical_EbN0_dB = 10 * np.log10(E / (2*k) / critical_σ2_arr[i])
                logger.debug(f"Critical EbN0 (dB) for SC-AMP: {critical_EbN0_dB}")
                start_σ2_idx = j  # start scanning σ2 from this index for the next μ
                break
        if critical_σ2_arr[i] == -1:
            logger.debug(f"Critical σ2 for SC-AMP: not found in range for μ={μ_arr[i]}")
            start_σ2_idx = j

    critical_σ2_arr[critical_σ2_arr == -1] = np.nan
    return ψ_arr, F_arr, min_F_idx_arr, min_F_Pe_arr, critical_σ2_arr
