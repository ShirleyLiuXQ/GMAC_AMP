import jax.numpy as jnp
import jax
import jax.scipy as jsp
import numpy as np
import sys, os
from jax import config
from ldpc_jossy.py.ldpc import code, decode
from ldpc_jossy.py.ldpc_awgn import bpsk, ch2llr
config.update("jax_enable_x64", True)
# numpy uses float64 by default but jax uses float32
# use jax_enable_x64=True to use float64 in jax for fair comparison with numpy
from numba import jit

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from src.init_model import create_codebook

# The denoisers here assume E=1, i.e., the signal entries are of magnitude 1 or zero.
@jit(nopython=True)
def app2eta_bp(app: np.array) -> np.array:
    """
    app: LLR of the posterior probability of each codeword bit being
    +1 (binary 0) versus -1 (binary 1).

    Converts app to output of eta.
    """
    eta = 1 - 2.0 / (1 + np.exp(app))
    assert np.all(eta >= -1) and np.all(eta <= 1)
    return eta


@jit(nopython=True)
def etaJac_diag_entries(eta, sigma2_arr):
    return (1 - eta ** 2) / sigma2_arr # Assuming E=1


@jit(nopython=True)
def calc_p_nz_cw(alpha, num_codewords):
    return (1 - alpha) / num_codewords # prob of nonzero codeword

class Denoiser_jax:
    def __init__(self, α: float=0, codebook: np.array=None, type: str='mmse', \
                ldpc_code: code=None, bp_max_itcount: int=None) -> None:
        """
        α refers to fraction of silent users.
        α=0 means all users are active.
        """
        self.α = α
        self.codebook = codebook
        self.type = type
        self.ldpc_code = ldpc_code
        self.bp_max_itcount = bp_max_itcount
        if codebook is None and ldpc_code is None:
            # This is the uncoded case:
            self.codebook = create_codebook(d=1)
            # mmse or mmse-marginal denoisers are equivalent for uncoded case:
            # Choose mmse-marginal denoiser because
            # it is compatible with any d
            assert type == 'mmse-marginal' or type == 'mmse-marginal-thesis' \
                or type == 'thres'
        if self.type == 'mmse':
            # Initialise using Denoiser_jax(α, codebook, type='mmse')
            assert self.codebook is not None, "codebook must be provided."
            assert self.ldpc_code is None, "ldpc_code cannot use mmse denoiser."
        elif self.type == 'mmse-marginal' or type == 'mmse-marginal-thesis':
            # Initialise using Denoiser_jax(α, type='mmse-marginal')
            # Linear codes all share the same bitwise marginal distribution:
            # +1/-1 appear with equal probability, and 0 appears with prob α.
            assert (self.codebook is not None) != (self.ldpc_code is not None), \
                "mmse-marginal is compatible with both simple linear codes and " + \
                    "LDPC codes. Either codebook or ldpc_code must be provided, " + \
                        "but not both."
        elif self.type == 'ldpc-bp':
            # Initialise using Denoiser_jax(α, type='ldpc-bp', ldpc_code, bp_max_itcount)
            # TODO: I think my code is already compatible with α>0.
            assert self.α == 0, "Only α=0 is supported for ldpc-bp."
            assert self.ldpc_code is not None, "ldpc_code must be provided."
            assert self.bp_max_itcount is not None, \
                "bp_max_itcount must be provided."
            assert self.codebook is None, "codebooks of LDPC codes must be None."
        elif self.type == 'thres':
            # Only works for RA without outer code.
            self.codebook = create_codebook(d=1) # uncoded
            assert self.ldpc_code is None, "ldpc_code cannot use thres denoiser."
        else:
            raise NotImplementedError

    def η_sum_dη_XMAP(self, S, Σ_or_σ2, calc_sum_dη: bool=False, calcMAP_estimate: bool=False):
        """
        Unified interface for all denoisers:
        η_mmse and η_bp take in Σ, while η_marginal_mmse takes in σ2.

        TODO: when σ2=0, return η=S, X_MAP=S. This isn't implemented yet.
        """
        if self.type == 'mmse':
            η, X_MAP = η_mmse_xMAP_all_s(S, Σ_or_σ2, self.α, self.codebook, calcMAP_estimate)
            sum_dη = sum_dη_mmse(S, Σ_or_σ2, self.α, self.codebook) if calc_sum_dη else None
        elif self.type == 'mmse-marginal':
            # TODO: replace mmse-marginal with mmse-marginal-thesis
            η, X_MAP = η_marginal_mmse_xMAP_all_s(S, Σ_or_σ2, self.α, calcMAP_estimate)
            sum_dη = sum_dη_marginal_mmse(S, Σ_or_σ2, self.α) if calc_sum_dη else None
        elif self.type == 'mmse-marginal-thesis':
            η, X_MAP = η_marginal_mmse_xMAP_all_s_thesis(S, Σ_or_σ2, self.α, calcMAP_estimate)
            sum_dη = sum_dη_marginal_mmse_thesis_analytical(S, Σ_or_σ2, self.α) if calc_sum_dη else None
        elif self.type == 'ldpc-bp':
            # This functionis fully numpy, not jax.
            η, sum_dη, X_MAP = η_sum_dη_bp_xMAP_all_s(S, Σ_or_σ2, \
                self.ldpc_code, self.bp_max_itcount, calc_sum_dη, calcMAP_estimate)
        elif self.type == 'thres':
            # Only works for RA without outer code.
            η, sum_dη, X_MAP = η_thres_sum_dη_XMAP_all_s(S, Σ_or_σ2, self.α, calcMAP_estimate)
        else:
            raise NotImplementedError
        return η, sum_dη, X_MAP

    # @partial(jax.jit, static_argnames=['self', 'calc_sum_dη', 'calcMAP_estimate'])
    # def _η_sum_dη_XMAP(self, S, Σ, calc_sum_dη, calcMAP_estimate):
    #     η, X_MAP = η_mmse_xMAP_all_s(S, Σ, self.α, self.codebook, calcMAP_estimate)
    #     sum_dη = sum_dη_mmse(S, Σ, self.α, self.codebook) if calc_sum_dη else None
    #     return η, sum_dη, X_MAP

# η_mmse should only ever be applied to small d, so we vectorized across d.
def η_mmse_xMAP(s, Σ, α, codebook, calcMAP_estimate=False):
    """
    s: 1xd row vector
    Σ: dxd covariance matrix
    codebook: num_codewords x d
    returns η: 1xd row vector
    """
    s = s.reshape(1, -1)
    num_cw, d = codebook.shape
    Σ = jnp.array([[Σ]]) if jnp.ndim(Σ) == 0 else Σ + np.eye(d)*1e-12 # for numerical stability
    # if jnp.allclose(Σ, 0):
    #     η = s
    if α == 1: # codewords are all-zero
        η = jnp.zeros_like(s, dtype=jnp.float64)
        x_MAP = jnp.zeros_like(s, dtype=jnp.float64) if calcMAP_estimate else jnp.nan
    else:
        p_nz_cw = calc_p_nz_cw(α, num_cw)
        def γ(c, s):
            return -1/2 * (c.reshape(1, -1) - 2 * s.reshape(1, -1)) @ \
                jnp.linalg.pinv(Σ) @ c.reshape(-1, 1)
        γ_all_c = jax.jit(jax.vmap(γ, in_axes=(0, None), out_axes=0))(\
            codebook, s).reshape(num_cw, 1)
        max_γ = jnp.max(γ_all_c)
        # jax.debug.breakpoint()
        # jax.debug.print("{max_γ}", max_γ=max_γ)
        val_log_num, sgn_num = jsp.special.logsumexp(
            a=γ_all_c-max_γ, b=codebook, axis=0, return_sign=True)
        val_log_denom, sgn_denom = jsp.special.logsumexp(
            a=γ_all_c-max_γ, axis=0, return_sign=True)
        η = p_nz_cw * sgn_num*jnp.exp(val_log_num) / \
            (α*jnp.exp(-max_γ) + p_nz_cw * sgn_denom*jnp.exp(val_log_denom))
        # jax.debug.breakpoint()
        if calcMAP_estimate:
            γ_zero_cw = 0
            posteriors = jnp.append(p_nz_cw * jnp.exp(γ_all_c-max_γ), \
                             α*jnp.exp(γ_zero_cw-max_γ)).reshape(-1,)
            # posteriors is (num_cw+1) x 1
            concat_codebook = jnp.concatenate((codebook, jnp.zeros((1, d))), axis=0)
            x_MAP = concat_codebook[np.argmax(posteriors), :]
        else:
            x_MAP = jnp.nan
    return η, x_MAP
η_mmse_xMAP = jax.jit(η_mmse_xMAP, static_argnames=['α', 'calcMAP_estimate'])

def η_mmse(s, Σ, α, codebook):
    """Returns only η, not x_MAP."""
    return η_mmse_xMAP(s, Σ, α, codebook, calcMAP_estimate=False)[0]
η_mmse = jax.jit(η_mmse, static_argnames=['α'])


def η_mmse_xMAP_all_s(S, Σ, α, codebook, calcMAP_estimate=False):
    """
    Input S: num_rows x d
    Output η: num_rows x d
    """
    num_rows, d = S.shape
    η_all, X_MAP = jax.vmap(η_mmse_xMAP, in_axes=(0, None, None, None, None),
                    out_axes=0)(S, Σ, α, codebook, calcMAP_estimate)
    X_MAP = X_MAP.reshape(num_rows, d) if calcMAP_estimate else None
    return η_all.reshape(num_rows, d), X_MAP
η_mmse_xMAP_all_s = jax.jit(η_mmse_xMAP_all_s, static_argnames=['α', 'calcMAP_estimate'])

# Derivative of η_mmse with respect to s.
dη_mmse = jax.jit(jax.jacfwd(η_mmse, argnums=0), static_argnames=['α'])
# dη_mmse = jax.jacfwd(η_mmse, argnums=0)
# returns dxd covariance matrix

def sum_dη_mmse(S, Σ, α, codebook):
    """
    Input S: num_rows x d
    Output sum_dη: dxd matrix which is the sum of num_rows dxd Jacobian matrices

    AD is prone to nans output when there are overflow/underflow in η itself.
    """
    num_rows, d = S.shape
    # assert np.linalg.eigvals(Σ).min() > 0, "Σ must be positive definite."
    dη_mmse_all_s = jax.vmap(dη_mmse, in_axes=(0, None, None, None), out_axes=0)(
                S, Σ, α, codebook).reshape(num_rows, d, d)
    # print(f"dη_mmse_all_s = {dη_mmse_all_s}")
    # jax.debug.breakpoint()
    # assert not np.any(np.isnan(dη_mmse_all_s))
    return jnp.sum(dη_mmse_all_s, axis=0)
sum_dη_mmse = jax.jit(sum_dη_mmse, static_argnames=['α'])


def η_marginal_mmse_xMAP(s, σ2, α, calcMAP_estimate=False):
    """
    Prior: x = 0 w/ prob α and =+1 or -1 w/ prob (1-α)/2,
           w is N(0, σ2), s = x+w.

    Apply mmse marginal denoiser to each column of s, i.e. entrywise to s.
    σ2 is scalar noise variance.
    Cannot be a covariance matrix other than a scaled identity.

    Compared to old version, the benefit of this version is its
    agnosticity to d, so η can be initialised once and apply to both
    d>1 (AMP) and d=1 (SE).
    """
    σ2 = σ2.squeeze() if jnp.ndim(σ2) > 0 else σ2 # ensure σ2 is scalar
    codebook = create_codebook(d=1)
    s_T = s.reshape(-1, 1)
    η_T, x_MAP_T = η_mmse_xMAP_all_s(s_T, σ2, α, codebook, calcMAP_estimate)
    x_MAP = x_MAP_T.reshape(1, -1) if calcMAP_estimate else jnp.nan
    return η_T.reshape(1, -1), x_MAP # return row vectors
η_marginal_mmse_xMAP = jax.jit(η_marginal_mmse_xMAP,
                            static_argnames=['α', 'calcMAP_estimate'])

def η_marginal_mmse_xMAP_all_s(S, σ2, α, calcMAP_estimate=False):
    num_rows, d = S.shape
    η_all, X_MAP = jax.vmap(η_marginal_mmse_xMAP, in_axes=(0, None, None, None),
                    out_axes=0)(S, σ2, α, calcMAP_estimate)
    X_MAP = X_MAP.reshape(num_rows, d) if calcMAP_estimate else None
    return η_all.reshape(num_rows, d), X_MAP
η_marginal_mmse_xMAP_all_s = jax.jit(η_marginal_mmse_xMAP_all_s,
                                static_argnames=['α', 'calcMAP_estimate'])


def η_marginal_mmse_xMAP_all_s_thesis(S, Σ, α, calcMAP_estimate=False):
    """
    Realised that η_marginal_mmse_xMAP_all_s is suboptimal because it
    denoised the different columns of S using the same noise variance σ2.
    """
    num_rows, d = S.shape
    # η_marginal_mmse_xMAP takes in each column of S (Lx1) at a time
    # with corresponding σ2 in the diagonal of Σ, and returns a 1xL row vector.
    η_all, X_MAP = jax.vmap(η_marginal_mmse_xMAP, in_axes=(1, 0, None, None),
                    out_axes=0)(S, jnp.diag(Σ), α, calcMAP_estimate)
    # Stack 1xL row vectors into a dxL matrix.
    X_MAP = X_MAP.reshape(d, num_rows).T if calcMAP_estimate else None
    return η_all.reshape(d, num_rows).T, X_MAP
η_marginal_mmse_xMAP_all_s_thesis = jax.jit(η_marginal_mmse_xMAP_all_s_thesis,
                                static_argnames=['α', 'calcMAP_estimate'])

def sum_dη_marginal_mmse(S, σ2, α):
    """
    dη: Lxdxd tensor; each dxd matrix is the Jacobian matrix
        corresponding to each row of S. Since we assume
        independence across columns of S, the Jacobian matrices
        are diagonal.
    """
    codebook = create_codebook(d=1)
    def dη_i(S_i):
        # Differentiate the ith column of S.
        # Each row gives an 1x1 Jacobian matrix.
        # Sum over num_rows 1x1 matrices:
        return jnp.squeeze(sum_dη_mmse(S_i.reshape(-1, 1), σ2, α, codebook))
    diag_entries = jax.vmap(dη_i, in_axes=1, out_axes=0)(S)
    return jnp.diag(diag_entries)
sum_dη_marginal_mmse = jax.jit(sum_dη_marginal_mmse, static_argnames=['α'])



def sum_dη_marginal_mmse_thesis(S, Σ, α):
    """
    dη: Lxdxd tensor; each dxd matrix is the Jacobian matrix
        corresponding to each row of S. Since we assume
        independence across columns of S, the Jacobian matrices
        are diagonal.
    """
    codebook = create_codebook(d=1)
    def dη_i(S_i, σ2):
        # Differentiate the ith column of S.
        # Each row gives an 1x1 Jacobian matrix.
        # Sum over num_rows 1x1 matrices:
        return jnp.squeeze(sum_dη_mmse(S_i.reshape(-1, 1), σ2, α, codebook))
    diag_entries = jax.vmap(dη_i, in_axes=(1,0), out_axes=0)(S, jnp.diag(Σ))
    return jnp.diag(diag_entries)
sum_dη_marginal_mmse_thesis = jax.jit(sum_dη_marginal_mmse_thesis, static_argnames=['α'])

# def sum_dη_marginal_mmse_thesis_analytical(η, Σ, α):
#     """
#     While sum_dη_marginal_mmse_thesis uses AD,
#     this uses the analytical formula.
#     S and η are Lxd matrices, Σ is a dxd matrix
#     Assuming E=1 and α=0
#     """
#     assert α == 0, "This function is only for α=0."
#     def dη_i(η_i, σ2):
#         # η_i is the ith column of η and σ2=Σ_ii
#         # Assume E=1
#         return np.sum((1 - η_i**2) / σ2) # sum over length-L vector yielding a scalar
#         # same as etaJac_diag_entries, but now in jax.jit instead of numba.jit
#     diag_entries = jax.vmap(dη_i, in_axes=(1,0), out_axes=0)(η, jnp.diag(Σ))
#     return jnp.diag(diag_entries)
# sum_dη_marginal_mmse_thesis_analytical = \
#     jax.jit(sum_dη_marginal_mmse_thesis_analytical)

def sum_dη_marginal_mmse_thesis_analytical(S, Σ, α):
    diag_entries = jax.vmap(sum_dη_marginal_d1, in_axes=(1,0,None),
                            out_axes=0)(S, jnp.diag(Σ), α)
    return jnp.diag(diag_entries)
sum_dη_marginal_mmse_thesis_analytical = \
    jax.jit(sum_dη_marginal_mmse_thesis_analytical, static_argnames=['α'])

def subtract_max_val(data_arr):
    """
    Same as subtract_max_val in denoiser.py, but now in jax.
    data_arr: n1-by-1 (1D) or n1-by-L (2D) containing
              floating point numbers
    max_arr: 1-by-1 or 1-by-L;
             ith entry = max of ith coln
             i.e. the max +ve val if the ith coln has +ve vals
             or the least -ve val if the coln has -ve vals only
    data_minus_max_arr: data_arr minus max_arr column-wise

    This function is used to subtract the max exponent from
    a series of exponents for numerical stability.
    TODO: potential future solution: numba.jit each column and
    numba.vectorize across columns.
    """
    # compatible with both 1D, 2D data_arr:
    max_arr = jnp.max(data_arr, axis=0) # axis argument is not supported by numba
    # max_arr[max_arr < 0] = 0
    data_minus_max_arr = data_arr - max_arr
    assert data_minus_max_arr.shape == data_arr.shape
    return max_arr, data_minus_max_arr

def sum_dη_marginal_d1(s, σ2, α) -> np.array:
    """
    Adapted from _etaJac_mmse_d1 in Denoiser class.
    α is %silent users.
    The derivative/ Jacobian deta(s)/ds: R1-->R1 applies
    entrywise to its inputs.
    sigma2 is the scalar noise variance.

    when s is Lx1, etaJac is Lx1x1 with ith entry=d(eta(s_i))/d(s_i).
    """
    s = s.reshape(-1, 1)
    L = s.shape[0]
    # if σ2 == 0:
    #     dη = jnp.ones((L, 1, 1))  # eta(S) = E[X | X+0 = S] = S
    if α == 1:  # x is all-zero
        dη = jnp.zeros((L, 1, 1))
    else:
        # sigma2>0 and 0<=alpha<1
        # Divide through max exponent for numerical stability:
        p_nz = (1 - α) / 2
        σ2_inv_term = 1 / (2 * σ2)
        oppo_pwr_arr = s / σ2 * jnp.array([1, -1])  # Lx2
        # assert oppo_pwr_arr.shape == (L, 2)
        # numerator
        if α == 0:
            num_max_pwr = 0
            num_factor = 1 / σ2
        else:
            num_pwr_arr = jnp.concatenate(
                ((oppo_pwr_arr + σ2_inv_term).T, jnp.zeros((1, L)))
            )
            # assert num_pwr_arr.shape == (3, L)
            num_max_pwr, num_pwr_arr_minus_max = subtract_max_val(num_pwr_arr)
            num_sum_oppo_expo = jnp.exp(num_pwr_arr_minus_max[0]) + jnp.exp(
                num_pwr_arr_minus_max[1]
            )
            tmp = α * num_sum_oppo_expo + 2 * (1 - α) * jnp.exp(
                num_pwr_arr_minus_max[2]
            )
            num_factor = p_nz / σ2 * tmp
            # assert num_factor.shape == (L,)
        # denominator
        if α == 0:
            den_pwr_arr = oppo_pwr_arr.T
            # assert den_pwr_arr.shape == (2, L)
        else:
            den_pwr_arr = jnp.concatenate(
                (σ2_inv_term * jnp.ones((1, L)), oppo_pwr_arr.T)
            )
            # assert den_pwr_arr.shape == (3, L)
        den_max_pwr, den_pwr_arr_minus_max = subtract_max_val(den_pwr_arr)

        den_sum_oppo_expo = jnp.exp(den_pwr_arr_minus_max[-2]) + jnp.exp(
            den_pwr_arr_minus_max[-1]
        )
        if α == 0:
            den_factor = 1 / 4 * den_sum_oppo_expo**2
        else:
            den_factor = (
                α * jnp.exp(den_pwr_arr_minus_max[0]) + p_nz * den_sum_oppo_expo
            ) ** 2
        assert den_factor.shape == (L,)
        dη = jnp.exp(num_max_pwr - den_max_pwr * 2) * num_factor / den_factor
        # assert np.logical_not(np.any(np.isnan(dη)))  # no nans
    # assert dη.shape == (L, 1, 1)
    return jnp.sum(dη) # sum over L rows to get a scalar

def thres_RA(α, σ2):
    """σ2 is avg noise variance per row of effective observation."""
    E = 1
    θ = jnp.sqrt(E)/2 - jnp.log(α/(1-α)) * σ2/jnp.sqrt(E)
    return θ

def η_thres_xMAP(s, Σ, α, norm_s, calcMAP_estimate=False):
    """
    The suboptimal thresholding denoiser for random access
    (defined in thesis, May 2024), considering the binary-CDMA scheme
    without outer code.

    norm_s is l2-norm of s.
    If the sqrt(norm_s^2/k)<θ, estimate s as all-zero.
    Otherwise, apply the entrywise mmse denoiser (assuming α=1 for that row).
    """
    s = s.reshape(1,-1) # row vector
    k = s.shape[1]
    if α == 0: # all active
        θ = -jnp.inf # everything will be above this threshold
    elif α == 1: # all silent
        θ = jnp.inf # everything will be below this threshold
    else:
        σ2 = jnp.mean(jnp.diag(Σ)) # avg noise variance
        θ = thres_RA(α, σ2)
    if jnp.sqrt(norm_s**2/k) < θ:
        # return η(s) and xMAP estimate:
        return jnp.zeros((1,k), dtype=np.float64), \
            jnp.zeros((1,k), dtype=np.float64) if calcMAP_estimate else jnp.nan
    else:
        # applies marginal-MMSE denoiser entrywise with α_marg = 0:
        α_marg = 0
        η, x_MAP = jax.vmap(η_marginal_mmse_xMAP, in_axes=(1, 0, None, None),
                    out_axes=0)(s, jnp.diag(Σ), α_marg, calcMAP_estimate)
        return η.reshape(1,k), x_MAP.reshape(1,k) if calcMAP_estimate else jnp.nan
# η_thres_xMAP = jax.jit(η_thres_xMAP,
#     static_argnames=['α', 'norm_s', 'calcMAP_estimate'])
# I wasn't able to avoid tracing s above (as its norm is compared to threshold)
# so for now I'll avoid using η_thres_xMAP and use η_thres_xMAP_all_s instead.
# η_thres_xMAP is only used for testing η_thres_xMAP_all_s.


def η_thres(s, Σ, α, norm_s):
    return η_thres_xMAP(s, Σ, α, norm_s, calcMAP_estimate=False)[0]

def η_thres_sum_dη_XMAP_all_s(S, Σ, α, calcMAP_estimate=False):
    """
    The suboptimal thresholding denoiser for random access
    (defined in thesis, May 2024), considering the binary-CDMA scheme
    without outer code.

    For each row s of S:
    If the sqrt(norm_s^2/k)<θ, estimate s as all-zero.
    Otherwise, apply the entrywise mmse denoiser (assuming α=1 for that row).
    """
    num_rows, d = S.shape
    Σ = Σ.reshape(d, d)
    if α == 1: # all silent
        η_all = jnp.zeros((num_rows, d), dtype=jnp.float64)
        X_MAP = jnp.zeros((num_rows, d), dtype=jnp.float64) if calcMAP_estimate else jnp.nan
        sum_dη = jnp.zeros((d, d), dtype=jnp.float64)
        return η_all, sum_dη, X_MAP
    else:
        # different columns in S need to be denoised using different noise variances
        α_marg = 0 # assume all active, because thresholding step will simply
        # zero out silent users
        η_marg, X_marg_MAP = jax.vmap(η_marginal_mmse_xMAP, in_axes=(1, 0, None, None),
                out_axes=0)(S, jnp.diag(Σ), α_marg, calcMAP_estimate)
        # η applies to each column and returns a row vector. stack these as rows into a dxL matrix.
        η_marg = η_marg.transpose().reshape(num_rows, d)
        X_marg_MAP = X_marg_MAP.transpose().reshape(num_rows, d) if calcMAP_estimate \
            else jnp.nan
        E = 1
        diag_dη = (E - η_marg.reshape(num_rows, d)**2)/(jnp.diag(Σ).reshape(1,d)) # num_rows x d
        assert jnp.shape(diag_dη) == (num_rows, d)
        # Above: each row is the diagonal of the Jacobian matrix for that row
        # Now apply thresholding:
        if α > 0: # not all active
            norm_per_row = jnp.linalg.norm(S, axis=1)
            assert jnp.shape(norm_per_row) == (num_rows,)
            σ2 = jnp.mean(jnp.diag(Σ)) # avg noise variance
            θ = thres_RA(α, σ2)
            # zeroton:
            is_silent = jnp.sqrt(norm_per_row**2/d) < θ # jnp.minimum(2*jnp.sqrt(σ2), jnp.sqrt(E))
            # is_silent rows fill in zeros, others fill in η_marg
            η_all = jnp.where(is_silent.reshape(-1, 1), jnp.zeros_like(S), η_marg)
            X_MAP = jnp.where(is_silent.reshape(-1, 1), jnp.zeros_like(S), X_marg_MAP) \
                if calcMAP_estimate else jnp.nan
            diag_dη = jnp.where(is_silent.reshape(-1, 1), jnp.zeros_like(S), diag_dη)
            sum_dη = jnp.diag(jnp.sum(diag_dη, axis=0))
        else:
            η_all = η_marg
            X_MAP = X_marg_MAP
            sum_dη = jnp.diag(jnp.sum(diag_dη, axis=0))
        return η_all, sum_dη, X_MAP
# η_thres_sum_dη_XMAP_all_s = jax.jit(η_thres_sum_dη_XMAP_all_s,
#     static_argnames=['α', 'calcMAP_estimate'])

# Derivative of η_thres with respect to s.
dη_thres = jax.jacfwd(η_thres, argnums=0)


def η_dη_bp_xMAP(s, σ2_arr, ldpc_code: code, bp_max_itcount: int,
              calc_sum_dη: bool=False, calcMAP_estimate: bool=False):
    """
    Applies BP to the effective observation row vector s, returns bit-wise
    MAP estimate of x (instead of codeword-wise MAP).

    s: effective observation, 1xd
    σ2_arr: noise variances, 1xd

    This currently only supports α=0 (all users active).
    Inputs have to be numpy arrays, not jax arrays.
    """
    # LLR based on channel evidence:
    # ln(P(ch|x=0)/ P(ch|x=1)) = ln(P(x=0|ch)/ P(x=1|ch))
    # where zero => channel input 1 and one => channel input -1
    # llr = 2.0 / σ2_arr * s
    s = s.flatten()
    llr = ch2llr(s, σ2_arr)
    # llr = np.array(2.0/σ2_arr*s)
    # llr = np.array(2.0/σ2_arr*s) # convert to numpy array for ldpc.decode
    # (app,_) = ldpc_code.decode(llr, bp_max_itcount)
    # by default uses sumprod2:
    # decode cannot be jitted because of certain ctypes properties are not supported by numba:
    (app,_) = decode(ldpc_code.vdeg, ldpc_code.cdeg, ldpc_code.intrlv, \
        ldpc_code.Nv, ldpc_code.Nc, ldpc_code.Nmsg, llr, bp_max_itcount)
    η = app2eta_bp(app)
    # η = 1 - 2.0 / (1 + np.exp(app))
    # assert np.all(η >= -1) and np.all(η <= 1)

    dη_diag_entries = etaJac_diag_entries(η, σ2_arr) \
        if calc_sum_dη else None # length-d vector
    # dη_diag_entries = (1 - η ** 2) / σ2_arr if calc_sum_dη else np.nan # length-d vector
    x_MAP = bpsk(app < 0.0) if calcMAP_estimate else None
    return η, dη_diag_entries, x_MAP


def η_sum_dη_bp_xMAP_all_s(S, Σ, ldpc_code: code, bp_max_itcount: int,
              calc_sum_dη: bool=False, calcMAP_estimate: bool=False):
    num_rows, d = S.shape
    """
    Inputs have to be numpy arrays, not jax arrays.
    Returns η, sum_dη, X_MAP
    """
    σ2_arr = np.diag(Σ).flatten()
    # NOTE: vmap over rows of S means S is traced, which is not supported
    # by the c code underlying ldpc.decode.
    # η_all_s, dη_diag_entries_all_s, X_MAP = \
    #     jax.vmap(η_dη_bp_xMAP, in_axes=(0, None, None, None, None, None),
    #     out_axes=0)(S, σ2_arr, ldpc_code, bp_max_itcount, calc_sum_dη, calcMAP_estimate)
    # NOTE: instead we use simple for loop over num_rows:
    η_all_s = np.zeros((num_rows, d), dtype=np.float64)
    dη_diag_entries_all_s = np.zeros((num_rows, d), dtype=np.float64) \
        if calc_sum_dη else None
    X_MAP = np.zeros((num_rows, d), dtype=np.float64) \
        if calcMAP_estimate else None

    # print(f"η_sum_dη_bp_xMAP_all_s processing {num_rows} num_rows")
    for i in range(num_rows):
        # start_time = time.time()
        η, dη_diag_entries, x_MAP = \
            η_dη_bp_xMAP(S[i, :], σ2_arr, ldpc_code, bp_max_itcount, \
                calc_sum_dη, calcMAP_estimate)
        η_all_s[i, :] = η
        if calc_sum_dη:
            dη_diag_entries_all_s[i, :] = dη_diag_entries
        if calcMAP_estimate:
            X_MAP[i, :] = x_MAP
        # print(f"η_sum_dη_bp_xMAP_all_s finished processing row {i}/{num_rows}" + \
        #       f" in {time.time()-start_time} seconds.")
    sum_dη = np.diag(
        np.sum(dη_diag_entries_all_s.reshape(num_rows, d), axis=0).reshape(d)) \
        if calc_sum_dη else None
    X_MAP = X_MAP.reshape(num_rows, d) if calcMAP_estimate else None
    return η_all_s.reshape(num_rows, d), sum_dη, X_MAP
