import numpy as np
from numba import jit, int64, float64
from numba.experimental import jitclass
from typing import Tuple
from scipy.fftpack import fft

from ldpc_jossy.py.ldpc import code
from ldpc_jossy.py.ldpc_awgn import bpsk  # dct, idct


# Functions to construct A, X, W and Y for use in the AMP algorithm
# or SE.
@jit(nopython=True)
def design_mat_A(n: int, L: int, type: str='gaussian') -> np.array:
    """
    Create an nxL random design matrix A.
    A_choice:
        gaussian: With iid Gaussian entries N(0,1/n)
        bipolar: With iid entries in {+1/sqrt(n), -1/sqrt(n)} with
           uniform probability
    """
    # assert n <= L  # fat design matrix
    if type.lower() == 'gaussian':
        A = np.random.randn(n, L) / np.sqrt(n)
    elif type.lower() == 'bipolar':
        A = np.random.rand(n, L)
        # A = (A < 0.5) / np.sqrt(n)
        # A[np.logical_not(A)] = -1 / np.sqrt(n)
        # bool indexing isnt supported by numba, work around:
        # logic = np.logical_not(A)
        # A.ravel()[logic.ravel()] = -1 / np.sqrt(n)
        A = ((-1) ** (A < 0.5)) / np.sqrt(n)
    elif type.lower() == 'fourier':
        raise ValueError("Fourier matrix isnt supported by numba" +
                         " unless we force A to be complex type")
    else:
        raise ValueError("Type must be 'gaussian', 'bipolar' or 'fourier'")
    # Check the norm of the columns of A are 1
    # assert np.allclose(np.linalg.norm(A, axis=0), 1, atol=0.05)
    assert A.shape == (n, L)
    return A

@jit(nopython=True)
def design_mat_A_archive(n: int, L: int, type: str='gaussian') -> np.array:
    """
    Create an nxL random design matrix A.
    A_choice:
        gaussian: With iid Gaussian entries N(0,1/n)
        bipolar: With iid entries in {+1/sqrt(n), -1/sqrt(n)} with
           uniform probability
        fourier: Random partial Fourier matrix
    NOTE: we make return type complex to accommodate all three cases,
    otherwise numba throws a type unification error.
    """
    # assert n <= L  # fat design matrix
    if type.lower() == 'gaussian':
        A = (np.random.randn(n, L) / np.sqrt(n)).astype(np.complex128)
    elif type.lower() == 'bipolar':
        A = np.random.rand(n, L)
        # A = (A < 0.5) / np.sqrt(n)
        # A[np.logical_not(A)] = -1 / np.sqrt(n)
        # bool indexing isnt supported by numba, work around:
        # logic = np.logical_not(A)
        # A.ravel()[logic.ravel()] = -1 / np.sqrt(n)
        A = (((-1) ** (A < 0.5)) / np.sqrt(n)).astype(np.complex128)
    elif type.lower() == 'fourier':
        # NEED TO CHECK THIS CONSTRUCTION:
        if n % 2 == 0:  # even
            # A = fft(np.eye(L))  # LxL FFT matrix
            # scipy.fftpack.fft isnt supported by numba
            A = np.fft.fft(np.eye(L)) # complex return type not supported by numba,
            # install and use rocket-fft instead
            # Sample from row 1 to row (L-1) (excluding row 0)
            # without replacement:
            selected_rows_idx = np.random.choice(np.arange(1, L), size=n, replace=False)
            # May also need to exclude row L/2 from the sampling?
            # Normalize so that L2 norm of each column = 1:
            A = A[selected_rows_idx, :] / np.sqrt(n)
        else:
            # With n = odd, cannot normaize columns of A to have unit norm
            raise ValueError(
                "n must be an even number for random partial"
                "Fourier design matrix to be properly normalised"
            )
    else:
        raise ValueError("Type must be 'gaussian', 'bipolar' or 'fourier'")
    # Check the norm of the columns of A are 1
    # assert np.allclose(np.linalg.norm(A, axis=0), 1, atol=0.05)
    assert A.shape == (n, L)
    return A

@jit(nopython=True)
def design_mat_A_sc(n: int, L: int, W: np.array=np.array([[1]]), \
                    type: str='gaussian') -> np.array:
    """Create a SC design matrix. When W=1, the SC design matrix
    reduces to an iid design."""
    assert type.lower() == 'gaussian', "Only Gaussian design matrix is supported right now"
    R, C = W.shape
    M = int(n/R)
    N = int(L/C)
    assert M == int(M) and N == int(N), "n/R and L/C must be integers"
    A = np.zeros((n, L))
    for r in range(R):
        for c in range(C):
            if W[r,c] > 0:
                A[r*M: (r+1)*M, c*N: (c+1)*N] = \
                    np.random.normal(0, np.sqrt((1/M)*W[r,c]), (M, N))
                    # np.random.randn(M, N) * np.sqrt(R * W[r,c]/n)
            else:
                assert W[r,c] == 0 # leave the corresponding block in A as zeros
    return A

# @jit(nopython=True)
def noise_mat_W(n: int, d: int, Σ_or_σ2):
    """
    Create an nxd noise matrix W. Each row of W is drawn iid
    from a dim-d Gaussian distribution with zero mean and
    covariance matrix Σ.

    When Σ is diagonal, or when it is a scalar σ2,
    the entries in each row of W are independent.

    This function will be used by initialise_model when
    Sigma is +ve definite.
    This function is also needed for some Monte Carlo estimation.
    """
    Σ_or_σ2 = np.array(Σ_or_σ2) # avoid jax arrays
    # assert np.all(np.linalg.eigvals(Sigma) > 0)  # +ve definite
    if Σ_or_σ2.size == 1: # i.e. Σ_or_σ2 is a scalar
        Σ_or_σ2 = Σ_or_σ2.item() # convert to scalar
        return np.random.normal(0, np.sqrt(Σ_or_σ2), (n, d))
    else:
        assert Σ_or_σ2.shape[0] == Σ_or_σ2.shape[1] and \
            np.max(Σ_or_σ2 - Σ_or_σ2.T) < 1e-10 # np.allclose(Sigma, Sigma.T) not supported by numba
        return _noise_W_from_Σ(n, d, Σ_or_σ2)
@jit(nopython=True)
def _noise_W_from_Σ(n, d, Σ):
    Sigma_chol = np.linalg.cholesky(Σ) # + 1e-10 * np.eye(d))
    # assert np.allclose(Sigma_chol @ Sigma_chol.T.conj(), Sigma)
    return (Sigma_chol @ np.random.randn(d, n)).T

@jit(nopython=True)
def add_iid_noise(Y_nl, sigma2=0):
    """
    Return ny_Y = nl_Y + W, where the noise matrix W is drawn iid from N(0, sigma2)

    This function seems simple but is highly error-prone:
    e.g. sigma2 must undergo sqrt operation before multiplying with standard Gaussian noise;
    Celeste accidentally used rand instead of randn which resulted in a much easier noise distribution.
    """
    assert sigma2 >= 0
    n, d = Y_nl.shape
    return Y_nl + np.random.normal(0, np.sqrt(sigma2), (n, d)) if sigma2 > 0 else Y_nl
    # Y_ny = Y_nl + np.sqrt(sigma2) * np.random.randn(n, d) if sigma2 > 0 else Y_nl
    # return Y_ny

@jit(nopython=True)
def EbN0db_to_sigma2(EbN0db):
    """
    For fixed P, convert Eb/N0 in dB to sigma2 for real signals.
    Eb: energy per user bit transmitted
    N0: power spectral density (PSD) of noise.
        When noise is Gaussian N0=2*sigma2.
    k : single user payload = d when user signal is +1/-1
    nTotal: num real channel uses = nr
    P : when the signal Xi is +1 or -1 for an active user,
        and A is iid N(0,1/n), P=1/n
    Energy used by each user to transmit k bits is nTotal*P.

    Eb/N0=nTotal*P/k/sigma2 when the signals are complex.
    Eb/N0=nTotal*P/k/(2*sigma2) when the signals are real.

    nTotal = n*k
    P = 1/n gives correctly scaled design matrix A.
    Thus, for real channels, Eb/N0 simply = (n*k)*P/k/(2*sigma2) = 1/(2*sigma2).

    TODO: I believe Eb/N0 should remain unchanged regardless of whether
    the signal is encoded. "Per bit" here refers to a bit transmitted, not
    an information bit.
    """
    EbN0 = 10 ** (EbN0db / 10) # convert dB to linear scale
    sigma2 = 1 / (2 * EbN0)
    return sigma2

@jit(nopython=True)
def bpsk_codebook(len: int):
    """
    Returns an 2^len x len matrix where each row stores a distinct bpsk sequence
    (i.e. a sequence with -1,+1 uniformly random).
    """
    assert len > 0, "len must be a positive integer"
    # 2^len x 1 matrix storing numbers from 0 to (2^len)-1:
    a = np.arange(2**len, dtype=np.int8)[:, np.newaxis]
    # 1 x len matrix for the powers of 2 with exponents from 0 to (len-1):
    b = np.arange(start=len - 1, stop=-1, step=-1, dtype=np.int8)[np.newaxis, :]
    powers_of2 = 2**b
    # u & v = bitwise-and of binary u and binary v:
    # binary_mat = np.array((a & powers_of2) > 0, dtype=np.int64)
    binary_mat = (a & powers_of2) > 0
    codebook = (-1) ** binary_mat
    assert codebook.shape == (2**len, len)
    return codebook.astype(np.int8)

# @jit(nopython=True)
def hamming_code() -> Tuple[np.array, np.array]:
    """3F7 handout version, equivalence exists"""
    H = np.array([[1, 1, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 1, 0, 0, 1]], dtype=np.int8)
    G = np.array(
        [[1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 1]], dtype=np.int8)
    # sanity check: GH^T = all-zero matrix
    assert np.all(G @ H.T % 2 == 0)
    # numba doesnt allow integer matrix multiplication:
    # assert np.all(G.astype(float64) @ H.astype(float64).T % 2 == 0)
    return G, H

# @jit(nopython=True)
def create_codebook(d: int, code: str=None) -> np.array:
    """
    Returns a num_codewordsxd codebook matrix, where each row is a distinct codeword,
    i.e. all possible rows of X, consist of +1s, -1s or zeros.

    For the non-zero entries of X: +1s are bit zeros; -1s are bit ones.
    """
    if code is None: # without coding
        cb = bpsk_codebook(d)
    else:
        if code.lower() == 'hamming':
            k, n = 4, 7
            assert d == n, f"Hamming code only supports d={n}"
            msg_bits = bpsk_codebook(k) == -1
            G, H = hamming_code()
            # coded_bits = msg_bits @ G % 2
            coded_bits = (msg_bits.astype(np.int8) @ G.astype(np.int8) % 2).astype(np.int8)
        elif code.lower() == 'single-bit-parity-check':
            # Single-bit parity check code:
            msg_bits = bpsk_codebook(d - 1) == -1
            coded_bits = np.concatenate((msg_bits, np.sum(msg_bits, axis=1, keepdims=True) % 2), axis=1)
            # coded_bits = np.concatenate((msg_bits.astype(int64), \
            #             np.sum(msg_bits, axis=1)[:, np.newaxis] % 2), axis=1)
        else:
            raise ValueError(f"Unknown code: {code}")
        cb = (-1) ** coded_bits
    assert cb.shape[1] == d
    assert np.all(np.abs(cb) == 1), "Codebook should contain BPSK entries rather than 0/1s."
    return cb.astype(np.int8)

@jit(nopython=True)
def calc_signal_power(alpha: float, codebook: np.array) -> np.array:
    """
    Return the analytical solution for E[X0_bar X0_bar^T] where X0_bar is
    the len-d column vector distributed as the rows of X.
    """
    assert 0 <= alpha <= 1
    num_codewords, d = codebook.shape
    # signal_power = (1 - alpha) / num_codewords * (codebook.T @ codebook)
    # Matrix multiplication only works for floats or complex inputs in numba:
    signal_power = (1 - alpha) / num_codewords * \
        (codebook.astype(float64).T @ codebook.astype(float64))
    assert signal_power.shape == (d, d)
    return signal_power

# spec = [('L', int64), ('alpha', float64), ('num_allzero_rows', int64),
#         ('num_codewords', int64), ('d', int64),
#         ('cb', int64[:, :]), ('ldpc_code', code.class_type.instance_type)]
# @jitclass(spec)
class SignalMatrix:
    def __init__(self, L: int, alpha: float, codebook: np.array=None,
                ldpc_code: code=None, d_uncoded: int=None) -> None:
        """
        L: num rows of X.
        Draw these rows with alpha probability of being all-zero, and 1-alpha
        probability of being drawn from either codebook/ldpc_code.
        Special case: for the uncoded case with large d and marginal denoiser,
        codebook may simply be [1,0], in which case use d_uncoded to override the codebook
        shape.
        """
        self.L = L
        self.alpha = alpha # probability of all-zero rows,
        # alpha=1 means all rows are zeros.
        assert 0 <= alpha <= 1
        self.num_allzero_rows = np.random.binomial(L, alpha) if alpha > 0 else 0
        self.cb = codebook
        self.ldpc_code = ldpc_code
        if ldpc_code is not None:
            self.num_codewords = 2**ldpc_code.K
            self.d = ldpc_code.N
            self.d_uncoded = None
        elif codebook is not None:
            if d_uncoded is not None and d_uncoded != codebook.shape[1]: # uncoded
                assert codebook.shape[1] == 1, \
                    "d_uncoded is not None must be the uncoded case"
                self.num_codewords, self.d = None, d_uncoded
                self.d_uncoded = d_uncoded
            else:
                self.num_codewords, self.d = codebook.shape
                self.d_uncoded = None
        else:
            raise ValueError("Either codebook or ldpc_code must be provided.")

    def sample(self) -> np.array:
        """Sample a signal matrix X."""
        allzero_rows = np.zeros((self.num_allzero_rows, self.d), dtype=np.int8)
        # int8 forces allzero_rows to be integer which can cause floats
        # failing to overwrite them later on. One solution is to always
        # initialise estimates as floats in AMP. zeros_like(X0) with
        # ground truth X0 being integer will result in estimate Xt forced into integer.
        num_nz_rows = self.L - self.num_allzero_rows
        if self.ldpc_code is not None:
            nz_rows = sample_ldpc_codewords(num_nz_rows, self.d, self.ldpc_code)
        else:
            if self.d_uncoded is None: # coded case:
                nz_codeword_idx = np.random.randint(self.num_codewords, size=num_nz_rows)
                nz_rows = self.cb[nz_codeword_idx, :]
            else: # uncoded case: sample 0,1s randomly
                nz_rows = np.random.randint(0, 2, (num_nz_rows, self.d_uncoded))
                nz_rows = bpsk(nz_rows) # convert to BPSK
        X = np.concatenate((allzero_rows, nz_rows), axis=0)
        assert X.shape == (self.L, self.d)
        np.random.shuffle(X)
        return X.astype(np.float64)

@jit(nopython=True)
def sample_ldpc_codewords(num_nz_rows: int, d: int, ldpc_code: code):
    nz_rows = np.zeros((num_nz_rows, d))
    for i in range(num_nz_rows):
        u = np.random.randint(0, 2, ldpc_code.K)
        nz_rows[i, :] = ldpc_code.encode(u)
    nz_rows = bpsk(nz_rows) # convert to BPSK
    return nz_rows.astype(np.int8)


def create_W(lam, omega, rho=0):
    """
    Creates a random lambda, omega base matrix for spatial coupling.
    coupling width omega ≥ 1, coupling length lambda ≥ 2*omega-1, and rho \in [0, 1).
    """
    assert omega >= 1 and lam >= 2*omega-1 and 0 <= rho < 1
    R = lam + omega - 1
    C = lam

    W = rho/(lam - 1) * np.ones((R,C)) if lam>1 else np.zeros((R,C))
    for c in range(C):
        for r in range(c, c+omega): #(c+w) index NOT included
            W[r,c]=(1-rho)/omega

    assert np.all(W >= 0), "All entries of W should be non-negative"
    assert np.allclose(np.sum(W, axis=0), 1), "Sum of entries in each column should be 1"
    return W

def calc_eff_user_density(L: int, d: int, n: int, code: str=None):
    """
    The signal matrix X is Lxd.
    L: number of users, d: length of each user's codeword.
    The channel observation Y is nxd.

    Returns the effective user density, i.e. the number of
    users message bits transmitted per channel use.

    Allows one of L, d, n to be an array.
    """
    if code is None:
        num_msg_bits = d
    elif code.lower() == 'hamming':
        assert d == 7
        num_msg_bits = 4
    elif code.lower() == 'single-bit-parity-check':
        num_msg_bits = d - 1
    else:
        raise ValueError(f"Unknown code: {code}")

    return (L * num_msg_bits) / (n * d)


def scale_n_for_fixed_S(S: float, L: int, d: int=None, \
                code: str=None, ldpc_rate: float=None):
    """
    Returns the number of channel uses n, given
    Spectral density S = total # user bits/ total # channel uses
                       = Lk/(nd) = L/n * code rate.

    Allows one of L, d, n to be an array.
    """
    if code is not None and code.lower() == 'ldpc':
        assert ldpc_rate is not None
        n = L / S * ldpc_rate
    else:
        n = L * calc_k(d, code) / S / d
    return int(n) if np.ndim(n) == 0 else n.astype(int)

def calc_δ_for_fixed_S(S: float, d: int=None, \
                code: str=None, ldpc_rate: float=None):
    """S = (L/n) * (k/d) = (1/δ) * code rate"""
    if code is not None and code.lower() == 'ldpc':
        assert ldpc_rate is not None
        δ = ldpc_rate / S
    else:
        δ = calc_k(d, code) / S / d
    return δ

def scale_σ2_for_fixed_EbN0(σ2_uncoded, d: int=None, \
                code: str=None, ldpc_rate: float=None):
    """Scale channel noise to match the desired Eb/N0."""
    if code is not None and code.lower() == 'ldpc':
        assert ldpc_rate is not None
        σ2 = σ2_uncoded / ldpc_rate
    else:
        num_msg_bits = calc_k(d, code)
        σ2 = σ2_uncoded / (num_msg_bits/d)
    return σ2

def calc_k(d: int, code = None):
    if code is None:
        k = d
    elif code.lower() == 'hamming':
        assert d == 7
        k = 4
    elif code.lower() == 'single-bit-parity-check':
        k = d - 1
    else:
        raise ValueError(f"Unknown code: {code}")
    return k

def calc_δ_in_from_δ(δ, lam, omega):
    return δ *(lam/(lam+omega-1))
