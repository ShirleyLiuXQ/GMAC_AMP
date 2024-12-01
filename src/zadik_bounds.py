import numpy as np
from scipy.stats import norm, entropy

# Kuan's code to reproduce achievability and converse bounds in Thm3, Zadik19.


# Converse bound:
def min_energy_converse(M, eps):
    """
    From Y. Polyanskiy, H. V. Poor, and S. Verdu ́,
    “Minimum energy to send k bits with and without feedback,” (2011)
    Thm1 instead of Thm2.

    In our case k=log2(M) and there is no feedback.
    eps is the probability of error tolerated.

    SL: assumes only one active user, all others are silent.
    """
    EbN0 = (norm.isf(1 / M) - norm.isf(1 - eps)) ** 2 / (2 * np.log2(M * 1.0))
    return EbN0 # scalar


def gmac_converse(M, PUPE, mu_arr):
    """
    PUPE: per-user probability of error
    Converse bound based on Fanos inequality.
    """
    # Second expression for 'exp' gives tighter bound (because
    # H(X|XHat, E=1) \le log2(M-1)).
    # Won't make much difference at small PUPE and large M.

    rho = mu_arr * np.log2(M * 1.0)  # the spectral efficientcy i.e.
    # total num user bits per transmitted bit
    # exp  = 2*(1-PUPE)*rho - mu*entropy([PUPE,1-PUPE], base=2)
    exp = (2 * mu_arr * (
            np.log2(M * 1.0)
            - PUPE * np.log2(M * 1.0 - 1)
            - entropy([PUPE, 1 - PUPE], base=2)
        )
    )
    EbN0 = (2**exp - 1) / (2 * rho)
    return EbN0 # array


def gmac_converse_comb(M, PUPE, mu_arr):
    """
    Combine min_energy_converse and gmac_converse.
    """
    EbN0_a = min_energy_converse(M, PUPE) # scalar
    print(f"EbN0_a = {EbN0_a}")
    EbN0_b = gmac_converse(M, PUPE, mu_arr) # array
    print(f"EbN0_b = {EbN0_b}")
    # Pick the larger value between the two, because:
    # The larger one corresponds to a tighter converse, i.e. closer to the achievability bound.
    # EbN0 = EbN0_a if EbN0_b < EbN0_a else EbN0_b
    # EbN0_SL = np.max(EbN0_a, EbN0_b)
    # assert EbN0 == EbN0_SL
    EbN0 = np.maximum(EbN0_a, EbN0_b) # array
    print(f"EbN0 = {EbN0}")
    return EbN0


# Achievability bound:
def check_eq13(b, M, PUPE, mu):
    """
    Checking if equation (13) holds, for a fixed b>0 and for all
    theta in [PUPE, 1]. We check a discrete set of points in [PUPE, 1]
    as we cannot check all points in the continuous range.

    b, M, PUPE, mu are all scalars.
    theta below is the set of points in [PUPE, 1].
    """

    # See equation (7)
    def gamma_fn(s):
        return np.exp(-0.5 * (norm.isf(s)) ** 2) / np.sqrt(2 * np.pi)

    # See equation (12)
    def psi_func(b, theta, mu):
        return np.maximum(
            0, np.sqrt(1 + b**2 * theta * mu) - b * mu * gamma_fn(theta)
        )

    theta = np.linspace(PUPE, 1, 300)  # See Thm 3. Can change # of steps.
    r = b**2 * theta * mu  # See remark 5
    psi = psi_func(b, theta, mu)  # See remark 5
    Lambda = np.maximum(
        0, (r + np.sqrt(r**2 + 4 * psi**2) - 2) / (4 * r)
    )  # See remark 5
    # Above: RV: Added the max because (13) requires Lambda>=0

    LHS = theta * mu * np.log(M * 1.0) + mu * entropy([theta, 1 - theta])  # LHS of (13)
    # Above: RV: previously entropy([PUPE,1-PUPE])
    RHSa = 0.5 * np.log(1 + 2 * b**2 * theta * mu * Lambda)
    RHSb = Lambda * psi**2 / (1 + 2 * b**2 * theta * mu * Lambda)
    RHS = RHSa + RHSb - Lambda  # RHS of (13)

    # SL: when negative Lambda is clipped off, the LHS is always greater than the RHS=0.
    # So we should only check whether LHS < RHS when Lambda >= 0. (BOTH VERSIONS ARE FINE)
    # return np.all(LHS[Lambda > 0] < RHS[Lambda > 0])
    return np.all(LHS < RHS)


def gmac_achieve_zadik19(M, PUPE, mu_list, EbN0_list_dB):
    """
    For a fixed M and mu, find the minimum Eb/N0 such that a target
    per-user probability of error (PUPE) is achievable.

    M           : messages per user (log2(M) bits per user is the user payload)
    PUPE        : target per-user probability of error
    mu_list     : list of mu values
    EbN0_list_dB: list of Eb/N0 in dBs in which to find the minimum Eb/N0
    """

    EbN0_list = 10 ** (EbN0_list_dB / 10)
    b_list = np.sqrt(2 * EbN0_list * np.log2(M * 1.0))

    EbN0_ach = np.zeros(mu_list.size)
    for i, mu in enumerate(mu_list):
        for b in b_list:
            if check_eq13(b, M, PUPE, mu):
                EbN0_ach[i] = b**2 / (2 * np.log2(M * 1.0))
                break  # increase EbN0 and stop when we eqn (13) is first satisfied

    return EbN0_ach
