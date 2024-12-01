"""
Created on Tue Feb 21 17:36:45 2023

@author: pp423
"""
import numpy as np
from scipy.fftpack import dct, idct


def sub_dct_iid(n, L, seed=0, order0=None, order1=None):
    """
    Returns functions to compute the sub-sampled Discrete Cosine Transform,
    i.e., matrix-vector multiply with subsampled rows from the DCT matrix.
    This is a direct modification of Adam Greig's pyfht source code which can
    be found at https://github.com/adamgreig/pyfht/blob/master/pyfht.py

    [Inputs]
    n: number of rows in the sub-sampled DCT matrix A
    L: number of columns
    n < L
    Most efficient (but not required) for max(n+1,L+1) to be a power of 2.
    seed:   determines choice of random matrix
    order0: optional n-long array of row indices in [1, max(n+1,L+1)]
            which indicates the rows to be subsampled;
            generated by seed if unspecified.
    order1: optional L-long array of column indices in [1, max(n+1,L+1)]
            which indicates the columns to be subsampled;
            generated by seed if unspecified.

    [Outputs]
    Ax(x):  function handle. Computes A.x (A multiplied by x),
            with x of length L and A.x of length n.
    Ay(y):  function handle. Computes A*.y (A transposed multiplied by y),
            with y of length n and A*.y of length L.

    [Notes]
    There is a scaling of 1/sqrt(n) in the outputs of Ax() and Ay().
    """

    assert type(n) == int and n > 0
    assert type(L) == int and L > 0
    w = 2 ** int(np.ceil(np.log2(max(n + 1, L + 1))))
    # nearest power of 2 thats greater than max(m+1,n+1)

    if order0 is not None and order1 is not None:
        assert order0.shape == (n,)
        assert order1.shape == (L,)
    else:
        rng = np.random.RandomState(seed)
        # np.random.seed(seed) changes the global numpy state while
        # rng = np.random.RandomState(seed) only creates a local
        # instance of random number generator.
        idxs0 = np.arange(1, w, dtype=np.uint32)
        idxs1 = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs0)
        rng.shuffle(idxs1)
        order0 = idxs0[:n]  # items from index=0 through to (n-1)
        order1 = idxs1[:L]

    def Ax(x):
        # Function that computes the subsampled A matrix multiplied by x.
        # We use the full DCT matrix to multiply with x_ext, whose entries
        # corresponding to the subsampled columns are the entries of x, and
        # while the remaining are zeros.
        assert x.size == L, "x must be n long"
        x_ext = np.zeros(w)  # x extended
        # print("x", x, "Size ", x.size)
        x_ext[order1] = x.reshape(L)
        # print("Shuffled:", x_ext, "Size ", x.size)
        # print(order1)
        y = np.sqrt(w) * dct(x_ext, norm="ortho")
        # print(y.size)
        return y[order0] / np.sqrt(n)  # this sqrt(n) is to account for the
        # factor of 1/n in the variance of our iid Gaussian measurement matrix

    def Ay(y):
        # Function that computes A transposed multiplied by x.
        assert y.size == n, "input must be n long"
        y_ext = np.zeros(w)
        y_ext[order0] = y
        x_ext = np.sqrt(w) * idct(y_ext, norm="ortho")
        return x_ext[order1] / np.sqrt(n)

    return Ax, Ay


def sub_dct_iid_mat(n: int, L: int, d: int, seed=0, order0=None, order1=None):
    """
    Returns functions to compute the sub-sampled Discrete Cosine Transform,
    i.e., matrix-matrix multiply with subsampled rows from the DCT matrix.
    This does not seem to provide significant reduction in runtime (less
    significant as n,L become larger) wrt using a for loop over each of
    the d columns of the signal matrix (for d small).

    [Inputs]
    n: number of rows of design matrix
    L: number of columns of design matrix
    d: number of columns of signal X
    Most efficient (but not required) for max(n+1,L+1) to be a power of 2.
    seed:   determines choice of random matrix
    order0: optional n-long array of row indices in [1, max(n+1,L+1)] to
            implement subsampling of rows; generated by seed if not specified.
    order1: optional L-long array of row indices in [1, max(n+1,L+1)] to
            implement subsampling of columns; generated by seed if not specified.

    [Outputs]
    AX(X):    computes A.X (of dim. m*k), with x having dim. L x d
    AY(Y):    computes A*.Y (of dim. n*k), with y having dim. n x d

    [Notes]
    There is a scaling of 1/sqrt(n) in the outputs of Ax() and Ay().
    """

    assert (type(n) == int or type(n) == np.int64) and n > 0
    assert (type(L) == int or type(L) == np.int64) and L > 0
    w = 2 ** int(np.ceil(np.log2(max(n + 1, L + 1))))

    if order0 is not None and order1 is not None:
        assert order0.shape == (n,)
        assert order1.shape == (L,)
    else:
        rng = np.random.RandomState(seed)
        idxs0 = np.arange(1, w, dtype=np.uint32)
        idxs1 = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs0)
        rng.shuffle(idxs1)
        order0 = idxs0[:n]
        order1 = idxs1[:L]

    def AX(X):
        assert X.shape[0] == L, "X must have L rows"
        X_ext = np.zeros((w, d))
        # The reshape below is necessary: without it 1D array X will cause errors.
        X_ext[order1] = X.reshape(L, d)
        Y = np.sqrt(w) * dct(X_ext, axis=0, norm="ortho")  # DCT on each column of X
        return Y[order0] / np.sqrt(n)

    def AY(Y):
        assert Y.shape[0] == n, "Y must have n rows"
        Y_ext = np.zeros((w, d))
        Y_ext[order0] = Y.reshape(n, d)
        X_ext = np.sqrt(w) * idct(Y_ext, axis=0, norm="ortho")  # IDCT on each col. of Y
        return X_ext[order1] / np.sqrt(n)

    return AX, AY