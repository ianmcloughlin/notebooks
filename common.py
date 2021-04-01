# Numerical arrays.
import numpy as np
# Combinatorics.
import itertools as it
# Symbolic computation.
import sympy as sm


def code_from_generator(G, mod=2):
    words = np.array(allwords(G.shape[0], mod=mod))
    code = np.matmul(words, G) % mod
    return code

def code_from_check(H, mod=2):
    words = np.array(allwords(H.shape[1], mod=mod))
    bools = ~np.any(np.matmul(H, words.T) % mod, axis=0)
    return words[bools]

def display(M):
    """Display a matrix."""
    L = max([len(j) for i in M.astype(str) for j in i])
    for r in M:
        strs = ['.' if i == 0 else str(i) for i in r]
        strs = [((" " * L) + s)[-L:] for s in strs]
        print(" ".join(strs))
        
def allwords(n, mod=2):
    return np.array(list(it.product(range(mod), repeat=n)))

def mod(x, m):
    """Dealing with fractions: x modulo m."""
    num, den = x.as_numer_denom()
    return num * sm.mod_inverse(den, m) % m

def brref(M):
    """Reduced row echelon form of a binary matrix."""
    # Sympyify M.
    S = sm.Matrix(M)
    # Zero function for binary.
    def iszerofunc(x): return ((x % 2) == 0)
    # Reduced row echelon form.
    R = S.rref(pivots=False, iszerofunc=iszerofunc)
    # Deal with fractions.
    R = R.applyfunc(lambda x: mod(x, 2))
    # Numpyify and return.
    return np.array(R.tolist(), dtype=np.uint8)

def deletez(M):
    """Remove zero rows of M."""
    return M[~np.all(M == 0, axis=1)]

def paritybit(M):
    """Append parity bit to rows of M."""
    return np.hstack([M, (M.sum(axis=1) % 2).reshape(-1,1)])

def rcirculant(r1):
    """Reverse circulant matrix with first row r1"""
    return [lcycle(r1, i) for i in range(len(r1))]

def lcycle(r, i=1):
    """Left cycle the list r by i places."""
    return r[i:] + r[:i]