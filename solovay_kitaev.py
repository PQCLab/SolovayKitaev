import gates as gate
import numpy as np
from utils import multiply, gcd

# Solovay-Kitaev approximation algorithm
def solovay_kitaev(U, tree, n):
    def skfunc(U, n):
        if n == 0:
            M = U.full()
            v = np.array([[np.real(M[0, 0]), np.imag(M[0, 0]), np.real(M[1, 0]), np.imag(M[1, 0])]])
            dist, index = tree['tree'].query(v, k=1)
            name = tree['names'][index[0, 0]]
            if name == '':
                return gate.I
            else:
                basis = tree['basis']
                return multiply([basis[int(x)] for x in name])
        else:
            U_next = skfunc(U, n - 1)
            V, W = gcd(U * U_next.dag())
            V_next = skfunc(V, n - 1)
            W_next = skfunc(W, n - 1)
            return V_next * W_next * V_next.dag() * W_next.dag() * U_next

    return skfunc(U, n)