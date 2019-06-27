import numpy as np
import pickle as pkl
from sklearn.neighbors import KDTree
import gates as gate
import qutip as qt

# List multiplication
def multiply(arr):
    A = arr[0]
    for i in range(1, len(arr)):
        A = A * arr[i]
    return A


# Determinant
def det(A):
    return np.linalg.det(A.full())


# Diagonalization
def diagonalize(A):
    d, V = np.linalg.eig(A.full())
    return d, qt.Qobj(V)


# Group Commutator Decomposition
def gcd(U):
    theta = 2 * np.arccos(np.real(U.tr() / 2))
    phi = 2 * np.arcsin(np.sqrt(np.sqrt((0.5 - 0.5 * np.cos(theta / 2)))))

    print('---', U.isunitary)

    axis, angle = gate.bloch(U)
    V = gate.Rx(phi)
    if axis[2] < 0:
        W = gate.Ry(2 * np.pi - phi)
    else:
        W = gate.Ry(phi)

    _, V1 = diagonalize(U)
    _, V2 = diagonalize(V * W * V.dag() * W.dag())
    S = V1 * V2.dag()
    V_tilde = S * V * S.dag()
    W_tilde = S * W * S.dag()
    return V_tilde, W_tilde


# Creating tree
def create_tree(basis, max_depth=10):
    def array_increment(arr, dims):
        for i in reversed(range(len(arr))):
            arr[i] += 1
            if arr[i] < dims[i]:
                break
            else:
                arr[i] = 0
        return arr

    n = len(basis)
    X = np.array([[1, 0, 0, 0]])
    eps = 1e-10
    names = ['']
    templates = []
    for i in range(1, max_depth + 1):
        arr = [0] * i
        dims = [n] * i
        print('Creating tree: {0}/{1}'.format(i, max_depth))
        for j in range(n ** i):
            name = ''.join(str(x) for x in arr)
            if not any(t in name for t in templates):
                U = multiply([basis[x] for x in arr]).full()
                v = np.array([np.real(U[0, 0]), np.imag(U[0, 0]), np.real(U[1, 0]), np.imag(U[1, 0])])
                if abs(abs(v[0]) - 1) < eps or abs(abs(v[1]) - 1) < eps:
                    templates.append(name)
                else:
                    X = np.vstack([X, v])
                    names.append(name)
            arr = array_increment(arr, dims)
    print('Creating tree')
    tree = KDTree(X, metric='euclidean')
    return {'tree': tree, 'names': names, 'basis': basis}


def save_tree(tree, filename):
    with open(filename, 'wb') as f:
        pkl.dump(tree, f, pkl.HIGHEST_PROTOCOL)


def load_tree(filename):
    with open(filename, 'rb') as f:
        tree = pkl.load(f)
    return tree