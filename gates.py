import qutip as qt
import numpy as np
from utils import det

# 1 qubit gates
I = qt.qeye(2)
X = qt.sigmax()
Y = qt.sigmay()
Z = qt.sigmaz()
H = qt.hadamard_transform()
S = qt.phasegate(np.pi / 2)
T = qt.phasegate(np.pi / 4)
SQNOT = qt.sqrtnot()

def Rx(angle):
    return qt.rx(angle)

def Ry(angle):
    return qt.ry(angle)

def Rz(angle):
    return qt.rz(angle)

def R(axis, angle):
    angle = np.remainder(angle, 2 * np.pi)
    if not (type(axis) is np.ndarray):
        axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    return qt.Qobj(np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * (axis[0] * X + axis[1] * Y + axis[2] * Z))

def Phase(angle):
    return qt.phasegate(angle)

# 2 qubit gates
CNOT = qt.cnot()
CZ = qt.csign()
Berkeley = qt.berkeley()
SWAP = qt.swap()
iSWAP = qt.iswap()
SQSWAP = qt.sqrtswap()
SQiSWAP = qt.sqrtiswap()

def aSWAP(angle):
    return qt.swapalpha(angle)

# 3 qubit gates
Fredkin = qt.fredkin()
Toffoli = qt.toffoli()



# Get unitary axis and angle
def bloch(U):
    if isinstance(U, qt.Qobj):
        U = U.full()
    angle = np.real(2 * np.arccos(np.trace(U) / 2))
    sin = np.sin(angle / 2)
    eps = 1e-10
    if sin < eps:
        axis = [0, 0, 1]
    else:
        nz = np.imag(U[1, 1] - U[0, 0]) / (2 * sin)
        nx = -np.imag(U[1, 0]) / sin
        ny = np.real(U[1, 0]) / sin
        axis = [nx, ny, nz]
    return axis, angle

# Create SU2 operator
def su2(U):
    t = 0.0j + det(U)
    phase = np.sqrt(1 / t)
    return phase * U
