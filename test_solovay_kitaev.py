import qutip as qt
import gates as gate
import numpy as np
from solovay_kitaev import solovay_kitaev
from utils import load_tree

U = gate.R([0.21, 0.14, 0.7], 7 * np.pi / 6)

tree = load_tree('trees/HT_15.pkl')

U_approx = solovay_kitaev(U, tree, 2)

print(U.full())
print(U_approx.full())
print(qt.tracedist(U, U_approx))