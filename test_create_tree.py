import time
import gates as gate
from utils import create_tree, save_tree

basis = [gate.su2(gate.H), gate.su2(gate.T)]
depth = 15

t = time.time()
tree = create_tree(basis, max_depth=depth)
print('Elapsed time:', time.time() - t)
print('Number of gates:', len(tree['names']))

save_tree(tree, 'trees/HT_{}.pkl'.format(depth))
