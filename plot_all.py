import numpy as np
import scipy.stats
from matplotlib import pyplot as pp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("start_frame", type=int)
args = parser.parse_args()

FRAME = 0
INDEX = 1
GOOD = 2
BAD = 3
RMSD = 4
results = np.load(args.filename)

ind = results[:, FRAME] > args.start_frame
pp.scatter(results[ind, RMSD], results[ind, INDEX], alpha=0.3)
pp.xlim(0, 25)
pp.show()