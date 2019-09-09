import numpy as np
import scipy.stats
from matplotlib import pyplot as pp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()


BINS = 50
SCALE = 3
DENSITY_THRESH = 50

# Columns: good_start bad_start good_end bad_end index_start index_end rmsd_start rmsd_end
GOOD_START = 0
BAD_START = 1
GOOD_END = 2
BAD_END = 3
INDEX_START = 4
INDEX_END = 5
RMSD_START = 6
RMSD_END = 7

results = np.load(args.filename)

x_max = np.max(results[:, GOOD_START])
y_max = np.max(results[:, BAD_START])
max_ = max(x_max, y_max)

means, x_edges, y_edges, _ = scipy.stats.binned_statistic_2d(
    results[:, GOOD_START],
    results[:, BAD_START],
    [results[:, GOOD_END], results[:, BAD_END]],
    statistic="mean",
    bins=BINS,
    range=([0, max_], [0, max_]),
)
counts, _, _, _ = scipy.stats.binned_statistic_2d(
    results[:, GOOD_START],
    results[:, BAD_START],
    results[:, GOOD_END],
    statistic="count",
    bins=BINS,
    range=([0, max_], [0, max_]),
)

counts[counts > DENSITY_THRESH] = DENSITY_THRESH
alphas = counts / np.max(counts)
alphas = alphas.T.ravel()
n = alphas.shape[0]
color = np.zeros((n, 4))
color[:, 0] = 1.0 - alphas
color[:, 1] = 1.0 - alphas
color[:, 2] = 1.0 - alphas

xs = [(a + b) / 2.0 for a, b in zip(x_edges, x_edges[1:])]
ys = [(a + b) / 2.0 for a, b in zip(y_edges, y_edges[1:])]

pp.quiver(
    xs,
    ys,
    means[0, :, :].T,
    means[1, :, :].T,
    alpha=1.0,
    color=color,
    angles="xy",
    scale_units="xy",
    scale=SCALE,
)

pp.xlabel("# Good Satisfied")
pp.ylabel("# Bad Satisfied")
pp.xlim(0, x_max)
pp.ylim(0, y_max)
pp.show()
