import numpy as np
import scipy.stats
from matplotlib import pyplot as pp

results = np.load("transitions.npy")
max_ = max(np.max(results[:, 0]), np.max(results[:, 1]))

means, x_edges, y_edges, _ = scipy.stats.binned_statistic_2d(
    results[:, 0],
    results[:, 1],
    [results[:, 2], results[:, 3]],
    statistic="mean",
    bins=20,
    range=([0, max_], [0, max_]),
)
counts, _, _, _ = scipy.stats.binned_statistic_2d(
    results[:, 0],
    results[:, 1],
    results[:, 2],
    statistic="count",
    bins=20,
    range=([0, max_], [0, max_]),
)
alphas = counts / np.max(counts)
alphas = alphas.T.ravel()
n = alphas.shape[0]
color = np.zeros((n, 4))
# color[:, 0] = 1.0 - alphas
# color[:, 1] = 1.0 - alphas
# color[:, 2] = 1.0 - alphas

xs = [(a + b) / 2.0 for a, b in zip(x_edges, x_edges[1:])]
ys = [(a + b) / 2.0 for a, b in zip(y_edges, y_edges[1:])]

pp.quiver(xs, ys, means[0, :, :].T, means[1, :, :].T, alpha=1.0, color=color)
# pp.quiver(results[:, 0], results[:, 1], results[:, 2], results[:, 3])
# pp.imshow(counts, extent=[0, max_, 0, max_])
pp.xlabel("# Good Satisfied")
pp.ylabel("# Bad Satisfied")
# pp.xlim(0, max_)
# pp.ylim(0, max_)
pp.show()
