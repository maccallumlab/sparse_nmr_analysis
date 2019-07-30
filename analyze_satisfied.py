import restraints
import mdtraj as md
import meld.vault as vault
import numpy as np
import os
import multiprocessing
import functools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fraction', type=float)
args = parser.parse_args()
fraction = args.fraction


RESTRAINT_FILES = [
    "c13_restraints.dat",
    "n15_restraints.dat",
    "cc_restraints.dat",
    "cn_restraints.dat",
    "nc_restraints.dat",
    "nn_restraints.dat",
]

reference = md.load("renumbered.pdb")

data = vault.DataStore.load_data_store()
data.initialize("r")
simulation = data.load_system()

missing = restraints.find_missing_residues(reference, simulation)

collections = []
for fn in RESTRAINT_FILES:
    if os.path.exists(fn):
        collection = restraints.load(fn, fraction, missing, reference, simulation)
        collections.append(collection)
system = restraints.System(collections)

# Calculate the stats for the reference structure
n_good_ref, n_bad_ref = system.calc_satisfied(reference.xyz[0, :], use_reference=True)
print()
print(
    f"Reference structure satisfies {n_good_ref} good and {n_bad_ref} bad restraints."
)
print()

# Get our permutation vectors, these start at frame 1.
traces = restraints.get_traces(data)

xs = []
ys = []
us = []
vs = []
indices = list(range(traces.shape[1]))
pool = multiprocessing.Pool()
func = functools.partial(restraints.get_transitions, traces=traces, system=system)
results = pool.map(func, indices)

for x, y, u, v in results:
    xs.extend(x)
    ys.extend(y)
    us.extend(u)
    vs.extend(v)

results = np.vstack([xs, ys, us, vs]).T
np.save("transitions.npy", results)