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

s = []
t = []
u = []
v = []
w = []
x = []
y = []
z = []
indices = list(range(traces.shape[1]))
pool = multiprocessing.Pool()
func = functools.partial(restraints.get_transitions, traces=traces, system=system)
results = pool.map(func, indices)

transition_results = [result[0] for result in results]
all_results = [result[1] for result in results]

# save the transition results
for result in  transition_results:
    s.extend(result.good_start)
    t.extend(result.bad_start)
    u.extend(result.good_end)
    v.extend(result.bad_end)
    w.extend(result.rep_start)
    x.extend(result.rep_end)
    y.extend(result.rmsd_start)
    z.extend(result.rmsd_end)
results = np.vstack([s, t, u, v, w, x, y, z]).T
np.save("transitions.npy", results)

frames = []
indices = []
goods = []
bads = []
rmsds = []
for result in all_results:
    frames.extend(result.frame)
    indices.extend(result.index)
    goods.extend(result.good)
    bads.extend(result.bad)
    rmsds.extend(result.rmsd)

results = np.vstack([frames, indices, goods, bads, rmsds]).T
np.save("all_results.npy", results)