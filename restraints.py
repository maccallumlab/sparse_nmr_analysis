import meld.vault as vault
import mdtraj as md
from collections import namedtuple
import functools
import numpy as np
import uuid
import os


NOE_DIST = 0.45
NOE_FUDGE = 0.15
TOLERANCE = 0.2


class Restraint:
    def __init__(self, upper_bound, nmr_i, nmr_j, ref_i, ref_j, is_good):
        self.upper_bound = upper_bound
        self.nmr_i = nmr_i
        self.nmr_j = nmr_j
        self.ref_i = ref_i
        self.ref_j = ref_j
        self.is_good = is_good

    def __eq__(self, other):
        return self._members() == other._members()

    def __hash__(self):
        return hash(self._members())

    def _members(self):
        return (
            self.upper_bound,
            self.nmr_i,
            self.nmr_j,
            self.ref_i,
            self.ref_j,
            self.is_good,
        )

    def is_satisfied(self, structure, use_reference):
        if use_reference:
            coords_i = structure[self.ref_i, :]
            coords_j = structure[self.ref_j, :]
        else:
            coords_i = structure[self.nmr_i, :]
            coords_j = structure[self.nmr_j, :]
        dist = np.linalg.norm(coords_i - coords_j)
        return dist < self.upper_bound


class Group:
    def __init__(self, restraints):
        self.restraints = restraints

    def __repr__(self):
        return f"RestraintGroup with {len(self.restraints)} restraints."

    def calc_good_bad_satisfied(self, structure, use_reference):
        sat = [rest.is_satisfied(structure, use_reference) for rest in self.restraints]
        good = [rest.is_good for rest in self.restraints]
        good_sat = any((s and g) for s, g in zip(sat, good))

        if good_sat:
            return (True, False)
        elif any(sat):
            return (False, True)
        else:
            return (False, False)


class Collection:
    def __init__(self, groups, n_active):
        self.groups = groups
        self.n_active = n_active

    def calc_satisfied(self, structure, use_reference):
        sats = [
            g.calc_good_bad_satisfied(structure, use_reference) for g in self.groups
        ]
        n_good = sum(1 if s[0] else 0 for s in sats)
        n_bad = sum(1 if s[1] else 0 for s in sats)

        if n_good + n_bad <= self.n_active:
            return n_good, n_bad
        elif n_good >= self.n_active:
            return self.n_active, 0
        else:
            return n_good, self.n_active - n_good


class System:
    def __init__(self, collections):
        self.collections = collections

    def calc_satisfied(self, structure, use_reference):
        total_good = 0
        total_bad = 0
        for collection in self.collections:
            n_good, n_bad = collection.calc_satisfied(structure, use_reference)
            total_good += n_good
            total_bad += n_bad
        return total_good, total_bad

    def get_all_satisfied(self, indices, data):
        results = []
        for t, index in enumerate(indices):
            # Time starts from 1, not 0
            t = t + 1
            try:
                coords = data.load_positions_random_access(t)[index, :, :] / 10.0
            except:
                print(
                    f"Exception occured while loading positions for walker {index} at time {t}."
                )
                raise
            results.append(self.calc_satisfied(coords, use_reference=False))
        return results


def find_missing_residues(reference, system):
    reference_residues = {r.resSeq for r in reference.topology.residues}
    system_residues = {num for num in system.residue_numbers}
    missing = system_residues - reference_residues
    return missing


def load(filename, active_fraction, missing, reference, system):
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]

    dists = []
    rest_group = set()

    for line in lines:
        if not line:
            # process the group here
            if rest_group:
                dists.append(Group(rest_group))
            rest_group = set()
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = NOE_DIST

            # Skip over any restraints that correspond to residues that are missing
            # in the reference.
            if i in missing:
                print(f"Residue {i} is missing from reference, skipping.")
                continue
            if j in missing:
                print(f"Residue {j} is missing from reference, skipping.")
                continue

            # Map the restraints onto heavy atoms.
            name_i, dist = hydrogen_to_heavy(name_i, dist)
            name_j, dist = hydrogen_to_heavy(name_j, dist)

            # Find the indices.
            try:
                nmr_i = system.index_of_atom(i, name_i)
                nmr_j = system.index_of_atom(j, name_j)
                ref_i = ref_lookup(reference, i, name_i)
                ref_j = ref_lookup(reference, j, name_j)
            except:
                print(f"Failed to find {i} {name_i} or {j} {name_j}. Skipping.")
                continue

            # Decide if this is a good restraint or not.
            coords_i = reference.xyz[0, ref_i]
            coords_j = reference.xyz[0, ref_j]
            if np.linalg.norm(coords_i - coords_j) < NOE_DIST + TOLERANCE:
                good = True
            else:
                good = False

            rest = Restraint(dist, nmr_i, nmr_j, ref_i, ref_j, good)
            rest_group.add(rest)
    n_active = int(len(dists) * active_fraction)
    return Collection(dists, n_active)


hmap = {
    "HG11": "CG1",
    "HG12": "CG1",
    "HG13": "CG1",
    "HG21": "CG2",
    "HG22": "CG2",
    "HG23": "CG2",
    "HD11": "CD1",
    "HD12": "CD1",
    "HD13": "CD1",
    "HD21": "CD2",
    "HD22": "CD2",
    "HD23": "CD2",
    "H": "N",
}


def hydrogen_to_heavy(name, dist):
    return hmap[name], dist + NOE_FUDGE


class Lookup:
    def __init__(self):
        self._cache = {}

    def __call__(self, ref, residue, name):
        if (residue, name) in self._cache:
            return self._cache[(residue, name)]
        else:
            result = ref.topology.select(f"residue {residue} and name {name}")[0]
            self._cache[(residue, name)] = result
            return result


ref_lookup = Lookup()


def get_traces(store):
    vecs = load_permutation_vectors(store, None, None)
    traces = deshuffle_traces(vecs)

    n_replicas = traces.shape[1]
    n_steps = traces.shape[0]

    current_index = np.array(list(range(n_replicas)))

    results = []
    for step in range(n_steps):
        new_value = np.zeros_like(current_index)
        new_value[traces[step, :]] = current_index
        results.append(new_value)
    results = np.array(results)

    return results


def load_permutation_vectors(store, start, end):
    if start is None:
        start = 1
    if end is None:
        end = store.max_safe_frame - 1

    perm_vecs = np.zeros((store.n_replicas, end - start), dtype=int)

    for index, frame in enumerate(range(start, end)):
        perm_vecs[:, index] = store.load_permutation_vector(frame)

    return perm_vecs


def deshuffle_traces(perm_vecs):
    n_replicas = perm_vecs.shape[0]
    n_steps = perm_vecs.shape[1]

    results = []
    current_indices = np.array(list(range(n_replicas)))

    for i in range(n_steps):
        current_indices = current_indices[perm_vecs[:, i]]
        results.append(current_indices)
    return np.array(results)


def calc_rmsds(indices, data):
    selection_string = open("selection_string.txt").read().strip()

    # load the reference
    ref = md.load("renumbered.pdb")
    ref_ind = ref.topology.select(selection_string)

    # setup a template to load coordinates into
    system = data.load_system()
    coordinates = data.load_positions_random_access(0)[0, :, :]
    pdb_writer = system.get_pdb_writer()
    filename = uuid.uuid4().hex + ".pdb"
    with open(filename, "w") as outfile:
        outfile.write(pdb_writer.get_pdb_string(coordinates, 0))
        outfile.flush()
    template = md.load(filename)
    os.unlink(filename)
    traj_ind = template.topology.select(selection_string)

    if len(ref_ind) == len(traj_ind):
        print("Reference and trajectory indices differ in length.")
        print(ref_ind)
        print(traj_ind)
        raise ValueError("Reference and trajectory indices differ in length.")

    # calculate RMSDs
    rmsds = []
    for frame, index in enumerate(indices):
        coords = (
            data.load_positions_random_access(frame)[index, :, :] / 10.0
        )  # Angstrom to nm
        template.xyz[0, :, :] = coords
        rmsd = md.rmsd(template, ref, atom_indices=traj_ind, ref_atom_indices=ref_ind)[
            0
        ]
        rmsds.append(rmsd * 10.0)  # nm to Angstrom

    return np.array(rmsds)


def get_transitions(walker_index, traces, system, lag=5):
    data = vault.DataStore.load_data_store()
    data.initialize("r")
    n_stages = traces.shape[0]

    # get all replica indices for our walker
    replica_indices = traces[:, walker_index]

    # get all RMSDs for our walker
    rmsds = calc_rmsds(replica_indices, data)

    # get all restraints satisfied
    satisfied = system.get_all_satisfied(traces[:, walker_index], data)

    # store all results
    all_results = AllResult(
        list(range(len(satisfied))),
        [item[0] for item in satisfied],
        [item[1] for item in satisfied],
        replica_indices,
        rmsds,
    )

    # store only transitions
    s = []
    t = []
    u = []
    v = []
    w = []
    x = []
    y = []
    z = []
    for i, j in zip(range(n_stages), range(lag, n_stages)):
        if traces[i, walker_index] > traces[j, walker_index]:
            print(
                walker_index,
                i,
                j,
                traces[i, walker_index],
                traces[j, walker_index],
                satisfied[i],
                " -> ",
                satisfied[j],
            )
            s.append(satisfied[i][0])
            t.append(satisfied[i][1])
            u.append(satisfied[j][0])
            v.append(satisfied[j][1])
            w.append(traces[i, walker_index])
            x.append(traces[j, walker_index])
            y.append(rmsds[i])
            z.append(rmsds[j])
    transition_results = TransitionResult(s, t, u, v, w, x, y, z)
    return transition_results, all_results


TransitionResult = namedtuple(
    "TransitionResult",
    "good_start bad_start good_end bad_end rep_start rep_end rmsd_start rmsd_end",
)
AllResult = namedtuple("AllResult", "frame good bad index rmsd")
