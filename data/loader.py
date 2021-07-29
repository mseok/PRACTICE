import os
import pickle

import numpy as np
from rdkit import Chem
from scipy.constants import physical_constants
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# GLOBAL VARIABLES
ATOM_SYMBOLS = ["H", "C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
ATOM_DEGREE = [0, 1, 2, 3, 4, 5]
ATOM_HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
ATOM_FORMAL_CHARGE = [-2, -1, 0, 1, 2, 3, 4]
N_NODE_FEATURES = (
    len(ATOM_SYMBOLS)
    + len(ATOM_DEGREE)
    + len(ATOM_HYBRIDIZATION)
    + len(ATOM_FORMAL_CHARGE)
    + 1
)

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
    0,
]
BOND_STEREOTYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
N_EDGE_FEATURES = len(BOND_TYPES) + len(BOND_STEREOTYPES)
UNIT_CONVERSION = [
    "homo",
    "lumo",
    "homo-lumo-gap",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
]
PROPERTIES = [
    "A",
    "B",
    "C",
    "dipole-moment",
    "isotropic-polarizability",
    "homo",
    "lumo",
    "homo-lumo-gap",
    "electronic-spatial-extent",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]
HARTREE2EV = physical_constants["hartree-electron volt relationship"][0]
DTYPE = np.float32
DTYPE_INT = np.int32


def get_mol_feature(m):
    H = []
    n = m.GetNumAtoms()
    for i in range(n):
        H.append(get_atom_feature(m, i))
    H = np.array(H, dtype="uint8")
    return H


def get_atom_feature(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), ATOM_DEGREE)
        + one_of_k_encoding_unk(atom.GetHybridization(), ATOM_HYBRIDIZATION)
        + one_of_k_encoding_unk(atom.GetFormalCharge(), ATOM_FORMAL_CHARGE)
        + [atom.GetIsAromatic()]
    )


def get_edge_feature(molecule, extra_features=True):
    n_bond_features = 5
    n_extra_bond_features = 6

    n_atoms = molecule.GetNumAtoms()
    E = np.zeros((n_atoms, n_atoms, n_bond_features + n_extra_bond_features))
    for i in range(n_atoms):
        for j in range(n_atoms):
            e_ij = molecule.GetBondBetweenAtoms(i, j)  # rdkit.Chem.Bond
            if e_ij is not None:
                e_ij = get_bond_feature(
                    e_ij, extra_features
                )  # ADDED edge feat; one-hot vector
                e_ij = list(
                    map(lambda x: 1 if x else 0, e_ij)
                )  # ADDED edge feat; one-hot vector
                E[i, j, :] = np.array(e_ij)
    return E  # (N, N, 11)


def get_bond_feature(bond, include_extra=False):
    bt = bond.GetBondType()  # rdkit.Chem.BondType
    retval = one_of_k_encoding_unk(bt, BOND_TYPES)
    if include_extra:
        bs = bond.GetStereo()
        retval += one_of_k_encoding_unk(bs, BOND_STEREOTYPES)
    return np.array(retval)


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def mol_to_sample(mol):
    mol = Chem.AddHs(mol)
    natoms = mol.GetNumAtoms()

    h = get_mol_feature(mol).astype(DTYPE_INT)
    e = get_edge_feature(mol).astype(DTYPE_INT)
    adj = Chem.GetAdjacencyMatrix(mol) + np.eye(natoms)
    adj = adj.astype(DTYPE_INT)
    valid = np.ones((natoms,)).astype(DTYPE_INT)

    sample = {
        "h": h,
        "e": e,
        "adj": adj,
        "valid": valid,
    }
    return sample


class QM9Dataset(Dataset):
    """QM9 dataset."""

    def __init__(self, keys: list, data_dir: str, task: str = "homo"):
        self.keys = keys
        self.data_dir = data_dir
        self.task = task

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        fn = os.path.join(self.data_dir, key)
        prop, pos, mol = self._read_pickle(fn=fn, task=self.task)
        sample = mol_to_sample(mol)
        sample["pos"] = pos
        sample["y"] = prop
        sample["keys"] = key
        return sample

    def _read_pickle(self, fn, task):
        with open(fn, "rb") as f:
            features, pos, mol = pickle.load(f)
        prop = np.array([features[PROPERTIES.index(task)]]).astype(DTYPE)
        if task in UNIT_CONVERSION:
            prop = prop * HARTREE2EV
        return prop, pos, mol


def check_dimension(tensors):
    size = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            size.append(tensor.shape)
        else:
            size.append(0)
    size = np.asarray(size)
    return np.max(size, 0)


def collate_tensor(tensor, max_tensor, batch_idx):
    if isinstance(tensor, np.ndarray):
        dims = tensor.shape
        slice_list = tuple([slice(0, dim) for dim in dims])
        slice_list = [slice(batch_idx, batch_idx + 1), *slice_list]
        max_tensor[tuple(slice_list)] = tensor
    elif isinstance(tensor, str):
        max_tensor[batch_idx] = tensor
    else:
        max_tensor[batch_idx] = tensor
    return max_tensor


def tensor_collate_fn(batch):
    batch_items = [it for e in batch for it in e.items()]
    dim_dict = dict()
    total_key, total_value = list(zip(*batch_items))
    batch_size = len(batch)
    n_elem = int(len(batch_items) / batch_size)
    total_key = total_key[0:n_elem]
    for i, k in enumerate(total_key):
        value_list = [v for j, v in enumerate(total_value) if j % n_elem == i]
        if isinstance(value_list[0], np.ndarray):
            dim_dict[k] = np.zeros(
                np.array([batch_size, *check_dimension(value_list)]),
            )
        elif isinstance(value_list[0], str):
            dim_dict[k] = ["" for _ in range(batch_size)]
        else:
            dim_dict[k] = np.zeros((batch_size,))
    ret_dict = {}
    for j in range(batch_size):
        if batch[j] is None:
            continue
        for key, value in dim_dict.items():
            value = collate_tensor(batch[j][key], value, j)
            if not isinstance(value, list):
                value = torch.from_numpy(value).float()
            ret_dict[key] = value
    return ret_dict


def get_data(keys, ngpus_per_node, shuffle, FLAGS):
    dataset = QM9Dataset(keys, FLAGS.data_dir, FLAGS.task)
    sampler = DistributedSampler(
        dataset,
        num_replicas=ngpus_per_node,
        shuffle=shuffle
    ) if FLAGS.is_distributed else None
    loader = DataLoader(
        dataset,
        FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        pin_memory=(sampler is not None),
        collate_fn=tensor_collate_fn,
        shuffle=shuffle and (sampler is None),
        sampler=sampler,
    )
    return dataset, loader, sampler
