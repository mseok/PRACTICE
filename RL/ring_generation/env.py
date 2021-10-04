from copy import deepcopy

import torch
import rdkit
from rdkit import Chem

from utils import seed_all

""" Global Variables """
ATOM_SYMBOLS = ["C", "N", "O"]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
N_ATOM_FEATURES = len(ATOM_SYMBOLS) + len(ATOM_DEGREES)


def one_of_k_encoding(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set: {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def check_valence(mol: Chem.Mol) -> bool:
    """
    Check the validity of the molecule valence.

    Arguments:
        mol: Molecule to check valence validity.
    Returns:
        Boolean whether valence is ok or not.
    """
    problems = Chem.DetectChemistryProblems(mol)
    problems = [problem.GetType() for problem in problems]
    valence_problems = [
        isinstance(problem, Chem.AtomValenceException) for problem in problems
    ]
    return len(valence_problems) == 0


def check_added(prev_mol: Chem.Mol, curr_mol: Chem.Mol) -> bool:
    """
    Check whether atom or bond added to the molecule.

    Arguments:
        prev_mol: Previous molecule.
        curr_mol: Previous molecule.
    Returns:
        Boolean whether addition happened or not.
    """
    prev_count = prev_mol.GetNumAtoms() + prev_mol.GetNumBonds()
    curr_count = curr_mol.GetNumAtoms() + curr_mol.GetNumBonds()
    return prev_count < curr_count


def check_chemical_validity(mol: Chem.Mol) -> bool:
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Arguments:
        mol: Molecule for check validity
    Returns:
        True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s)
    return m is not None


class MoleculeEnv:
    def __init__(self, natoms: int = 5, max_action: int = 5):
        """Only single bond can be added."""
        self.max_action = max_action
        self.natoms = natoms
        self.nodes = [i for i in range(natoms)]
        self.initialize()

    def initialize(self):
        """
        Reset the environment.
        """
        self.mol = Chem.RWMol()
        for _ in range(self.natoms):
            self.mol.AddAtom(Chem.Atom("C"))
        self.action_count = 0
        return None

    def seed(self, seed: int = 0) -> None:
        """Set the seed."""
        seed_all(seed)
        return None

    def step(self, action):
        """Add single bond between two atoms."""
        start, end = action
        prev_mol = deepcopy(self.mol)
        self.__add_bond(start, end, Chem.BondType.SINGLE)
        Chem.SanitizeMol(self.mol)
        reward = self.__intermediate_reward(prev_mol)
        # Finish condition
        done = self.action_count >= self.max_action
        # print("action count: ", self.action_count, self.max_action)
        self.action_count += 1
        if done:
            final_reward = self.__final_reward()
            reward += final_reward
        return reward, done

    def __intermediate_reward(self, prev_mol):
        valid_valence = check_valence(self.mol)
        added = check_added(prev_mol, self.mol)
        if not valid_valence:
            reward = -1
            self.mol = prev_mol
        elif added:
            reward = 2
        else:
            reward = -3
        return reward

    def __final_reward(self):
        valid = check_chemical_validity(self.mol)
        ringinfo = self.mol.GetRingInfo()
        # We only create single ring here
        rings = ringinfo.AtomRings()
        if not valid:
            reward = -1
        elif len(rings) > 0 and len(rings[0]) >= 3:
            reward = 3
        else:
            reward = -3
        return reward

    def get_observation(self):
        node_features = []  # Currently all atoms are same
        for atom in self.mol.GetAtoms():
            symbol = one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOLS)
            degree = one_of_k_encoding(atom.GetDegree(), ATOM_DEGREES)
            features = torch.Tensor(symbol + degree)
            node_features.append(features)
        node_features = torch.stack(node_features).float()
        adj = torch.Tensor(Chem.GetAdjacencyMatrix(self.mol)).float()
        return node_features, adj

    def __add_bond(self, start: int, end: int, bond: Chem.BondType):
        if not self.mol.GetBondBetweenAtoms(start, end):
            self.mol.AddBond(start, end, bond)
        return None
