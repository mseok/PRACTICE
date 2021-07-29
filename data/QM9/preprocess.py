import glob
from multiprocessing import Pool
import os
import pickle
import sys

from rdkit import Chem
from rdkit import RDLogger
import numpy as np

RDLogger.DisableLog("*.rdApp")
PICKLE_PATH = "pickle"


def preprocess(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        features = lines[1][2:]
        features = list(map(float, features))

        pos = [line[1:-1] for line in lines[2:-3]]
        pos = np.array(pos).astype(np.float)

        smiles_list = lines[-2]
        try:
            mol = Chem.MolFromSmiles(smiles_list[1])
        except Exception:
            mol = Chem.MolFromSmiles(smiles_list[0])
        if mol is None:
            return

        pickle_fn = fn.split("_")[-1].split(".")[0] + ".pkl"
        with open(os.path.join(PICKLE_PATH, pickle_fn), "wb") as w:
            pickle.dump([features, pos, mol], w)
    return


def worker(fn):
    try:
        preprocess(fn)
    except Exception as e:
        print(e)


def mp(fns, func, ncpu=4):
    pool = Pool(ncpu)
    results = pool.map_async(func, fns)
    pool.close()
    pool.join()
    return results


def main():
    NCPU = int(sys.argv[1])
    if not os.path.exists(PICKLE_PATH):
        os.mkdir(PICKLE_PATH)
    xyz_fns = glob.glob("./xyz_files/*.xyz")
    mp(xyz_fns, worker, NCPU)
    return


if __name__ == "__main__":
    main()
